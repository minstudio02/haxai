const {HaxAI} = require('./haxai');
const {Model} = require('./AImodel');
const {Memory} = require('./memory');
const tf = require('@tensorflow/tfjs-node');

class Orchestrator {
  /**
   * @param {HaxAI} haxai
   * @param {Model} model
   * @param {Memory} memory
   * @param {number} discountRate
   */
  constructor(haxai, model, memory, discountRate) {
      this.haxai = haxai;
      this.model = model;
      this.memory = memory;
      this.discountRate = discountRate;

      this.not_started_yet = 0
      this.own_score=0
      this.opponent_score=0
      this.lastBallDistance = Infinity;  // 이전 프레임의 볼 거리
      this.lastBallPosition = {x: 0, y: 0};  // 이전 프레임의 볼 위치

      // PPO2 하이퍼파라미터 (vani-or/haxball-ai 참고)
      this.clipEpsilon = 0.1;        // 초기 정책 클리핑 범위
      this.valueClipEpsilon = 0.2;   // 가치 클리핑 범위
      this.ppoEpochs = 4;            // PPO 업데이트 에포크 수
      this.valueCoeff = 0.5;         // 가치 손실 계수
      this.entropyCoeff = 0.01;      // 엔트로피 정규화 계수
      this.learningRate = 7e-4;      // 초기 학습률
      this.gaeLambda = 0.95;         // GAE lambda 파라미터
      this.maxGradNorm = 0.5;        // 그래디언트 클리핑
      this.adamBeta1 = 0.99;
      this.adamBeta2 = 0.999;
      this.adamEpsilon = 1e-5;
      this.epsilon = 1e-8;

      // 스케줄링 파라미터 (LinearSchedule 유사)
      this.totalScheduleUpdates = 5000;
      this.initialLearningRate = this.learningRate;
      this.minLearningRate = 1e-5;
      this.initialClipEpsilon = this.clipEpsilon;
      this.minClipEpsilon = 0.01;
      this.updateCount = 0;

      // 옵티마이저 개선
      this.optimizer = tf.train.adam(
        this.learningRate,
        this.adamBeta1,
        this.adamBeta2,
        this.adamEpsilon
      );
  }

  sleep (time) {
    return new Promise((resolve) => setTimeout(resolve, time));
  }

  // 학습률 스케줄링: LinearSchedule
  updateLearningRate() {
    this.updateCount += 1;
    const progress = Math.max(0, 1 - (this.updateCount / this.totalScheduleUpdates));
    const newLR = Math.max(this.minLearningRate, this.initialLearningRate * progress);
    if (this.optimizer && typeof this.optimizer.dispose === 'function') {
      this.optimizer.dispose();
    }
    this.learningRate = newLR;
    this.optimizer = tf.train.adam(
      this.learningRate,
      this.adamBeta1,
      this.adamBeta2,
      this.adamEpsilon
    );
    this.updateClipRange(progress);
    try {
      console.log(`[스케줄러] 업데이트 ${this.updateCount}회 - 학습률: ${this.learningRate.toExponential(2)}, clip: ${this.clipEpsilon.toFixed(3)}`);
    } catch (_) {
      // 콘솔 포맷이 실패해도 학습에는 영향 없음
    }
  }

  updateClipRange(progressOverride) {
    const progress = typeof progressOverride === 'number'
      ? progressOverride
      : Math.max(0, 1 - (this.updateCount / this.totalScheduleUpdates));
    this.clipEpsilon = Math.max(this.minClipEpsilon, this.initialClipEpsilon * progress);
  }

  async train(page) {
      const stateTensor = this.haxai.getStateTensor();
      const [action, actionLogProb, value] = this.getActionAndValue(stateTensor);
      
      await this.haxai.update(action, page);
      await this.sleep(2000/60);

      const nextState = this.haxai.getStateTensor();
      const reward = this.computeReward(nextState.arraySync()[0], this.haxai.getState());
      const done = this.haxai.isDone();

      this.memory.addSample([stateTensor, action, reward, nextState, actionLogProb, value, done]);

      if (this.memory.samples.length % this.model.batchSize == 0) {
        this.ppoUpdate();
      }
  }
  
  async test(page){
    const action = this.getActions(this.haxai.getStateTensor())[0];
    await this.haxai.update(action,page);
  }

  getActionAndValue(stateTensor) {
    return tf.tidy(() => {
      const [policyLogits, value] = this.model.predict(stateTensor);
      const actionProbs = tf.softmax(policyLogits);
      const sampled = tf.multinomial(policyLogits, 1);
      const action = sampled.dataSync()[0];
      const actionProb = Math.max(actionProbs.dataSync()[action], this.epsilon);
      const actionLogProb = Math.log(actionProb);
      const valueScalar = value.dataSync()[0];
      sampled.dispose();
      policyLogits.dispose();
      actionProbs.dispose();
      value.dispose();
      return [action, actionLogProb, valueScalar];
    });
  }
  /**
 * Get policy-network logits and the action based on state-tensor inputs.
 *
 * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
 * @returns {[tf.Tensor, tf.Tensor]}
 *   1. The logits tensor, of shape `[batchSize, 1]`.
 *   2. The actions tensor, of shape `[batchSize, 1]`.
 */
  getLogitsAndActions(inputs) {
    return tf.tidy(() => {
      const [policyLogits, value] = this.model.predict(inputs);
      if (value) value.dispose(); // 가치 출력은 사용하지 않으므로 dispose
      const actionProbs = tf.softmax(policyLogits);
      const actions = tf.multinomial(policyLogits, 1);
      policyLogits.dispose();
      return [actionProbs, actions];
    });
  }

  /**
   * Get actions based on a state-tensor input.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @param {Float32Array} inputs The actions for the inputs, with length
   *   `batchSize`.
   */
  getActions(inputs) {
    return this.getLogitsAndActions(inputs)[1].dataSync();
  }

  ppoUpdate() {
    const batch = this.memory.samples.slice(-this.model.batchSize);
    if (!batch || batch.length === 0) {
      console.warn('Empty batch in ppoUpdate, skipping');
      return;
    }

    // 입력 데이터 검증 및 정규화
    const states = tf.concat(batch.map(([state]) => state));
    const actions = tf.tensor1d(batch.map(([, action]) => action), 'int32');
    const nextStates = tf.concat(batch.map(([,,, nextState]) => nextState));
    const oldLogProbs = tf.tensor1d(batch.map(([,,,, logProb]) => logProb));
    const oldValuesArr = batch.map(([,,,,, value]) => value);
    const rewardsArr = batch.map(([,, reward]) => reward);
    const notDoneMask = batch.map(([,,,,,, done]) => done ? 0 : 1);

    // nextStates에 대한 가치 예측 (두 번째 출력이 value)
    const [nextPolicyLogits, nextValues] = this.model.predict(nextStates);
    nextPolicyLogits.dispose();
    const nextValuesFlat = tf.squeeze(nextValues);
    const nextValuesArr = nextValuesFlat.dataSync();

    const advantagesArr = new Array(batch.length);
    const returnsArr = new Array(batch.length);
    let gae = 0;
    for (let t = batch.length - 1; t >= 0; t--) {
      const mask = notDoneMask[t];
      const delta = rewardsArr[t] + this.discountRate * nextValuesArr[t] * mask - oldValuesArr[t];
      gae = delta + this.discountRate * this.gaeLambda * mask * gae;
      advantagesArr[t] = gae;
      returnsArr[t] = gae + oldValuesArr[t];
    }

    const epsilonScalar = tf.scalar(this.epsilon);
    const returnsTensor = tf.tensor1d(returnsArr);
    const returnsClipped = tf.clipByValue(returnsTensor, -50, 50);
    const advantagesTensor = tf.tensor1d(advantagesArr);
    const advMean = advantagesTensor.mean();
    const advStd = tf.sqrt(advantagesTensor.sub(advMean).square().mean().add(epsilonScalar));
    const normalizedAdvantages = advantagesTensor.sub(advMean).div(advStd.add(epsilonScalar));
    advMean.dispose();
    advStd.dispose();
    advantagesTensor.dispose();
    returnsTensor.dispose();
    const oldValues = tf.tensor1d(oldValuesArr);

    for (let epoch = 0; epoch < this.ppoEpochs; epoch++) {
      // variableGrads를 사용하기 위해 함수 내에서 loss를 계산
      const {value: lossValue, grads} = tf.variableGrads(() => tf.tidy(() => {
        const [newPolicyLogits, newValues] = this.model.predict(states);
        const newActionProbs = tf.softmax(newPolicyLogits);
        newPolicyLogits.dispose();

        const actionMask = tf.oneHot(actions, this.model.numActions);
        const logProbs = tf.log(newActionProbs.add(this.epsilon));
        const selectedLogProbs = tf.sum(tf.mul(logProbs, actionMask), 1);
        const newValuesFlat = tf.squeeze(newValues);

        const ratio = tf.exp(tf.sub(selectedLogProbs, oldLogProbs));

        const surr1 = tf.mul(ratio, normalizedAdvantages);
        const surr2 = tf.mul(
          tf.clipByValue(ratio, 1 - this.clipEpsilon, 1 + this.clipEpsilon),
          normalizedAdvantages
        );

        const actorLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

        const valueDiff = tf.sub(newValuesFlat, oldValues);
        const valueClipped = tf.add(
          oldValues,
          tf.clipByValue(valueDiff, -this.valueClipEpsilon, this.valueClipEpsilon)
        );

        const valueError1 = tf.square(tf.sub(newValuesFlat, returnsClipped));
        const valueError2 = tf.square(tf.sub(valueClipped, returnsClipped));
        const criticLoss = tf.mul(0.5, tf.mean(tf.maximum(valueError1, valueError2)));

        const entropy = tf.neg(tf.sum(tf.mul(newActionProbs, tf.log(newActionProbs.add(this.epsilon))), 1));
        const entropyLoss = tf.mean(entropy);

        const totalLoss = actorLoss
          .add(tf.mul(this.valueCoeff, criticLoss))
          .sub(tf.mul(this.entropyCoeff, entropyLoss));

        logProbs.dispose();
        selectedLogProbs.dispose();
        entropy.dispose();
        entropyLoss.dispose();
        actionMask.dispose();

        return totalLoss;
      }));

      // Global gradient norm clipping
      const gradList = Object.values(grads);
      const { globalNorm, clipCoef, oneScalar, maxGradNormScalar } = tf.tidy(() => {
        const gradSquares = gradList.map(g => tf.sum(tf.square(g)));
        const globalNormInner = tf.sqrt(tf.addN(gradSquares));
        const one = tf.scalar(1);
        const maxGrad = tf.scalar(this.maxGradNorm);
        const clip = tf.minimum(
          one,
          maxGrad.div(globalNormInner.add(epsilonScalar))
        );
        return { globalNorm: globalNormInner, clipCoef: clip, oneScalar: one, maxGradNormScalar: maxGrad };
      });
      Object.keys(grads).forEach((key) => {
        const clipped = grads[key].mul(clipCoef);
        grads[key].dispose();
        grads[key] = clipped;
      });

      this.optimizer.applyGradients(grads);
      
      const lossScalar = lossValue.dataSync()[0];
      console.log(`[PPO 학습] 에포크 ${epoch + 1}/${this.ppoEpochs} - 손실: ${lossScalar.toFixed(4)}`);
      
      // 그래디언트 텐서 정리
      Object.values(grads).forEach(grad => grad.dispose());
      lossValue.dispose();
      globalNorm.dispose();
      clipCoef.dispose();
      oneScalar.dispose();
      maxGradNormScalar.dispose();
    }

    // 텐서 정리 (PPO2 업데이트 완료 후)
    states.dispose();
    actions.dispose();
    nextStates.dispose();
    oldLogProbs.dispose();
    oldValues.dispose();
    returnsClipped.dispose();
    normalizedAdvantages.dispose();
    epsilonScalar.dispose();
    nextValuesFlat.dispose();
    nextValues.dispose();
    batch.forEach(([state,,, nextState]) => {
      state.dispose();
      nextState.dispose();
    });

    // 학습률 및 클리핑 스케줄 업데이트
    this.updateLearningRate();

    this.memory.clear();
  }

    computeReward(status,more_state) {
      let reward = 0
/*
      reward -= Math.sqrt((more_state.bot_Team == 1 ? 320 : -320 - status[8]) ** 2 + status[9] ** 2)

      reward += 0.1 * Math.sqrt((more_state.bot_Team == 1 ? 320 : -320 + status[8]) ** 2 + status[9] ** 2)
*/
      let distanza_alla_palla = Math.sqrt((status[8] - status[0]) ** 2 + (status[9] - status[1]) ** 2)
      reward -= distanza_alla_palla / 2   

/*
      function prodotto_scalare(a, b){
          let lung_a = Math.max(1e-5, lung(a))
          let lung_b = Math.max(1e-5, lung(b))
          return (a[0] * b[0] + a[1] * b[1]) / lung_a / lung_b
      }
      function lung(a){
          return Math.sqrt(a[0] ** 2 + a[1] ** 2)
      }

      let vett_palla_porta = [more_state.bot_Team == 1 ? 320 : -320 - status[8], -status[9]]
      reward += prodotto_scalare(vett_palla_porta, [status[10], status[11]])


      if (!status[13]){
          let velocita_palla = Math.sqrt(status[10] ** 2 + status[11] ** 2)
          reward -= 100 * Math.max(0.0, 0.5 - velocita_palla)
      }

      if ((more_state.bot_Team == 1) ? status[8] < status[0] : status[8] > status[0]) reward -= Math.abs(status[0] - status[8])


      if (more_state.bot_Team == more_state.start_Team && !more_state.game_State){
          reward -= 0.25 * this.not_started_yet
          this.not_started_yet += 1
      }
      else  this.not_started_yet = 0

      
      let goal_reward = 0

      if (this.own_score != more_state.score.ownTeam){
          goal_reward = 1000
          this.own_score = more_state.score.ownTeam
      }
      else if (this.opponent_score != more_state.score.opponentTeam){
          goal_reward = -200
          this.opponent_score = more_state.score.opponentTeam  
      }

      reward += goal_reward */

      return reward
  }
}
  
module.exports = { Orchestrator };