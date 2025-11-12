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
      this.own_score = 0
      this.opponent_score = 0
      this.lastBallDistance = Infinity;  // 이전 프레임의 볼 거리
      this.lastBallPosition = {x: 0, y: 0};  // 이전 프레임의 볼 위치

      // PPO2 하이퍼파라미터 (vani-or/haxball-ai 참고)
      this.clipEpsilon = 0.2;        // 정책 클리핑 범위
      this.valueClipEpsilon = 0.2;   // 가치 함수 클리핑 범위
      this.ppoEpochs = 4;            // PPO 업데이트 에포크 수
      this.miniBatchSplits = 4;      // 한 에포크당 미니배치 개수
      this.valueCoeff = 0.5;         // 가치 손실 계수
      this.entropyCoeff = 0.01;      // 엔트로피 정규화 계수
      this.learningRate = 3e-4;      // 초기 학습률 (baselines PPO2 기본값)
      this.gaeLambda = 0.95;         // GAE lambda 파라미터
      this.maxGradNorm = 0.5;        // 그래디언트 클리핑
      this.adamBeta1 = 0.99;
      this.adamBeta2 = 0.999;
      this.adamEpsilon = 1e-5;
      this.epsilon = 1e-8;

      // 스케줄링 파라미터 (LinearSchedule 유사)
      this.totalScheduleUpdates = 25000;
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
      await this.sleep(10000/60);

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
    const batch = this.memory.sample(this.model.batchSize);
    if (!batch || batch.length < this.model.batchSize) {
      console.warn('Not enough samples for PPO update, skipping');
      return;
    }

    const batchSize = batch.length;
    const miniBatchSize = Math.max(1, Math.floor(batchSize / this.miniBatchSplits));

    // 입력 텐서 구성
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
    const nextValuesArr = nextValuesFlat.arraySync();

    const advantagesArr = new Array(batchSize);
    const returnsArr = new Array(batchSize);
    let gae = 0;
    for (let t = batchSize - 1; t >= 0; t--) {
      const mask = notDoneMask[t];
      const delta = rewardsArr[t] + this.discountRate * nextValuesArr[t] * mask - oldValuesArr[t];
      gae = delta + this.discountRate * this.gaeLambda * mask * gae;
      advantagesArr[t] = gae;
      returnsArr[t] = gae + oldValuesArr[t];
    }

    const epsilonScalar = tf.scalar(this.epsilon);
    const advantagesTensor = tf.tensor1d(advantagesArr);
    const returnsTensor = tf.tensor1d(returnsArr);
    const advMean = advantagesTensor.mean();
    const advStd = tf.sqrt(
      advantagesTensor.sub(advMean).square().mean().add(epsilonScalar)
    );
    const normalizedAdvantages = advantagesTensor.sub(advMean).div(advStd.add(epsilonScalar));
    const returnsClipped = tf.clipByValue(returnsTensor, -200, 200);
    const oldValues = tf.tensor1d(oldValuesArr);

    advMean.dispose();
    advStd.dispose();
    advantagesTensor.dispose();
    returnsTensor.dispose();

    let epochLossAccumulator = 0;
    let epochBatchCounter = 0;

    for (let epoch = 0; epoch < this.ppoEpochs; epoch++) {
      const shuffledIndices = tf.util.createShuffledIndices(batchSize);

      for (let start = 0; start < batchSize; start += miniBatchSize) {
        const end = Math.min(start + miniBatchSize, batchSize);
        const idxArray = Array.from(shuffledIndices.slice(start, end));
        const idxTensor = tf.tensor1d(idxArray, 'int32');

        const statesMB = tf.gather(states, idxTensor);
        const actionsMB = tf.gather(actions, idxTensor);
        const oldLogProbsMB = tf.gather(oldLogProbs, idxTensor);
        const returnsMB = tf.gather(returnsClipped, idxTensor);
        const advMB = tf.gather(normalizedAdvantages, idxTensor);
        const oldValuesMB = tf.gather(oldValues, idxTensor);

        const { value: lossValue, grads } = tf.variableGrads(() => tf.tidy(() => {
          const [policyLogits, valuePred] = this.model.predict(statesMB);
          const logProbsAll = tf.logSoftmax(policyLogits);
          const actionMask = tf.oneHot(actionsMB, this.model.numActions);
          const selectedLogProbs = tf.sum(logProbsAll.mul(actionMask), 1);
          const ratio = tf.exp(selectedLogProbs.sub(oldLogProbsMB));
          const clippedRatio = tf.clipByValue(
            ratio,
            1 - this.clipEpsilon,
            1 + this.clipEpsilon
          );

          const surrogate1 = ratio.mul(advMB);
          const surrogate2 = clippedRatio.mul(advMB);
          const actorLoss = tf.neg(tf.mean(tf.minimum(surrogate1, surrogate2)));

          const valueFlat = tf.squeeze(valuePred, [1]);
          const valueClipped = oldValuesMB.add(
            tf.clipByValue(valueFlat.sub(oldValuesMB), -this.valueClipEpsilon, this.valueClipEpsilon)
          );
          const valueError1 = valueFlat.sub(returnsMB).square();
          const valueError2 = valueClipped.sub(returnsMB).square();
          const criticLoss = tf.mean(tf.maximum(valueError1, valueError2)).mul(0.5);

          const probs = tf.softmax(policyLogits);
          const entropy = tf.mean(
            tf.sum(probs.mul(logProbsAll), 1).neg()
          );

          const totalLoss = actorLoss
            .add(criticLoss.mul(this.valueCoeff))
            .sub(entropy.mul(this.entropyCoeff));

          return totalLoss;
        }));

        // 글로벌 그래디언트 클리핑
        const gradList = Object.values(grads);
        const { globalNorm, clipCoef, maxGradNormScalar, oneScalar } = tf.tidy(() => {
          if (gradList.length === 0) {
            return {
              globalNorm: tf.scalar(0),
              clipCoef: tf.scalar(1),
              maxGradNormScalar: tf.scalar(this.maxGradNorm),
              oneScalar: tf.scalar(1)
            };
          }
          const gradSquares = gradList.map(g => tf.sum(g.square()));
          const globalNormInner = tf.sqrt(tf.addN(gradSquares).add(this.epsilon));
          const one = tf.scalar(1);
          const maxGrad = tf.scalar(this.maxGradNorm);
          const clip = tf.minimum(one, maxGrad.div(globalNormInner));
          return { globalNorm: globalNormInner, clipCoef: clip, maxGradNormScalar: maxGrad, oneScalar: one };
        });

        Object.keys(grads).forEach((key) => {
          const clipped = grads[key].mul(clipCoef);
          grads[key].dispose();
          grads[key] = clipped;
        });

        this.optimizer.applyGradients(grads);

        const lossScalar = lossValue.dataSync()[0];
        epochLossAccumulator += lossScalar;
        epochBatchCounter += 1;

        Object.values(grads).forEach(grad => grad.dispose());
        lossValue.dispose();
        globalNorm.dispose();
        clipCoef.dispose();
        maxGradNormScalar.dispose();
        oneScalar.dispose();

        statesMB.dispose();
        actionsMB.dispose();
        oldLogProbsMB.dispose();
        returnsMB.dispose();
        advMB.dispose();
        oldValuesMB.dispose();
        idxTensor.dispose();
      }

      console.log(`[PPO 학습] 에포크 ${epoch + 1}/${this.ppoEpochs} - 평균 손실: ${(epochLossAccumulator / Math.max(epochBatchCounter, 1)).toFixed(4)}`);
      epochLossAccumulator = 0;
      epochBatchCounter = 0;
    }

    // 텐서 정리
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
      reward -= Math.sqrt((320 - status[8]) ** 2 + status[9] ** 2)

      reward += 0.1 * Math.sqrt((320 + status[8]) ** 2 + status[9] ** 2)
*/
      let distanza_alla_palla = Math.sqrt((status[8] - status[0]) ** 2 + (status[9] - status[1]) ** 2)
      reward -= distanza_alla_palla / 2   


      function prodotto_scalare(a, b){
          let lung_a = Math.max(1e-5, lung(a))
          let lung_b = Math.max(1e-5, lung(b))
          return (a[0] * b[0] + a[1] * b[1]) / lung_a / lung_b
      }
      function lung(a){
          return Math.sqrt(a[0] ** 2 + a[1] ** 2)
      }

      let vett_palla_porta = [320 - status[8], -status[9]]
      reward += prodotto_scalare(vett_palla_porta, [status[10], status[11]])


      if (!status[13]){
          let velocita_palla = Math.sqrt(status[10] ** 2 + status[11] ** 2)
          reward -= 100 * Math.max(0.0, 0.5 - velocita_palla)
      }

      if (status[8] < status[0]) reward -= Math.abs(status[0] - status[8])


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

      reward += goal_reward

      return reward
  }
}
  
module.exports = { Orchestrator };