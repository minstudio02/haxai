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

      this.own_score=0
      this.opponent_score=0

      // PPO 하이퍼파라미터 개선
      this.clipEpsilon = 0.2;  // PPO 클리핑 범위
      this.ppoEpochs = 3;      // PPO 업데이트 에포크 수
      this.valueCoeff = 0.5;   // 가치 손실 계수
      this.entropyCoeff = 0.01; // 엔트로피 정규화 계수
      this.learningRate = 1e-4; // 학습률

      // 옵티마이저 개선
      this.optimizer = tf.train.adam(this.learningRate);
  }

  sleep (time) {
    return new Promise((resolve) => setTimeout(resolve, time));
  }

  async stop() {
      await this.replay()
  }

  async train(page) {
      const stateTensor = this.haxai.getStateTensor();
      const [action, actionProb, value] = this.getActionAndValue(stateTensor);
      
      await this.haxai.update(action, page);
      await this.sleep(2000/60);

      const nextState = this.haxai.getStateTensor();
      const reward = this.computeReward(nextState.arraySync()[0], this.haxai.getState());
      const done = this.haxai.isDone();

      this.memory.addSample([stateTensor, action, reward, nextState, actionProb, value, done]);

      if (this.memory.samples.length >= this.model.batchSize) {
        await this.ppoUpdate();
      }
  }
  
  async test(page){
    tf.tidy(async() => {
      const action = this.getActions(this.haxai.getStateTensor())[0];
      await this.haxai.update(action,page);
    });  
  }

  async replay() {
      // Sample from memory
      const batch = this.memory.sample(this.model.batchSize);
      if (!batch || batch.length === 0) {
        console.warn('Empty batch in replay, skipping');
        return;
      }

      const states = batch.map(([state, , , ]) => state);
      const nextStates = batch.map(
          ([, , , nextState]) => nextState ? nextState : tf.zeros([1, this.model.numStates])
      );
      
      // Predict the values of each action at each state (두 개의 출력 중 첫 번째만 사용)
      const qsa = states.map((state) => {
        const [actionProbs, value] = this.model.predict(state);
        if (value) value.dispose(); // 가치 출력은 사용하지 않으므로 dispose
        return actionProbs; // 정책 출력만 반환
      });
      
      // Predict the values of each action at each next state
      const qsad = nextStates.map((nextState) => {
        const [actionProbs, value] = this.model.predict(nextState);
        if (value) value.dispose(); // 가치 출력은 사용하지 않으므로 dispose
        return actionProbs; // 정책 출력만 반환
      });

      let x = new Array();
      let y = new Array();

      // Update the states rewards with the discounted next states rewards
      batch.forEach(
          ([state, action, reward, nextState], index) => {
              const currentQ = qsa[index].dataSync();
              const nextQ = qsad[index].dataSync();
              const maxNextQ = Math.max(...nextQ);
              
              // Q-learning 업데이트
              const updatedQ = [...currentQ];
              updatedQ[action] = nextState ? reward + this.discountRate * maxNextQ : reward;
              
              x.push(state.arraySync()[0]);
              y.push(updatedQ);
          }
      );

      // Clean unused tensors
      qsa.forEach((q) => q.dispose());
      qsad.forEach((q) => q.dispose());
      states.forEach((s) => s.dispose());
      nextStates.forEach((s) => s.dispose());

      // Reshape the batches to be fed to the network
      const xTensor = tf.tensor2d(x, [x.length, this.model.numStates]);
      const yTensor = tf.tensor2d(y, [y.length, this.model.numActions]);

      // Learn the Q(s, a) values given associated discounted rewards
      await this.model.train(xTensor, yTensor);

      xTensor.dispose();
      yTensor.dispose();
  }

  getActionAndValue(stateTensor) {
    return tf.tidy(() => {
      const [actionProbs, value] = this.model.predict(stateTensor);
      const action = tf.multinomial(actionProbs, 1).dataSync()[0];
      return [action, actionProbs.dataSync()[action], value.dataSync()[0]];
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
      const [actionProbs, value] = this.model.predict(inputs);
      if (value) value.dispose(); // 가치 출력은 사용하지 않으므로 dispose
      
      // actionProbs는 이미 softmax가 적용된 확률이므로 직접 사용
      const actions = tf.multinomial(actionProbs, 1, null, true);
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

  async ppoUpdate() {
    const batch = this.memory.sample(this.model.batchSize);
    if (!batch || batch.length === 0) {
      console.warn('Empty batch in ppoUpdate, skipping');
      return;
    }

    const states = tf.concat(batch.map(([state]) => state));
    const actions = tf.tensor1d(batch.map(([, action]) => action), 'int32');
    const rewards = tf.tensor1d(batch.map(([,, reward]) => reward));
    const nextStates = tf.concat(batch.map(([,,, nextState]) => nextState));
    const oldActionProbs = tf.tensor1d(batch.map(([,,,, actionProb]) => actionProb));
    const oldValues = tf.tensor1d(batch.map(([,,,,, value]) => value));
    const dones = tf.tensor1d(batch.map(([,,,,,, done]) => done ? 0 : 1));

    // nextStates에 대한 가치 예측 (두 번째 출력이 value)
    const [, nextValues] = this.model.predict(nextStates);
    // nextValues는 [batchSize, 1] 형태이므로 [batchSize]로 변환
    const nextValuesFlat = tf.squeeze(nextValues);
    const returns = rewards.add(tf.mul(this.discountRate, tf.mul(dones, nextValuesFlat)));
    const advantages = returns.sub(oldValues);

    for (let epoch = 0; epoch < this.ppoEpochs; epoch++) {
      // variableGrads를 사용하기 위해 함수 내에서 loss를 계산
      const grads = tf.variableGrads(() => {
        const [newActionProbs, newValues] = this.model.predict(states);
        
        // 각 샘플의 선택된 액션에 대한 확률만 가져오기
        // newActionProbs는 [batchSize, numActions] 형태
        // actions는 [batchSize] 형태의 액션 인덱스
        // oneHot으로 선택된 액션만 1로 만들고 곱셈으로 해당 확률만 추출
        const actionMask = tf.oneHot(actions, this.model.numActions);
        const selectedActionProbs = tf.sum(tf.mul(newActionProbs, actionMask), 1);
        
        // newValues는 [batchSize, 1] 형태이므로 [batchSize]로 변환
        const newValuesFlat = tf.squeeze(newValues);
        
        // ratio 계산: 새로운 확률 / 이전 확률
        const ratio = tf.div(selectedActionProbs, oldActionProbs);
        
        const surr1 = tf.mul(ratio, advantages);
        const surr2 = tf.mul(
          tf.clipByValue(ratio, 1 - this.clipEpsilon, 1 + this.clipEpsilon),
          advantages
        );

        const actorLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
        const criticLoss = tf.mean(tf.squaredDifference(newValuesFlat, returns));
        
        // 엔트로피 계산: -sum(p * log(p))
        const entropy = tf.neg(tf.sum(tf.mul(newActionProbs, tf.log(tf.add(newActionProbs, 1e-8))), 1));
        const entropyLoss = tf.mean(entropy);

        const totalLoss = actorLoss.add(tf.mul(this.valueCoeff, criticLoss)).sub(tf.mul(this.entropyCoeff, entropyLoss));
        
        return totalLoss;
      });
      
      this.optimizer.applyGradients(grads.grads);
      
      // 그래디언트 텐서 정리
      Object.values(grads.grads).forEach(grad => grad.dispose());
    }

    tf.dispose([states, actions, rewards, nextStates, oldActionProbs, oldValues, dones, 
                returns, advantages, nextValues, nextValuesFlat]);
    this.memory.clear();
  }

    computeReward(status, moreState) {
      // 보상 함수 개선
      let reward = 0;

      // 볼과의 거리 페널티
      const ballDistance = Math.sqrt(
          Math.pow(status[0] - status[8], 2) + 
          Math.pow(status[1] - status[9], 2)
      );
      reward -= ballDistance / 100;

      // 골대 방향으로의 진행 보상
      const goalVector = [320 - status[8], status[9]];
      const ballVelocity = [status[10], status[11]];
      const alignmentBonus = this.vectorAlignment(goalVector, ballVelocity);
      reward += alignmentBonus * 10;

      // 골 보상
      if (moreState.score.ownTeam > this.own_score) {
          reward += 500;  // 골 성공 시 큰 보상
          this.own_score = moreState.score.ownTeam;
      }
      if (moreState.score.opponentTeam > this.opponent_score) {
          reward -= 250;  // 실점 시 페널티
          this.opponent_score = moreState.score.opponentTeam;
      }

      // 게임 상태 보상
      if (!moreState.game_State) {
          reward -= 0.1;  // 게임 시작 지연에 대한 페널티
      }

      return reward;
  }

  vectorAlignment(vec1, vec2) {
      const dotProduct = vec1[0] * vec2[0] + vec1[1] * vec2[1];
      const mag1 = Math.sqrt(vec1[0]**2 + vec1[1]**2);
      const mag2 = Math.sqrt(vec2[0]**2 + vec2[1]**2);
      return dotProduct / (mag1 * mag2 + 1e-8);
  }
}
  
module.exports = { Orchestrator };