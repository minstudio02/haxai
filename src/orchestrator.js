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
      this.lastBallDistance = Infinity;  // 이전 프레임의 볼 거리
      this.lastBallPosition = {x: 0, y: 0};  // 이전 프레임의 볼 위치

      // PPO 하이퍼파라미터 개선
      this.clipEpsilon = 0.2;  // PPO 클리핑 범위
      this.ppoEpochs = 3;      // PPO 업데이트 에포크 수
      this.valueCoeff = 0.5;   // 가치 손실 계수
      this.entropyCoeff = 0.01; // 엔트로피 정규화 계수
      this.learningRate = 3e-5; // 학습률 (NaN 방지를 위해 낮춤)

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

      if (this.memory.samples.length % this.model.batchSize == 0) {
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
          ([, , , nextState]) => nextState ? nextState : tf.zeros([this.model.numStates])
      );
      
      // Predict the values of each action at each state
      const qsa = states.map((state) => {
        const [actionProbs, value] = this.model.predict(state);
        return { actionProbs, value };
      });
      
      // Predict the values of each action at each next state
      const qsad = nextStates.map((nextState) => {
        const [actionProbs, value] = this.model.predict(nextState);
        return { actionProbs, value };
      });

      let x = new Array();
      let yPolicy = new Array();
      let yValue = new Array();

      // Update the states rewards with the discounted next states rewards
      batch.forEach(
          ([state, action, reward, nextState], index) => {
              const currentQ = qsa[index].actionProbs
              const maxNextQ = qsad[index].actionProbs.max().dataSync()
              currentQ[action] = nextState ? reward + this.discountRate * maxNextQ : reward;
              
              // 다음 상태의 가치를 사용하여 현재 상태의 가치 타겟 계산
              const nextValue = nextState ? qsad[index].value.dataSync() : 0;
              const targetValue = reward + this.discountRate * nextValue;
              
              x.push(state.dataSync());
              yPolicy.push(currentQ.dataSync());
              yValue.push(targetValue);
          }
      );

      // Clean unused tensors
      qsa.forEach(({ actionProbs, value }) => {
        actionProbs.dispose();
        value.dispose();
      });
      qsad.forEach(({ actionProbs, value }) => {
        actionProbs.dispose();
        value.dispose();
      });

      // Reshape the batches to be fed to the network
      x = tf.tensor2d(x, [x.length, this.model.numStates]);
      const yPolicyTensor = tf.tensor2d(yPolicy, [yPolicy.length, this.model.numActions]);
      const yValueTensor = tf.tensor2d(yValue, [yValue.length, 1]);

      // Learn the Q(s, a) values given associated discounted rewards
      // 모델이 2개의 출력을 가지므로 2개의 타겟 텐서를 전달
      await this.model.train(x, [yPolicyTensor, yValueTensor]);

      x.dispose();
      yPolicyTensor.dispose();
      yValueTensor.dispose();
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

    // 입력 데이터 검증 및 정규화
    const states = tf.concat(batch.map(([state]) => state));
    const actions = tf.tensor1d(batch.map(([, action]) => action), 'int32');
    const rewards = tf.tensor1d(batch.map(([,, reward]) => {
      // 보상 값 정규화 및 NaN 체크
      const r = isNaN(reward) || !isFinite(reward) ? 0 : reward;
      return Math.max(-100, Math.min(100, r)); // 보상 클리핑
    }));
    const nextStates = tf.concat(batch.map(([,,, nextState]) => nextState));
    const oldActionProbs = tf.tensor1d(batch.map(([,,,, actionProb]) => {
      // 확률 값 검증 및 클리핑
      const prob = isNaN(actionProb) || !isFinite(actionProb) ? 1e-8 : actionProb;
      return Math.max(1e-8, Math.min(1.0, prob));
    }));
    const oldValues = tf.tensor1d(batch.map(([,,,,, value]) => {
      // 가치 값 검증 및 클리핑
      const v = isNaN(value) || !isFinite(value) ? 0 : value;
      return Math.max(-1000, Math.min(1000, v));
    }));
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
        
        // 입력 데이터 검증 (NaN 체크)
        const hasNaN = tf.any(tf.isNaN(newActionProbs)).dataSync()[0] || 
                      tf.any(tf.isNaN(newValues)).dataSync()[0] ||
                      tf.any(tf.isNaN(advantages)).dataSync()[0];
        if (hasNaN) {
          console.warn('NaN detected in predictions or advantages, skipping update');
          return tf.scalar(0);
        }
        
        // 각 샘플의 선택된 액션에 대한 확률만 가져오기
        // newActionProbs는 [batchSize, numActions] 형태
        // actions는 [batchSize] 형태의 액션 인덱스
        // oneHot으로 선택된 액션만 1로 만들고 곱셈으로 해당 확률만 추출
        const actionMask = tf.oneHot(actions, this.model.numActions);
        const selectedActionProbs = tf.sum(tf.mul(newActionProbs, actionMask), 1);
        
        // newValues는 [batchSize, 1] 형태이므로 [batchSize]로 변환
        const newValuesFlat = tf.squeeze(newValues);
        
        // ratio 계산: 새로운 확률 / 이전 확률 (NaN 방지를 위해 작은 epsilon 추가)
        const ratio = tf.div(selectedActionProbs, tf.add(oldActionProbs, 1e-8));
        
        // ratio 클리핑 (NaN 방지)
        const clippedRatio = tf.clipByValue(ratio, 1e-8, 10.0);
        
        const surr1 = tf.mul(clippedRatio, advantages);
        const surr2 = tf.mul(
          tf.clipByValue(clippedRatio, 1 - this.clipEpsilon, 1 + this.clipEpsilon),
          advantages
        );

        const actorLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
        const criticLoss = tf.mean(tf.squaredDifference(newValuesFlat, returns));
        
        // 엔트로피 계산: -sum(p * log(p + eps)) (NaN 방지)
        const eps = 1e-8;
        const probsWithEps = tf.clipByValue(tf.add(newActionProbs, eps), eps, 1.0);
        const logProbs = tf.log(probsWithEps);
        const entropy = tf.neg(tf.sum(tf.mul(newActionProbs, logProbs), 1));
        const entropyLoss = tf.mean(entropy);

        const totalLoss = actorLoss.add(tf.mul(this.valueCoeff, criticLoss)).sub(tf.mul(this.entropyCoeff, entropyLoss));
        
        // 손실 값 검증
        const lossValue = totalLoss.dataSync()[0];
        if (isNaN(lossValue) || !isFinite(lossValue)) {
          console.warn('NaN or Inf loss detected, skipping update');
          return tf.scalar(0);
        }
        
        return totalLoss;
      });
      
      // 그래디언트 클리핑 적용
      const clippedGrads = {};
      const clipValue = 0.5; // 그래디언트 클리핑 값
      for (const [varName, grad] of Object.entries(grads.grads)) {
        const clippedGrad = tf.clipByValue(grad, -clipValue, clipValue);
        clippedGrads[varName] = clippedGrad;
        grad.dispose(); // 원본 그래디언트 dispose
      }
      
      this.optimizer.applyGradients(clippedGrads);
      
      // 그래디언트 텐서 정리
      Object.values(clippedGrads).forEach(grad => grad.dispose());
    }

    // 텐서 정리 (이미 dispose된 텐서는 제외)
    states.dispose();
    actions.dispose();
    rewards.dispose();
    nextStates.dispose();
    oldActionProbs.dispose();
    oldValues.dispose();
    dones.dispose();
    returns.dispose();
    advantages.dispose();
    nextValues.dispose();
    nextValuesFlat.dispose();
  }

    computeReward(status, moreState) {
      // 보상 함수 개선
      let reward = 0;

      // 상태 변수 추출
      const botPos = {x: status[0], y: status[1]};
      const botVel = {x: status[2], y: status[3]};
      const opPos = {x: status[4], y: status[5]};
      const ballPos = {x: status[8], y: status[9]};
      const ballVel = {x: status[10], y: status[11]};
      const ballDistance = status[12];  // 이미 계산된 거리
      
      // 골대 위치 (플레이어 팀에 따라 다름)
      const goalX = moreState.bot_Team == 1 ? 320 : -320;  // 상대 골대 x 좌표
      const goalY = 0;    // 골대 중앙 y 좌표

      // 1. 볼과의 거리 보상 (볼에 가까워질수록 보상, 지수적 감소)
      const ballProximityReward = Math.exp(-ballDistance / 50) * 5;
      reward += ballProximityReward;

      // 2. 볼에 가까워지는 것에 대한 보상 (거리 변화)
      if (this.lastBallDistance !== Infinity) {
        const distanceImprovement = this.lastBallDistance - ballDistance;
        reward += distanceImprovement * 2;  // 가까워질수록 보상
      }
      this.lastBallDistance = ballDistance;

      // 3. 볼 소유 보상 (볼과 매우 가까울 때)
      const BALL_POSSESSION_DISTANCE = 16;  // 볼 소유로 간주하는 거리
      if (ballDistance < BALL_POSSESSION_DISTANCE) {
        reward += 10;  // 볼 소유 보상
        
        // 4. 볼을 골대 방향으로 차는 것에 대한 보상
        const toGoalVector = {x: goalX - ballPos.x, y: goalY - ballPos.y};
        const ballToGoalAlignment = this.vectorAlignment(
          [toGoalVector.x, toGoalVector.y],
          [ballVel.x, ballVel.y]
        );
        reward += ballToGoalAlignment * 15;  // 골대 방향으로 차면 보상
        
        // 5. 골대 근처에서의 보상 (골대 100 이내)
        const goalDistance = Math.sqrt(
          Math.pow(ballPos.x - goalX, 2) + Math.pow(ballPos.y - goalY, 2)
        );
        if (goalDistance < 100) {
          reward += (100 - goalDistance) / 10;  // 골대에 가까울수록 보상
        }
      }

      // 6. 볼 속도 보상 (볼을 빠르게 움직일 때, 특히 골대 방향으로)
      const ballSpeed = Math.sqrt(ballVel.x ** 2 + ballVel.y ** 2);
      if (ballSpeed > 0) {
        const toGoalVector = {x: goalX - ballPos.x, y: goalY - ballPos.y};
        const speedAlignment = this.vectorAlignment(
          [toGoalVector.x, toGoalVector.y],
          [ballVel.x, ballVel.y]
        );
        reward += ballSpeed * speedAlignment * 0.5;  // 골대 방향으로 빠르게 움직일 때 보상
      }

      // 7. 움직임 보상 (가만히 있지 않고 움직일 때)
      const botSpeed = Math.sqrt(botVel.x ** 2 + botVel.y ** 2);
      if (botSpeed > 0.1) {
        reward += botSpeed * 0.1;  // 움직일 때 작은 보상
      } else {
        reward -= 0.5;  // 가만히 있으면 페널티
      }

      // 8. 상대방과의 거리 고려 (수비/공격 상황)
      const opDistance = Math.sqrt(
        Math.pow(botPos.x - opPos.x, 2) + Math.pow(botPos.y - opPos.y, 2)
      );
      
      // 볼을 가지고 있을 때 상대방과의 거리
      if (ballDistance < BALL_POSSESSION_DISTANCE) {
        if (opDistance < 50) {
          reward -= 5;  // 상대방이 가까이 있으면 페널티 (압박)
        }
      } else {
        // 볼을 가지고 있지 않을 때 상대방과 볼 사이에 위치
        const ballToOpDistance = Math.sqrt(
          Math.pow(opPos.x - ballPos.x, 2) + Math.pow(opPos.y - ballPos.y, 2)
        );
        if (ballToOpDistance < ballDistance) {
          // 상대방이 볼에 더 가까우면 수비 보상
          reward += 2;
        }
      }

      // 9. 골 보상/페널티
      if (moreState.score.ownTeam > this.own_score) {
          reward += 1000;  // 골 성공 시 큰 보상
          this.own_score = moreState.score.ownTeam;
      }
      if (moreState.score.opponentTeam > this.opponent_score) {
          reward -= 500;  // 실점 시 큰 페널티
          this.opponent_score = moreState.score.opponentTeam;
      }

      // 10. 게임 상태 페널티
      if (!moreState.game_State) {
          reward -= 0.5;  // 게임 시작 지연에 대한 페널티
      }

      // 11. 볼이 골대 근처로 이동할 때 보상
      if (this.lastBallPosition.x !== 0 || this.lastBallPosition.y !== 0) {
        const lastGoalDistance = Math.sqrt(
          Math.pow(this.lastBallPosition.x - goalX, 2) + 
          Math.pow(this.lastBallPosition.y - goalY, 2)
        );
        const currentGoalDistance = Math.sqrt(
          Math.pow(ballPos.x - goalX, 2) + Math.pow(ballPos.y - goalY, 2)
        );
        if (currentGoalDistance < lastGoalDistance) {
          reward += (lastGoalDistance - currentGoalDistance) * 0.1;  // 골대에 가까워질수록 보상
        }
      }
      this.lastBallPosition = {x: ballPos.x, y: ballPos.y};

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