const {HaxAI} = require('./haxai');
const {Model} = require('./AImodel');
const {Memory} = require('./memory');
const tf = require('@tensorflow/tfjs-node');

const ACTION_INVERSION_TABLE = {
  0: 0, // kick
  1: 5, // forward <-> backward
  2: 6, // forward-left <-> backward-right
  3: 7, // left <-> right
  4: 8, // backward-left <-> forward-right
  5: 1,
  6: 2,
  7: 3,
  8: 4,
  9: 9 // none
};

const invertActionIndex = (action) => {
  const mapped = ACTION_INVERSION_TABLE[action];
  return mapped != null ? mapped : 9;
};

const invertStateVector = (state) => {
  if (!Array.isArray(state)) {
    return [];
  }
  const mirrored = state.slice();
  if (mirrored.length < 14) {
    return mirrored;
  }

  // swap bot/op coordinates along X axis, keep Y as-is for field symmetry
  const botX = state[0];
  const botY = state[1];
  const botVx = state[2];
  const botVy = state[3];
  const opX = state[4];
  const opY = state[5];
  const opVx = state[6];
  const opVy = state[7];
  const ballX = state[8];
  const ballY = state[9];
  const ballVx = state[10];
  const ballVy = state[11];

  mirrored[0] = -opX;
  mirrored[1] = opY;
  mirrored[2] = -opVx;
  mirrored[3] = opVy;

  mirrored[4] = -botX;
  mirrored[5] = botY;
  mirrored[6] = -botVx;
  mirrored[7] = botVy;

  mirrored[8] = -ballX;
  mirrored[9] = ballY;
  mirrored[10] = -ballVx;
  mirrored[11] = ballVy;

  // Distance to ball and lock state remain identical after mirroring
  return mirrored;
};

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
      this.entropyCoeff = 0.05;      // 엔트로피 정규화 계수
      this.learningRate = 3e-4 / 10;      // 초기 학습률 (baselines PPO2 기본값)
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

      this.recurrentState = this.model.getInitialState(1);
      this.evalRecurrentState = this.model.getInitialState(1);
      this.lastTrainMask = 1;
      this.lastEvalMask = 1;
  }

  sleep (time) {
    return new Promise((resolve) => setTimeout(resolve, time));
  }

  resetTrainRecurrentState() {
    if (this.recurrentState) {
      this.model.disposeState(this.recurrentState);
    }
    this.recurrentState = this.model.getInitialState(1);
    this.lastTrainMask = 1;
  }

  resetEvalRecurrentState() {
    if (this.evalRecurrentState) {
      this.model.disposeState(this.evalRecurrentState);
    }
    this.evalRecurrentState = this.model.getInitialState(1);
    this.lastEvalMask = 1;
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
      const recurrentForStep = this.model.cloneState(this.recurrentState);
      const maskValue = this.lastTrainMask;
      const { action, negLogProb, value, nextState } = this.getActionAndValue(
        stateTensor,
        recurrentForStep,
        maskValue
      );

      const prevStateArrays = this.model.stateToArray(recurrentForStep);
      this.model.disposeState(recurrentForStep);

      this.model.disposeState(this.recurrentState);
      this.recurrentState = nextState;

      await this.haxai.update(action, page);
      await this.sleep(10000 / 60);

      const nextStateTensor = this.haxai.getStateTensor();
      const reward = this.computeReward(nextStateTensor.arraySync()[0], this.haxai.getState());
      const done = this.haxai.isDone();

      const nextStateClone = this.model.cloneState(this.recurrentState);
      const nextStateArrays = this.model.stateToArray(nextStateClone);
      this.model.disposeState(nextStateClone);

      this.memory.addSample({
        state: stateTensor,
        action,
        reward,
        nextState: nextStateTensor,
        negLogProb,
        value,
        done,
        mask: maskValue,
        prevRnnState: prevStateArrays,
        nextRnnState: nextStateArrays
      });

      if (done) {
        this.resetTrainRecurrentState();
      }

      this.lastTrainMask = done ? 0 : 1;

      if (this.memory.samples.length % this.model.batchSize === 0) {
        this.ppoUpdate();
      }
  }
  
  async test(page){
    const stateTensor = this.haxai.getStateTensor();
    const evalStateClone = this.model.cloneState(this.evalRecurrentState);
    const maskValue = this.lastEvalMask;
    const { action, nextState } = this.getActionAndValue(stateTensor, evalStateClone, maskValue);

    this.model.disposeState(evalStateClone);
    this.model.disposeState(this.evalRecurrentState);
    this.evalRecurrentState = nextState;

    await this.haxai.update(action, page);
    stateTensor.dispose();

    const done = this.haxai.isDone();
    if (done) {
      this.resetEvalRecurrentState();
    }
    this.lastEvalMask = done ? 0 : 1;
  }

  getActionAndValue(stateTensor, recurrentState, maskValue = 1, deterministic = false) {
    const { action, negLogProb, value, nextState } = this.model.step(
      stateTensor,
      recurrentState,
      maskValue,
      deterministic
    );

    const asScalarOrArray = (val) => {
      if (Array.isArray(val)) {
        return val.length === 1 ? val[0] : val.slice();
      }
      return val;
    };

    const processedAction = asScalarOrArray(action);
    const processedValue = asScalarOrArray(value);
    const processedNegLogProb = asScalarOrArray(negLogProb);

    return {
      action: processedAction,
      negLogProb: processedNegLogProb,
      value: processedValue,
      nextState
    };
  }
  ppoUpdate() {
    const rawBatch = this.memory.samples.slice();
    if (!rawBatch || rawBatch.length === 0) {
      console.warn('No samples available for PPO update, skipping');
      return;
    }

    const validBatch = rawBatch.filter((sample, idx) => {
      const issues = [];
      if (!sample || typeof sample !== 'object') {
        issues.push('sample is null');
      } else {
        const validateStateArray = (stateArray, label) => {
          if (!Array.isArray(stateArray)) {
            issues.push(`${label} missing`);
            return;
          }
          if (stateArray.length !== this.model.numStates) {
            issues.push(`${label} length invalid (${stateArray.length})`);
            return;
          }
          for (let i = 0; i < stateArray.length; i++) {
            if (!Number.isFinite(stateArray[i])) {
              issues.push(`${label} contains non-finite value at ${i}`);
              break;
            }
          }
        };

        validateStateArray(sample.state, 'state');
        validateStateArray(sample.nextState, 'nextState');

        if (!Number.isFinite(sample.action)) {
          issues.push(`action invalid (${sample.action})`);
        }
        if (!Number.isFinite(sample.negLogProb)) {
          issues.push(`negLogProb invalid (${sample.negLogProb})`);
        }
        if (!Number.isFinite(sample.value)) {
          issues.push(`value invalid (${sample.value})`);
        }
        if (
          !sample.prevRnnState ||
          !Array.isArray(sample.prevRnnState.h) ||
          !Array.isArray(sample.prevRnnState.c) ||
          sample.prevRnnState.h.length !== this.model.lstmUnits ||
          sample.prevRnnState.c.length !== this.model.lstmUnits
        ) {
          issues.push('prevRnnState invalid');
        }
        if (
          !sample.nextRnnState ||
          !Array.isArray(sample.nextRnnState.h) ||
          !Array.isArray(sample.nextRnnState.c) ||
          sample.nextRnnState.h.length !== this.model.lstmUnits ||
          sample.nextRnnState.c.length !== this.model.lstmUnits
        ) {
          issues.push('nextRnnState invalid');
        }
      }
      if (issues.length > 0) {
        console.warn(`[PPO] 샘플 무시 (index: ${idx}) - ${issues.join(', ')}`);
        return false;
      }
      return true;
    });

    if (validBatch.length < this.model.batchSize) {
      console.warn(`[PPO] 유효 샘플 부족: ${validBatch.length}/${this.model.batchSize}`);
      return;
    }

    const baseBatch = validBatch.slice(-this.model.batchSize);
    const baseBatchSize = baseBatch.length;
    const numEnvs = Math.max(1, this.model.numEnvs || 1);

    if (baseBatchSize % numEnvs !== 0) {
      console.warn(`[PPO] 샘플 수가 환경 수의 배수가 아닙니다 (${baseBatchSize} vs numEnvs ${numEnvs})`);
      return;
    }

    const stepsPerEnv = baseBatchSize / numEnvs;

    const baseStates = baseBatch.map(sample => sample.state);
    const baseNextStates = baseBatch.map(sample => sample.nextState);
    const baseActions = baseBatch.map(sample => sample.action);
    const baseNegLogProbs = baseBatch.map(sample => sample.negLogProb);
    const baseValues = baseBatch.map(sample => sample.value);
    const baseRewards = baseBatch.map(sample => sample.reward);
    const baseMasks = baseBatch.map(sample => (sample.mask != null ? sample.mask : 1));
    const baseNonTerminal = baseBatch.map(sample => (sample.done ? 0 : 1));
    const basePrevRnnStates = baseBatch.map(sample => sample.prevRnnState);
    const baseNextRnnStates = baseBatch.map(sample => sample.nextRnnState);

    const nextStatesTensor = tf.tensor2d(baseNextStates, [baseBatchSize, this.model.numStates]);
    const nextRecurrentBatch = this.model.batchStateArraysToTensors(baseNextRnnStates);
    const [nextPolicyLogits, nextValuesTensor, nextHiddenH, nextHiddenC] = this.model.forward(
      nextStatesTensor,
      nextRecurrentBatch,
      null
    );
    nextPolicyLogits.dispose();
    nextHiddenH.dispose();
    nextHiddenC.dispose();

    const nextValuesFlat = tf.squeeze(nextValuesTensor);
    const nextValuesArr = nextValuesFlat.arraySync();
    nextValuesFlat.dispose();
    nextValuesTensor.dispose();
    nextStatesTensor.dispose();
    nextRecurrentBatch.h.dispose();
    nextRecurrentBatch.c.dispose();

    const advantagesArr = new Array(baseBatchSize).fill(0);
    const returnsArr = new Array(baseBatchSize).fill(0);

    for (let envIdx = 0; envIdx < numEnvs; envIdx++) {
      let gae = 0;
      for (let step = stepsPerEnv - 1; step >= 0; step--) {
        const flatIdx = step * numEnvs + envIdx;
        const mask = baseNonTerminal[flatIdx];
        const delta = baseRewards[flatIdx] + this.discountRate * nextValuesArr[flatIdx] * mask - baseValues[flatIdx];
        gae = delta + this.discountRate * this.gaeLambda * mask * gae;
        advantagesArr[flatIdx] = gae;
        returnsArr[flatIdx] = gae + baseValues[flatIdx];
      }
    }

    const mirroredStates = baseStates.map(invertStateVector);
    const mirroredActions = baseActions.map(invertActionIndex);

    const allStatesArray = baseStates.concat(mirroredStates);
    const allActionsArray = baseActions.concat(mirroredActions);
    const allNegLogProbsArray = baseNegLogProbs.concat(baseNegLogProbs);
    const allValuesArray = baseValues.concat(baseValues);
    const allAdvantagesArray = advantagesArr.concat(advantagesArr);
    const allReturnsArray = returnsArr.concat(returnsArr);
    const allMasksArray = baseMasks.concat(baseMasks);
    const allPrevRnnStates = basePrevRnnStates.concat(basePrevRnnStates);

    const batchSize = allStatesArray.length;
    const miniBatchSize = Math.max(1, Math.floor(batchSize / this.miniBatchSplits));

    const states = tf.tensor2d(allStatesArray, [batchSize, this.model.numStates]);
    const actions = tf.tensor1d(allActionsArray, 'int32');
    const oldNegLogProbsTensor = tf.tensor1d(allNegLogProbsArray);
    const oldLogProbs = oldNegLogProbsTensor.neg();
    const oldValues = tf.tensor1d(allValuesArray);
    const masksTensor = tf.tensor1d(allMasksArray, 'float32');

    const returnsTensorRaw = tf.tensor1d(allReturnsArray);
    const returnsClipped = tf.clipByValue(returnsTensorRaw, -200, 200);

    const epsilonScalar = tf.scalar(this.epsilon);
    const advantagesTensor = tf.tensor1d(allAdvantagesArray);
    const advMean = advantagesTensor.mean();
    const advStd = tf.sqrt(
      advantagesTensor.sub(advMean).square().mean().add(epsilonScalar)
    );
    const normalizedAdvantages = advantagesTensor.sub(advMean).div(advStd.add(epsilonScalar));

    returnsTensorRaw.dispose();
    advMean.dispose();
    advStd.dispose();
    advantagesTensor.dispose();

    const prevRecurrentBatch = this.model.batchStateArraysToTensors(allPrevRnnStates);

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
        const prevHMB = tf.gather(prevRecurrentBatch.h, idxTensor);
        const prevCMB = tf.gather(prevRecurrentBatch.c, idxTensor);
        const maskMB = tf.gather(masksTensor, idxTensor);

        const { value: lossValue, grads } = tf.variableGrads(() => tf.tidy(() => {
          const [policyLogits, valuePred, nextHMB, nextCMB] = this.model.forward(
            statesMB,
            { h: prevHMB, c: prevCMB },
            maskMB,
            true
          );
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

          nextHMB.dispose();
          nextCMB.dispose();

          return actorLoss
            .add(criticLoss.mul(this.valueCoeff))
            .sub(entropy.mul(this.entropyCoeff));
        }));

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
        prevHMB.dispose();
        prevCMB.dispose();
        maskMB.dispose();
        idxTensor.dispose();
      }

      console.log(`[PPO 학습] 에포크 ${epoch + 1}/${this.ppoEpochs} - 평균 손실: ${(epochLossAccumulator / Math.max(epochBatchCounter, 1)).toFixed(4)}`);
      epochLossAccumulator = 0;
      epochBatchCounter = 0;
    }

    states.dispose();
    actions.dispose();
    oldLogProbs.dispose();
    oldNegLogProbsTensor.dispose();
    oldValues.dispose();
    returnsClipped.dispose();
    normalizedAdvantages.dispose();
    epsilonScalar.dispose();
    prevRecurrentBatch.h.dispose();
    prevRecurrentBatch.c.dispose();
    masksTensor.dispose();

    this.updateLearningRate();

    this.memory.clear();
  }

    computeReward(status,more_state) {
      let reward = 0

      reward -= Math.sqrt((320 - status[8]) ** 2 + status[9] ** 2)

      reward += 0.1 * Math.sqrt((320 + status[8]) ** 2 + status[9] ** 2)

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