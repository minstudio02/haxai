const tf = require('@tensorflow/tfjs-node');


class Model {
    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;

        if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
            this.network = hiddenLayerSizesOrModel;
            this.network.summary();
            this.network.compile({
                optimizer: 'adam',
                loss: [
                    (yTrue, yPred) => tf.losses.softmaxCrossEntropy(yTrue, yPred),
                    'meanSquaredError'
                ]
            });
        } else {
            this.defineModel(hiddenLayerSizesOrModel);
        }
    }

    defineModel(hiddenLayerSizes) {
        if (!Array.isArray(hiddenLayerSizes)) {
            hiddenLayerSizes = [hiddenLayerSizes];
        }

        // Functional API를 사용하여 다중 출력 모델 생성
        const input = tf.input({ shape: [this.numStates] });
        
        // 공통 백본 레이어
        let x = input;
        hiddenLayerSizes.forEach((size,id) => {
            x = tf.layers.dense({
                units: size,
                activation: id == 2 || id  == 3 ? 'relu' : 'tanh'
            }).apply(x);
        });

        // 정책 헤드 (행동 확률)
        const policyHead = tf.layers.dense({
            units: this.numActions,
            activation: 'linear',
            name: 'policy'
        }).apply(x);

        // 가치 헤드 (상태 가치)
        const valueHead = tf.layers.dense({
            units: 1,
            activation: 'linear',
            name: 'value'
        }).apply(x);

        // 다중 출력 모델 생성
        this.network = tf.model({ inputs: input, outputs: [policyHead, valueHead] });

        this.network.summary();
        this.network.compile({
            optimizer: 'adam',
            loss: [
                (yTrue, yPred) => tf.losses.softmaxCrossEntropy(yTrue, yPred),
                'meanSquaredError'
            ]
        });
    }

    predict(states) {
        return tf.tidy(() => {
            const output = this.network.predict(states);
            const policyLogits = output[0];
            const stateValue = output[1];
            return [policyLogits, stateValue];
        });
    }

    async train(xBatch, yBatch) {
        // 입력 데이터 검증
        const xData = xBatch.arraySync();
        const yPolicyData = yBatch[0].arraySync();
        const yValueData = yBatch[1].arraySync();
        
        // NaN 체크
        const hasNaN = xData.some(row => row.some(val => isNaN(val) || !isFinite(val))) ||
                      yPolicyData.some(row => row.some(val => isNaN(val) || !isFinite(val))) ||
                      yValueData.some(row => row.some(val => isNaN(val) || !isFinite(val)));
        
        if (hasNaN) {
            console.warn('NaN detected in training data, skipping training step');
            return;
        }
        
        // epochs를 줄여서 안정성 향상
        await this.network.fit(xBatch, yBatch, {
            epochs: 4,
            verbose: 1,
            callbacks: {
                onBatchEnd: (batch, logs) => {
                    if (isNaN(logs.loss) || !isFinite(logs.loss)) {
                        console.warn('NaN loss detected during training');
                    }
                }
            }
        });
    }

    chooseAction(state, eps) {
        return tf.tidy(() => {
            const [policyLogits, stateValue] = this.predict(state);
            const actionProbs = tf.softmax(policyLogits);

            if (Math.random() < eps) {
                // 탐험: 무작위 행동
                policyLogits.dispose();
                actionProbs.dispose();
                stateValue.dispose();
                return Math.floor(Math.random() * this.numActions);
            } else {
                // 활용: 확률적 선택
                const sampled = tf.multinomial(policyLogits, 1);
                const action = sampled.dataSync()[0];
                sampled.dispose();
                policyLogits.dispose();
                actionProbs.dispose();
                stateValue.dispose();
                return action;
            }
        });
    }
}
module.exports = { Model };