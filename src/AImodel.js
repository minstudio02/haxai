const tf = require('@tensorflow/tfjs-node');


class Model {
    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;

        if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
            this.network = hiddenLayerSizesOrModel;
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
        hiddenLayerSizes.forEach(size => {
            x = tf.layers.dense({
                units: size,
                activation: 'relu'
            }).apply(x);
        });

        // 정책 헤드 (행동 확률)
        const policyHead = tf.layers.dense({
            units: this.numActions,
            activation: 'softmax',
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
            loss: ['categoricalCrossentropy', 'meanSquaredError']
        });
    }

    predict(states) {
        return tf.tidy(() => {
            const output = this.network.predict(states);
            const actionProbs = output[0];
            const stateValue = output[1];
            return [actionProbs, stateValue];
        });
    }

    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch, {epochs: 100});
    }

    chooseAction(state, eps) {
        return tf.tidy(() => {
            const [actionProbs] = this.predict(state);
            
            if (Math.random() < eps) {
                // 탐험: 무작위 행동
                return Math.floor(Math.random() * this.numActions);
            } else {
                // 활용: 확률적 선택
                return tf.multinomial(actionProbs, 1).dataSync()[0];
            }
        });
    }
}
module.exports = { Model };