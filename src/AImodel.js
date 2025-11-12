const tf = require('@tensorflow/tfjs-node');


class Model {
    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;

        if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
            this.network = hiddenLayerSizesOrModel;
            this.compileNetwork();
        } else {
            this.defineModel(hiddenLayerSizesOrModel);
        }
        this.network.summary();
    }

    defineModel(hiddenLayerSizes) {
        const defaultHidden = [64, 64];
        if (!Array.isArray(hiddenLayerSizes) || hiddenLayerSizes.length === 0) {
            hiddenLayerSizes = defaultHidden;
        }

        const input = tf.input({ shape: [this.numStates] });

        // Stable-Baselines MlpPolicy 스타일과 동일한 정규 직교 초기화 적용
        const hiddenInitializer = tf.initializers.orthogonal({ gain: Math.sqrt(2) });
        const policyHeadInitializer = tf.initializers.orthogonal({ gain: 0.01 });
        const valueHeadInitializer = tf.initializers.orthogonal({ gain: 1.0 });

        // 공유 백본
        let shared = input;
        hiddenLayerSizes.forEach((size) => {
            shared = tf.layers.dense({
                units: size,
                activation: 'tanh',
                kernelInitializer: hiddenInitializer,
                biasInitializer: 'zeros'
            }).apply(shared);
        });

        // 정책 / 가치 독립 헤드
        const policyBranch = tf.layers.dense({
            units: hiddenLayerSizes[hiddenLayerSizes.length - 1],
            activation: 'tanh',
            kernelInitializer: hiddenInitializer,
            biasInitializer: 'zeros'
        }).apply(shared);

        const valueBranch = tf.layers.dense({
            units: hiddenLayerSizes[hiddenLayerSizes.length - 1],
            activation: 'tanh',
            kernelInitializer: hiddenInitializer,
            biasInitializer: 'zeros'
        }).apply(shared);

        const policyHead = tf.layers.dense({
            units: this.numActions,
            activation: 'linear',
            name: 'policy',
            kernelInitializer: policyHeadInitializer,
            biasInitializer: 'zeros'
        }).apply(policyBranch);

        const valueHead = tf.layers.dense({
            units: 1,
            activation: 'linear',
            name: 'value',
            kernelInitializer: valueHeadInitializer,
            biasInitializer: 'zeros'
        }).apply(valueBranch);

        this.network = tf.model({ inputs: input, outputs: [policyHead, valueHead] });
        this.compileNetwork();
    }

    compileNetwork() {
        this.network.compile({
            optimizer: tf.train.adam(),
            loss: [
                (yTrue, yPred) => tf.losses.softmaxCrossEntropy(yTrue, yPred),
                'meanSquaredError'
            ]
        });
    }

    predict(states) {
        return tf.tidy(() => {
            const [policyLogits, stateValue] = this.network.predict(states);
            return [policyLogits, stateValue];
        });
    }
}

module.exports = { Model };