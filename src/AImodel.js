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

        // 공유 백본
        let shared = input;
        hiddenLayerSizes.forEach((size, id) => {
            shared = tf.layers.dense({
                units: size,
                activation: (id == 2 || id == 3) ? 'relu' : 'tanh',
                kernelInitializer: 'ones',
                biasInitializer: 'ones'
            }).apply(shared);
        });

        const policyHead = tf.layers.dense({
            units: this.numActions,
            activation: 'linear',
            name: 'policy',
            kernelInitializer: 'ones',
            biasInitializer: 'ones'
        }).apply(shared);

        const valueHead = tf.layers.dense({
            units: 1,
            activation: 'linear',
            name: 'value',
            kernelInitializer: 'ones',
            biasInitializer: 'ones'
        }).apply(shared);

        this.network = tf.model({ inputs: input, outputs: [policyHead, valueHead] });
        this.compileNetwork();
    }

    compileNetwork() {
        this.network.compile({
            optimizer: tf.train.adam(7e-4),
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