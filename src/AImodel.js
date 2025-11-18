const tf = require('@tensorflow/tfjs-node');

const DEFAULT_HIDDEN = [128, 128];
const DEFAULT_LSTM_UNITS = 128;
const DEFAULT_POLICY_HIDDEN = [128, 128];
const DEFAULT_VALUE_HIDDEN = [128, 128];

const orthogonalInitializer = (gain = 1.0) => tf.initializers.orthogonal({ gain });

class Model {
    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize, options = {}) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;

        const config = this._normalizeConfig(hiddenLayerSizesOrModel, options);
        this.hiddenLayerSizes = config.hiddenLayerSizes;
        this.lstmUnits = config.lstmUnits;
        this.numEnvs = config.numEnvs;
        this.policyHiddenSizes = config.policyHiddenSizes;
        this.valueHiddenSizes = config.valueHiddenSizes;

        if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
            this.network = hiddenLayerSizesOrModel;
            const inferredUnits = this._inferLstmUnitsFromNetwork(this.network);
            if (typeof inferredUnits === 'number' && Number.isFinite(inferredUnits)) {
                this.lstmUnits = inferredUnits;
            }
        } else {
            this._defineFeatureExtractor(this.hiddenLayerSizes);
        }

        this._buildRecurrentHeads();

        if (this.network && typeof this.network.summary === 'function') {
            try {
                this.network.summary();
            } catch (_) {
                // summary 출력 실패는 학습에 영향을 주지 않음
            }
        }
    }

    _normalizeConfig(hiddenLayerSizesOrModel, options) {
        const defaultHidden = options.defaultHidden || DEFAULT_HIDDEN;
        const defaultPolicyHidden = options.policyHiddenSizes || DEFAULT_POLICY_HIDDEN;
        const defaultValueHidden = options.valueHiddenSizes || DEFAULT_VALUE_HIDDEN;
        const config = {
            hiddenLayerSizes: defaultHidden,
            lstmUnits: options.lstmUnits || DEFAULT_LSTM_UNITS,
            numEnvs: options.numEnvs || 1,
            policyHiddenSizes: defaultPolicyHidden,
            valueHiddenSizes: defaultValueHidden
        };

        if (Array.isArray(hiddenLayerSizesOrModel)) {
            config.hiddenLayerSizes = hiddenLayerSizesOrModel;
        } else if (typeof hiddenLayerSizesOrModel === 'number') {
            config.hiddenLayerSizes = [hiddenLayerSizesOrModel];
        } else if (
            hiddenLayerSizesOrModel &&
            typeof hiddenLayerSizesOrModel === 'object' &&
            !(hiddenLayerSizesOrModel instanceof tf.LayersModel)
        ) {
            const maybeHidden = hiddenLayerSizesOrModel.hiddenLayerSizes || hiddenLayerSizesOrModel.hiddenLayers;
            if (maybeHidden) {
                config.hiddenLayerSizes = Array.isArray(maybeHidden) ? maybeHidden : [maybeHidden];
            }
            if (hiddenLayerSizesOrModel.lstmUnits) {
                config.lstmUnits = hiddenLayerSizesOrModel.lstmUnits;
            }
            if (hiddenLayerSizesOrModel.numEnvs) {
                config.numEnvs = hiddenLayerSizesOrModel.numEnvs;
            }
            if (hiddenLayerSizesOrModel.policyHiddenSizes) {
                const sizes = hiddenLayerSizesOrModel.policyHiddenSizes;
                config.policyHiddenSizes = Array.isArray(sizes) ? sizes : [sizes];
            }
            if (hiddenLayerSizesOrModel.valueHiddenSizes) {
                const sizes = hiddenLayerSizesOrModel.valueHiddenSizes;
                config.valueHiddenSizes = Array.isArray(sizes) ? sizes : [sizes];
            }
        }

        if (!Array.isArray(config.hiddenLayerSizes) || config.hiddenLayerSizes.length === 0) {
            config.hiddenLayerSizes = defaultHidden;
        }

        config.hiddenLayerSizes = config.hiddenLayerSizes.map((size) => {
            const parsed = Number(size);
            if (!Number.isFinite(parsed) || parsed <= 0) {
                throw new Error(`Invalid hidden layer size: ${size}`);
            }
            return parsed;
        });

        if (!Number.isFinite(config.lstmUnits) || config.lstmUnits <= 0) {
            config.lstmUnits = DEFAULT_LSTM_UNITS;
        }

        const normalizeHiddenList = (list, fallback) => {
            if (!Array.isArray(list) || list.length === 0) {
                return fallback.slice();
            }
            return list.map((size) => {
                const parsed = Number(size);
                if (!Number.isFinite(parsed) || parsed <= 0) {
                    throw new Error(`Invalid hidden layer size: ${size}`);
                }
                return parsed;
            });
        };

        config.policyHiddenSizes = normalizeHiddenList(config.policyHiddenSizes, DEFAULT_POLICY_HIDDEN);
        config.valueHiddenSizes = normalizeHiddenList(config.valueHiddenSizes, DEFAULT_VALUE_HIDDEN);

        if (!Number.isFinite(config.numEnvs) || config.numEnvs <= 0) {
            config.numEnvs = 1;
        }

        return config;
    }

    _inferLstmUnitsFromNetwork(network) {
        if (!network) {
            return null;
        }
        const output = Array.isArray(network.outputs) ? network.outputs[0] : network.output;
        if (!output || !output.shape || output.shape.length === 0) {
            return null;
        }
        const lastDim = output.shape[output.shape.length - 1];
        if (typeof lastDim === 'number' && Number.isFinite(lastDim)) {
            return lastDim;
        }
        return null;
    }

    _defineFeatureExtractor(hiddenLayerSizes) {
        const stateInput = tf.input({ shape: [this.numStates], name: 'state' });

        const hiddenInitializer = orthogonalInitializer(Math.sqrt(2));
        let features = stateInput;
        hiddenLayerSizes.forEach((size, idx) => {
            features = tf.layers
                .dense({
                    units: size,
                    activation: 'tanh',
                    kernelInitializer: hiddenInitializer,
                    biasInitializer: 'zeros',
                    name: `mlp_${idx}`
                })
                .apply(features);
        });

        const projectionLayer = tf.layers.dense({
            units: this.lstmUnits,
            activation: 'tanh',
            kernelInitializer: hiddenInitializer,
            biasInitializer: 'zeros',
            name: 'feature_projection'
        });

        const projected = projectionLayer.apply(features);

        this.network = tf.model({
            inputs: stateInput,
            outputs: projected,
            name: 'ppo2_baselines_feature_extractor'
        });
    }

    _buildRecurrentHeads() {
        const hiddenInitializer = orthogonalInitializer(Math.sqrt(2));
        const lstmKernelInitializer = orthogonalInitializer();
        const lstmRecurrentInitializer = orthogonalInitializer();
        const policyHeadInitializer = orthogonalInitializer(0.01);
        const valueHeadInitializer = orthogonalInitializer(1.0);

        this.lstmCell = tf.layers.lstmCell({
            units: this.lstmUnits,
            kernelInitializer: lstmKernelInitializer,
            recurrentInitializer: lstmRecurrentInitializer,
            biasInitializer: 'zeros',
            name: 'ppo2_lstm_cell'
        });
        this.lstmCell.build([null, this.lstmUnits]);

        this.policyHiddenLayers = this.policyHiddenSizes.map((units, idx) =>
            tf.layers.dense({
                units,
                activation: 'tanh',
                kernelInitializer: hiddenInitializer,
                biasInitializer: 'zeros',
                name: `policy_latent_${idx}`
            })
        );

        this.policyHeadLayer = tf.layers.dense({
            units: this.numActions,
            activation: 'linear',
            name: 'policy_logits',
            kernelInitializer: policyHeadInitializer,
            biasInitializer: 'zeros'
        });

        this.valueHiddenLayers = this.valueHiddenSizes.map((units, idx) =>
            tf.layers.dense({
                units,
                activation: 'tanh',
                kernelInitializer: hiddenInitializer,
                biasInitializer: 'zeros',
                name: `value_latent_${idx}`
            })
        );

        this.valueHeadLayer = tf.layers.dense({
            units: 1,
            activation: 'linear',
            name: 'value_output',
            kernelInitializer: valueHeadInitializer,
            biasInitializer: 'zeros'
        });
    }

    _normalizeMask(mask, batchSize) {
        if (mask == null) {
            return null;
        }

        if (mask instanceof tf.Tensor) {
            if (mask.shape.length === 0) {
                return mask.reshape([1, 1]).tile([batchSize, 1]);
            }
            if (mask.shape.length === 1 && mask.shape[0] === batchSize) {
                return mask.reshape([batchSize, 1]);
            }
            if (mask.shape.length === 2 && mask.shape[0] === batchSize && mask.shape[1] === 1) {
                return mask;
            }
            throw new Error(`Mask tensor has incompatible shape ${mask.shape}`);
        }

        if (Array.isArray(mask)) {
            if (mask.length !== batchSize) {
                throw new Error(`Mask length ${mask.length} does not match batch size ${batchSize}`);
            }
            const normalized = mask.map((value) => (value ? 1 : 0));
            return tf.tensor2d(normalized, [batchSize, 1], 'float32');
        }

        const scalar = Number(mask) ? 1 : 0;
        return tf.fill([batchSize, 1], scalar);
    }

    forward(states, recurrentState = null, mask = null, training = false) {
        return tf.tidy(() => {
            if (!(states instanceof tf.Tensor)) {
                throw new Error('States must be provided as a tf.Tensor');
            }

            const rank = states.shape.length;
            const batchedStates = rank === 1 ? states.reshape([1, states.shape[0]]) : states;
            const batchSize = batchedStates.shape[0];

            const hasState =
                recurrentState &&
                recurrentState.h instanceof tf.Tensor &&
                recurrentState.c instanceof tf.Tensor;

            const defaultState = hasState ? null : this.getInitialState(batchSize);

            let initialH = hasState ? recurrentState.h.clone() : defaultState.h;
            let initialC = hasState ? recurrentState.c.clone() : defaultState.c;

            const maskTensor = this._normalizeMask(mask, batchSize);
            if (maskTensor) {
                const maskedH = initialH.mul(maskTensor);
                const maskedC = initialC.mul(maskTensor);
                initialH.dispose();
                initialC.dispose();
                initialH = maskedH;
                initialC = maskedC;
                maskTensor.dispose();
            }

            const features = this.network.apply(batchedStates, { training });

            if (this.lstmCell.dropoutMask != null) {
                this.lstmCell.dropoutMask.forEach((maskTensor) => maskTensor.dispose());
                this.lstmCell.dropoutMask = null;
            }
            if (this.lstmCell.recurrentDropoutMask != null) {
                this.lstmCell.recurrentDropoutMask.forEach((maskTensor) => maskTensor.dispose());
                this.lstmCell.recurrentDropoutMask = null;
            }

            const [cellOutput, nextH, nextC] = this.lstmCell.call(
                [features, initialH, initialC],
                { training }
            );

            initialH.dispose();
            initialC.dispose();

            const policyLatent = this.policyHiddenLayers.reduce(
                (acc, layer) => layer.apply(acc, { training }),
                cellOutput
            );
            const policyLogits = this.policyHeadLayer.apply(policyLatent, { training });

            const valueLatent = this.valueHiddenLayers.reduce(
                (acc, layer) => layer.apply(acc, { training }),
                cellOutput
            );
            const valueOut = this.valueHeadLayer.apply(valueLatent, { training });

            return [policyLogits, valueOut, nextH, nextC];
        });
    }

    step(states, recurrentState = null, mask = null, deterministic = false) {
        const result = tf.tidy(() => {
            const [policyLogits, valueTensor, nextH, nextC] = this.forward(
                states,
                recurrentState,
                mask,
                false
            );

            const actionTensor = deterministic
                ? tf.argMax(policyLogits, 1)
                : tf.squeeze(tf.multinomial(policyLogits, 1), [1]);

            const logProbsAll = tf.logSoftmax(policyLogits);
            const actionMask = tf.oneHot(actionTensor, this.numActions);
            const selectedLogProbs = tf.sum(logProbsAll.mul(actionMask), 1);
            const negLogProbs = selectedLogProbs.neg();
            const values = tf.squeeze(valueTensor, [1]);

            return {
                actionTensor: tf.keep(actionTensor),
                negLogProbTensor: tf.keep(negLogProbs),
                valueTensor: tf.keep(values),
                nextState: {
                    h: tf.keep(nextH),
                    c: tf.keep(nextC)
                }
            };
        });

        const extract = (arrayLike) => (arrayLike.length === 1 ? arrayLike[0] : Array.from(arrayLike));

        const actionArray = result.actionTensor.dataSync();
        const negLogArray = result.negLogProbTensor.dataSync();
        const valueArray = result.valueTensor.dataSync();

        const action = extract(actionArray);
        const negLogProb = extract(negLogArray);
        const value = extract(valueArray);

        result.actionTensor.dispose();
        result.negLogProbTensor.dispose();
        result.valueTensor.dispose();

        return { action, negLogProb, value, nextState: result.nextState };
    }

    predict(states, recurrentState = null, mask = null, training = false) {
        const [policyLogits, valueTensor, nextH, nextC] = this.forward(
            states,
            recurrentState,
            mask,
            training
        );
        return [
            policyLogits,
            valueTensor,
            {
                h: nextH,
                c: nextC
            }
        ];
    }

    getInitialState(batchSize = 1) {
        return {
            h: tf.zeros([batchSize, this.lstmUnits]),
            c: tf.zeros([batchSize, this.lstmUnits])
        };
    }

    cloneState(state) {
        if (!state) {
            return this.getInitialState(1);
        }
        return {
            h: state.h.clone(),
            c: state.c.clone()
        };
    }

    disposeState(state) {
        if (!state) {
            return;
        }
        if (state.h instanceof tf.Tensor) {
            state.h.dispose();
        }
        if (state.c instanceof tf.Tensor) {
            state.c.dispose();
        }
    }

    stateToArray(state) {
        if (!state || !(state.h instanceof tf.Tensor) || !(state.c instanceof tf.Tensor)) {
            return {
                h: [],
                c: []
            };
        }
        return {
            h: Array.from(state.h.dataSync()),
            c: Array.from(state.c.dataSync())
        };
    }

    batchStateArraysToTensors(stateArrayBatch) {
        if (!Array.isArray(stateArrayBatch) || stateArrayBatch.length === 0) {
            return {
                h: tf.zeros([0, this.lstmUnits]),
                c: tf.zeros([0, this.lstmUnits])
            };
        }

        const zeroRow = new Array(this.lstmUnits).fill(0);

        const hData = stateArrayBatch.map((state, idx) => {
            if (!state || !Array.isArray(state.h) || state.h.length !== this.lstmUnits) {
                if (!state || state.h == null) {
                    console.warn(`[Model] prevRnnState.h 손상 감지 (index: ${idx}) - 0으로 대체합니다.`);
                } else {
                    console.warn(
                        `[Model] prevRnnState.h 길이 불일치 (index: ${idx}, length: ${state.h.length}) - 0으로 대체합니다.`
                    );
                }
                return zeroRow.slice();
            }
            return state.h;
        });

        const cData = stateArrayBatch.map((state, idx) => {
            if (!state || !Array.isArray(state.c) || state.c.length !== this.lstmUnits) {
                if (!state || state.c == null) {
                    console.warn(`[Model] prevRnnState.c 손상 감지 (index: ${idx}) - 0으로 대체합니다.`);
                } else {
                    console.warn(
                        `[Model] prevRnnState.c 길이 불일치 (index: ${idx}, length: ${state.c.length}) - 0으로 대체합니다.`
                    );
                }
                return zeroRow.slice();
            }
            return state.c;
        });

        return {
            h: tf.tensor2d(hData, [stateArrayBatch.length, this.lstmUnits]),
            c: tf.tensor2d(cData, [stateArrayBatch.length, this.lstmUnits])
        };
    }
}

module.exports = { Model };