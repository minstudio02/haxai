const {sampleSize} = require('lodash');
const tf = require('@tensorflow/tfjs-node');

class Memory {
    /**
     * @param {number} maxMemory
     */
    constructor(maxMemory) {
        this.maxMemory = maxMemory;
        this.samples = [];
    }

    _sanitizeScalar(value, tag, fallback = 0) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            console.warn(`[Memory] ${tag} 값이 유효하지 않아 ${fallback}으로 대체합니다.`);
            return fallback;
        }
        return numeric;
    }

    _tensorToArray(tensor, tag) {
        if (!(tensor instanceof tf.Tensor)) {
            console.warn(`[Memory] ${tag}가 Tensor가 아님 - null로 대체`);
            return null;
        }
        if (tensor.isDisposedInternal) {
            console.warn(`[Memory] ${tag}가 이미 dispose됨`);
            return null;
        }
        try {
            const data = tensor.dataSync();
            return Array.from(data);
        } catch (error) {
            console.error(`[Memory] ${tag} 데이터 변환 실패: ${error.message}`);
            return null;
        }
    }

    _disposeTensor(tensor, tag) {
        if (tensor && tensor instanceof tf.Tensor && !tensor.isDisposedInternal) {
            try {
                tensor.dispose();
            } catch (error) {
                console.warn(`[Memory] ${tag} dispose 실패: ${error.message}`);
            }
        }
    }

    /**
     * @param {Array} sample
     */
    addSample(sample) {
        const stateArray = this._tensorToArray(sample.state, 'state');
        const nextStateArray = this._tensorToArray(sample.nextState, 'nextState');

        this._disposeTensor(sample.state, 'state');
        this._disposeTensor(sample.nextState, 'nextState');

        if (!stateArray || !nextStateArray) {
            console.warn('[Memory] 유효하지 않은 샘플이 감지되어 저장을 건너뜁니다.');
            return;
        }

        const storedSample = {
            ...sample,
            state: stateArray,
            nextState: nextStateArray,
            negLogProb: this._sanitizeScalar(sample.negLogProb, 'negLogProb'),
            value: this._sanitizeScalar(sample.value, 'value'),
            reward: this._sanitizeScalar(sample.reward, 'reward'),
            mask: this._sanitizeScalar(sample.mask != null ? sample.mask : 1, 'mask', 1),
            done: Boolean(sample.done)
        };

        this.samples.push(storedSample);
        if (this.samples.length > this.maxMemory) {
            this.samples.shift();
        }
    }

    /**
     * @param {number} nSamples
     * @returns {Array} Randomly selected samples
     */
    sample(nSamples) {
        // 메모리에 충분한 샘플이 없으면 빈 배열 반환
        if (this.samples.length < nSamples) {
            return [];
        }
        return sampleSize(this.samples, nSamples);
    }

    /**
     * 메모리 초기화
     */
    clear() {
        this.samples = [];
    }

    size() {
        return this.samples.length;
    }

    isReady(batchSize) {
        return this.samples.length >= batchSize;
    }
}
module.exports = { Memory };