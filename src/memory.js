const {sampleSize} = require('lodash');




class Memory {
    /**
     * @param {number} maxMemory
     */
    constructor(maxMemory) {
        this.maxMemory = maxMemory;
        this.samples = new Array();
    }

    /**
     * @param {Array} sample
     */
    addSample(sample) {
        this.samples.push(sample);
        if (this.samples.length > this.maxMemory) {
            let [state,,, nextState,,,] = this.samples.shift();
            state.dispose();
            nextState.dispose();
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
}
module.exports = { Memory };