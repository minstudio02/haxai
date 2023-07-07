const {HaxAI} = require('./haxai');
const {Model} = require('./AImodel');
const {Memory} = require('./memory');

const tf = require('@tensorflow/tfjs-node');

const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

class Orchestrator {
    /**
     * @param {HaxAI} haxai
     * @param {Model} model
     * @param {Memory} memory
     * @param {number} discountRate
     */
    constructor(haxai, model, memory, discountRate) {
        // The main components of the environment
        this.haxai = haxai;
        this.model = model;
        this.memory = memory;

        // The exploration parameter
        this.eps = MAX_EPSILON;

        // Keep tracking of the elapsed steps
        this.steps = 0;

        this.discountRate = discountRate;

        // Initialization of the rewards and max positions containers
        this.rewardStore = new Array();
        this.totalReward = 0

        this.state
        this.action = -1
        this.not_started_yet = 0
        this.own_score=0
        this.opponent_score=0
    }

    /**
     * @param {number} status
     * @returns {number} Reward corresponding to the position
     */
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

        let vett_palla_porta = [320 - status[8], status[9]]
        reward += prodotto_scalare(vett_palla_porta, [status[10], status[11]])


        if (!status[13]){
            let velocita_palla = Math.sqrt(status[10] ** 2 + status[11] ** 2)
            reward -= 2000 * Math.max(0.0, 0.1 - velocita_palla)
        }

        if (status[8] < status[0]) reward -= (status[0] - status[8])


        if (more_state.bot_Team == more_state.start_Team && !more_state.game_State){
            reward -= 0.25 * this.not_started_yet
            this.not_started_yet += 1
        }
        else  this.not_started_yet = 0

        let goal_reward = 0
        if(this.own_score!=more_state.score.ownTeam||this.opponent_score!=more_state.score.opponentTeam){
            if (this.own_score!=more_state.score.ownTeam){
                goal_reward = 50000
                this.own_score=more_state.score.ownTeam
            }
            else if (this.opponent_score!=more_state.score.opponentTeam){
                goal_reward = -5000
                this.opponent_score=more_state.score.opponentTeam  
            }
        }
        reward += goal_reward
        return reward/1000
    }
    sleep (time) {
      return new Promise((resolve) => setTimeout(resolve, time));
    }
    async run(page) {
        if(this.steps==0) this.state = this.haxai.getStateTensor();

        // Interaction with the environment
        this.action = this.model.chooseAction(this.state, this.eps);
        await this.haxai.update(this.action,page);
        this.sleep(2000/60).then(()=>{
          this.nextRun()
        })
    }
    nextRun(){
        let nextState = this.haxai.getStateTensor();
        const reward = this.computeReward(nextState.arraySync()[0],this.haxai.getState());

        this.memory.addSample([this.state, this.action, reward, nextState]);

        this.steps += 1;
        // Exponentially decay the exploration parameter
        this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);

        this.state = nextState;
        this.totalReward += reward;
    }
    async stop() {
        this.rewardStore.push(this.totalReward);
        console.log(this.totalReward)
        this.totalReward = 0;
        this.steps = 0;
        this.eps = MAX_EPSILON
        await this.replay()
    }

    async replay() {
        // Sample from memory
        const batch = this.memory.sample(this.model.batchSize);
        const states = batch.map(([state, , , ]) => state);
        const nextStates = batch.map(
            ([, , , nextState]) => nextState ? nextState : tf.zeros([this.model.numStates])
        );
        // Predict the values of each action at each state
        const qsa = states.map((state) => this.model.predict(state));
        // Predict the values of each action at each next state
        const qsad = nextStates.map((nextState) => this.model.predict(nextState));

        let x = new Array();
        let y = new Array();

        // Update the states rewards with the discounted next states rewards
        batch.forEach(
            ([state, action, reward, nextState], index) => {
                const currentQ = qsa[index];
                currentQ[action] = nextState ? reward + this.discountRate * qsad[index].max().dataSync() : reward;
                x.push(state.dataSync());
                y.push(currentQ.dataSync());
            }
        );

        // Clean unused tensors
        qsa.forEach((state) => state.dispose());
        qsad.forEach((state) => state.dispose());

        // Reshape the batches to be fed to the network
        x = tf.tensor2d(x, [x.length, this.model.numStates])
        y = tf.tensor2d(y, [y.length, this.model.numActions])

        // Learn the Q(s, a) values given associated discounted rewards
        await this.model.train(x, y);

        x.dispose();
        y.dispose();
    }
    getGradientsAndSaveActions(inputTensor) {
        const f = () => tf.tidy(() => {
          const [logits, actions] = this.getLogitsAndActions(inputTensor);
          this.currentActions_ = actions.dataSync();
          const labels =
              tf.sub(1, tf.tensor2d(this.currentActions_, actions.shape));
          return tf.losses.sigmoidCrossEntropy(labels, logits).asScalar();
        });
        return tf.variableGrads(f);
      }
    
      getCurrentActions() {
        return this.currentActions_;
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
          const logits = this.policyNet.predict(inputs);
    
          // Get the probability of the leftward action.
          const leftProb = tf.sigmoid(logits);
          // Probabilites of the left and right actions.
          const leftRightProbs = tf.concat([leftProb, tf.sub(1, leftProb)], 1);
          const actions = tf.multinomial(leftRightProbs, 1, null, true);
          return [logits, actions];
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
    
      /**
       * Push a new dictionary of gradients into records.
       *
       * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
       *   gradient: a map from variable name to the Array of gradient values for
       *   the variable.
       * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
       *   into `record`: a map from variable name to the gradient Tensor.
       */
      pushGradients(record, gradients) {
        for (const key in gradients) {
          if (key in record) {
            record[key].push(gradients[key]);
          } else {
            record[key] = [gradients[key]];
          }
        }
      }
}
/**
 * Discount the reward values.
 *
 * @param {number[]} rewards The reward values to be discounted.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
 */
function discountRewards(rewards, discountRate) {
    const discountedBuffer = tf.buffer([rewards.length]);
    let prev = 0;
    for (let i = rewards.length - 1; i >= 0; --i) {
      const current = discountRate * prev + rewards[i];
      discountedBuffer.set(current, i);
      prev = current;
    }
    return discountedBuffer.toTensor();
  }
  
  /**
   * Discount and normalize reward values.
   *
   * This function performs two steps:
   *
   * 1. Discounts the reward values using `discountRate`.
   * 2. Normalize the reward values with the global reward mean and standard
   *    deviation.
   *
   * @param {number[][]} rewardSequences Sequences of reward values.
   * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
   *   0.95.
   * @returns {tf.Tensor[]} The discounted and normalize reward values as an
   *   Array of tf.Tensor.
   */
  function discountAndNormalizeRewards(rewardSequences, discountRate) {
    return tf.tidy(() => {
      const discounted = [];
      for (const sequence of rewardSequences) {
        discounted.push(discountRewards(sequence, discountRate))
      }
      // Compute the overall mean and stddev.
      const concatenated = tf.concat(discounted);
      const mean = tf.mean(concatenated);
      const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
      // Normalize the reward sequences using the mean and std.
      const normalized = discounted.map(rs => rs.sub(mean).div(std));
      return normalized;
    });
  }
  
  /**
   * Scale the gradient values using normalized reward values and compute average.
   *
   * The gradient values are scaled by the normalized reward values. Then they
   * are averaged across all games and all steps.
   *
   * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
   *   name to all the gradient values for the variable across all games and all
   *   steps.
   * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
   *   for all the games. Each element of the Array is a 1D tf.Tensor of which
   *   the length equals the number of steps in the game.
   * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
   *   for the variables.
   */
  function scaleAndAverageGradients(allGradients, normalizedRewards) {
    return tf.tidy(() => {
      const gradients = {};
      for (const varName in allGradients) {
        gradients[varName] = tf.tidy(() => {
          // Stack gradients together.
          const varGradients = allGradients[varName].map(
              varGameGradients => tf.stack(varGameGradients));
          // Expand dimensions of reward tensors to prepare for multiplication
          // with broadcasting.
          const expandedDims = [];
          for (let i = 0; i < varGradients[0].rank - 1; ++i) {
            expandedDims.push(1);
          }
          const reshapedNormalizedRewards = normalizedRewards.map(
              rs => rs.reshape(rs.shape.concat(expandedDims)));
          for (let g = 0; g < varGradients.length; ++g) {
            // This mul() call uses broadcasting.
            varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
          }
          // Concatenate the scaled gradients together, then average them across
          // all the steps of all the games.
          return tf.mean(tf.concat(varGradients, 0), 0);
        });
      }
      return gradients;
    });
  }
  
module.exports = { Orchestrator };