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

    run() {
        this.state = this.haxai.getStateTensor();

        // Interaction with the environment
        this.action = this.model.chooseAction(this.state, this.eps);
        const actionName = this.haxai.update(this.action);

        return actionName
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
}
module.exports = { Orchestrator };