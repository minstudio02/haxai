/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * TensorFlow.js Reinforcement Learning Example: Balancing a Cart-Pole System.
 *
 * The simulation, training, testing and visualization parts are written
 * purely in JavaScript and can run in the web browser with WebGL acceleration.
 *
 * This reinforcement learning (RL) problem was proposed in:
 *
 * - Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
 *   Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
 *   Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983
 * - Sutton, "Temporal Aspects of Credit Assignment in Reinforcement Learning",
 *   Ph.D. Dissertation, Department of Computer and Information Science,
 *   University of Massachusetts, Amherst, 1984.
 *
 * It later became one of OpenAI's gym environmnets:
 *   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 */

const tf =  require('@tensorflow/tfjs-node');

const {HaxAI} = require('./haxai');
const {Model} = require('./AImodel');
const {Memory} = require('./memory');
const {Orchestrator} = require('./orchestrator');
const fs = require('fs');

/**
 * Policy network for controlling the cart-pole system.
 *
 * The role of the policy network is to select an action based on the observed
 * state of the system. In this case, the action is the leftward or rightward
 * force and the observed system state is a four-dimensional vector, consisting
 * of cart position, cart velocity, pole angle and pole angular velocity.
 *
 */
class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[] | tf.LayersModel} hiddenLayerSizes
   *   Can be any of the following
   *   - Size of the hidden layer, as a single number (for a single hidden
   *     layer)
   *   - An Array of numbers (for any number of hidden layers).
   *   - An instance of tf.LayersModel.
   */
  constructor(hiddenLayerSizesOrModel) {
    this.haxai = new HaxAI()
    this.memory = new Memory(10000);
    this.model = new Model(hiddenLayerSizesOrModel, 14, 10, 100)
    this.orchestrator= new Orchestrator(
      this.haxai,
      this.model,
      this.memory,
      0.95
    )
  }

  /**
   * Train the policy network's model.
   *
   * @param {CartPole} haxAISystem The cart-pole system object to use during
   *   training.
   * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
   *   Optimizer to use for training.
   * @param {number} discountRate Reward discounting rate: a number between 0
   *   and 1.
   * @param {number} numGames Number of game to play for each model parameter
   *   update.
   * @param {number} maxStepsPerGame Maximum number of steps to perform during
   *   a game. If this number is reached, the game will end immediately.
   * @returns {number[]} The number of steps completed in the `numGames` games
   *   in this round of training.
   */
  train() {

      // Randomly initialize the state of the cart-pole system at the beginning
      // of every game.
      return this.orchestrator.run();
  }
}

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = `file://./haxai_model`;
const MODEL_LOAD_PATH_ = `file://./haxai_model/model.json`;

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
class SaveablePolicyNetwork extends PolicyNetwork {
  /**
   * Constructor of SaveablePolicyNetwork
   *
   * @param {number | number[]} hiddenLayerSizesOrModel
   */
  constructor(hiddenLayerSizesOrModel) {
    super(hiddenLayerSizesOrModel);
  }

  /**
   * Save the model to IndexedDB.
   */
  async saveModel() {
    return await this.model.network.save(MODEL_SAVE_PATH_);
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns {SaveablePolicyNetwork} The instance of loaded
   *   `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
  static async loadModel() {
    if (fs.readFileSync(`./haxai_model/model.json`)!=null) {
      console.log(`Loading existing model...`);
      const model = await tf.loadLayersModel(MODEL_LOAD_PATH_);
      console.log(`Loaded model from ${MODEL_LOAD_PATH_}`);
      return new SaveablePolicyNetwork(model);
    } else {
      throw new Error(`Cannot find model at ${MODEL_LOAD_PATH_}.`);
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  static async checkStoredModelStatus() {
    try {
      return fs.readFileSync(`./haxai_model/model.json`)
    } catch (error) {
      return null
    }
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_LOAD_PATH_);
  }

  /**
   * Get the sizes of the hidden layers.
   *
   * @returns {number | number[]} If the model has only one hidden layer,
   *   return the size of the layer as a single number. If the model has
   *   multiple hidden layers, return the sizes as an Array of numbers.
   */
  hiddenLayerSizes() {
    const sizes = [];
    for (let i = 0; i < this.model.network.layers.length - 1; ++i) {
      sizes.push(this.model.network.layers[i].units);
    }
    return sizes.length === 1 ? sizes[0] : sizes;
  }
}
module.exports = { SaveablePolicyNetwork };
