const tf = require('@tensorflow/tfjs-node');
const { applyAction } = require('./bot_functions.js')

const ACTION_MAP = [
  'kick',
  'forward',
  'forward-left',
  'left',
  'backward-left',
  'backward',
  'backward-right',
  'right',
  'forward-right',
  'none'
];
/**
 * Mountain car system simulator.
 *
 * There are two state variables in this system:
 *
 *   - position: The x-coordinate of location of the car.
 *   - velocity: The velocity of the car.
 *
 * The system is controlled through three distinct actions:
 *
 *   - leftward acceleration.
 *   - rightward acceleration
 *   - no acceleration
 */
class HaxAI {
  /**
   * Constructor of MountainCar.
   */
  constructor() {
      // Constants that characterize the system.

    this.bot_Position = {x: 0, y: 0};
    this.bot_Speed = {x: 0, y: 0};
    this.op_Position = {x: 0, y: 0};
    this.op_Speed = {x: 0, y: 0};
    this.ball_Position = {x: 0, y: 0};
    this.ball_Speed = {x: 0, y: 0};
    this.dist_bw_myball = 0;
    this.field_Locked = 1
    this.bot_Team = 0
    this.start_Team = 1
    this.game_State = false
    this.score

  }

  /**
   * Set the state of the mountain car system randomly.
   */
  setGameState(env) {
    // The state variables of the mountain car system.
    this.bot_Position = env.bot.position;
    this.bot_Speed = env.bot.velocity;
    this.op_Position = env.opponents[0] ? env.opponents[0].position : {x: 0, y: 0};
    this.op_Speed = env.opponents[0] ? env.opponents[0].velocity : {x: 0, y: 0};
    this.ball_Position = env.ball.position;
    this.ball_Speed = env.ball.velocity;
    this.dist_bw_myball = Math.sqrt(Math.pow(env.bot.position.x - env.ball.position.x, 2) + Math.pow(env.bot.position.y - env.ball.position.y, 2));
    this.field_Locked = env.bot.team == 1?
      env.model.start_Team == 2 && !env.model.game_State :
      env.model.start_Team == 1 && !env.model.game_State
    this.bot_Team = env.bot.team
    this.start_Team = env.model.start_Team
    this.game_State = env.model.game_State
    this.score = env.score
  } 

  /**
   * Get current state as a tf.Tensor of shape [1, 2].
   */
  getStateTensor() {
    return tf.tensor2d([[
      this.bot_Position.x, 
      this.bot_Position.y,
      this.bot_Speed.x, 
      this.bot_Speed.y,
      this.op_Position.x, 
      this.op_Position.y,
      this.op_Speed.x, 
      this.op_Speed.y,
      this.ball_Position.x, 
      this.ball_Position.y,
      this.ball_Speed.x, 
      this.ball_Speed.y,
      this.dist_bw_myball,
      this.field_Locked]]);
  }
  getState() {
    return {
      bot_Team: this.bot_Team,
      start_Team: this.start_Team,
      game_State: this.game_State,
      score: this.score
    };
  }

  /**
   * Update the mountain car system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   Action is an integer, in [-1, 0, 1]
   *   A value of 1 leads to a rightward force of a fixed magnitude.
   *   A value of -1 leads to a leftward force of the same fixed magnitude.
   *   A value of 0 leads to no force applied.
   * @returns {bool} Whether the simulation is done.
   */
  async update(action,page) {
    const resolvedIndex = Number.isInteger(action) ? action : ACTION_MAP.length - 1;
    const boundedIndex = resolvedIndex >= 0 && resolvedIndex < ACTION_MAP.length
      ? resolvedIndex
      : ACTION_MAP.length - 1;
    const command = ACTION_MAP[boundedIndex];
    await applyAction(this.bot_Team, command, page);

  }

   /**
   * Determine whether this simulation is done.
   *
   * A simulation is done when `position` reaches `goalPosition`
   * and `velocity` is greater than zero.
   *
   * @returns {bool} Whether the simulation is done.
   */
  isDone() {
    return this.score == undefined ? false : this.score.ownTeam > this.score.opponentTeam
  }
}
module.exports = { HaxAI };