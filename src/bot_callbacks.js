const conf = require('./config.js');
const { refreshActionFunction, applyAction, resetAllKeysExceptFor, getBotRelativeGameEnv } = require('./bot_functions.js')
const tf = require('@tensorflow/tfjs-node');



var dataHistory = {};
var delayBeforePlay = conf.MAX_DELAY_BEFORE_PLAY;

async function onBotAuthentification(data, bot, page) {
  bot.roomId = data.roomId;
  console.log("The bot id "+bot.id+" is now authentificate as the bot roomId "+bot.roomId);
}

async function onGameTick(data, bot, page) {
  if(!bot.roomId || data.gameEnded) {
    dataHistory = {};
    await resetAllKeysExceptFor(page);
    return;
  }

  dataHistory[data.tick] = data;
  if(!(data.tick - 1 in dataHistory) || !(data.tick - 2 in dataHistory)) {
    return;
  }
  else {
    Object.keys(dataHistory).filter(tick => tick < data.tick - 2).forEach(tick => delete dataHistory[tick]);
  }

  var environment = getBotRelativeGameEnv(dataHistory, bot);
  if(!environment) {
    return; // the bot is not in the game
  }

  var lastData = dataHistory[data.tick - 1];
  var goalJustScored = lastData.scores.red != data.scores.red || lastData.scores.blue != data.scores.blue;

  if(delayBeforePlay > 0) {
    await resetAllKeysExceptFor(page);
    if(delayBeforePlay < conf.MAX_DELAY_BEFORE_PLAY) {
      if(delayBeforePlay % 8 == 0) {
        applyAction(environment.bot.team, "kick", page);
      }
      else if(delayBeforePlay == 10) {
        applyAction(environment.bot.team, "right", page);
      }
      else if(delayBeforePlay == 20) {
        applyAction(environment.bot.team, "left", page);
      }
      delayBeforePlay--;
    }
    return;
  }

  if(goalJustScored) {
    delayBeforePlay = conf.MAX_DELAY_BEFORE_PLAY;
  }
  bot.policy.haxai.setGameState(environment)
  if(conf.IS_TRAINING&&environment.tick%3==2&&environment.tick>=5){
    bot.policy.orchestrator.nextRun()
  }
  if(environment.bot.team!=0&&environment.tick%3==0){
    var actionName
    try {
      actionName = conf.IS_TRAINING==true ? bot.policy.train() : 
      tf.tidy(() => {
        const action = bot.policy.model.chooseAction(bot.policy.haxai.getStateTensor(), 0);
        return bot.policy.haxai.update(action);
      });
    } catch (error) {
      console.error(error);
      actionName = "none";
    }
  
    applyAction(environment.bot.team, actionName, page);
  }
}

async function onPositionsReset(data, bot, page) {
  delayBeforePlay = conf.DELAY_BEFORE_PLAY;
}

async function onGameStart(data, bot, page) {
  delayBeforePlay = conf.DELAY_BEFORE_PLAY;
  lastTickData = null;
}
async function onGameStop(data, bot, page) {
  if(conf.IS_TRAINING&&bot.policy.haxai.getState().bot_Team!=0){
    await bot.policy.orchestrator.stop()
    await bot.policy.saveModel();
  }
}

async function onActionFileRefresh(data, bot, page) {
  if(data.actionFile) {
    bot.actionFile = data.actionFile;
  }
  refreshActionFunction(bot);
}

module.exports = { onBotAuthentification, onGameTick, onPositionsReset, onGameStart, onGameStop, onActionFileRefresh };
