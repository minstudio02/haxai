const {connect} = require('puppeteer-real-browser');
const botCallbacks = require('./bot_callbacks.js');
const { refreshActionFunction } = require('./bot_functions.js');
const conf = require('./config.js');
const {SaveablePolicyNetwork} = require('./policy');


const botId = parseInt(process.argv[2]);
if (!botId) {
    throw "Please provide a bot name as a first argument";
}

const roomLink = process.argv[3];
if (!roomLink) {
    throw "Please provide URL as a second argument";
}

const adminToken = process.argv[4];
if (!adminToken) {
    throw "Please provide an admin token as a third argument";
}

const bot = {
  id: botId,
  name: "bot-"+botId,
  adminToken: adminToken
};
async function getPolicy(){
  let policyNet;
  if (await SaveablePolicyNetwork.checkStoredModelStatus() != null) {
    policyNet = await SaveablePolicyNetwork.loadModel();
  }else{
    const hiddenLayerSizes =
    '128,128,128,128'.trim().split(',').map(v => {
      const num = Number.parseInt(v.trim());
      if (!(num > 0)) {
        throw new Error(
            `Invalid hidden layer sizes string: ` +
            `${128}`);
      }
      return num;
    });
    policyNet = new SaveablePolicyNetwork(hiddenLayerSizes);
    console.log('DONE constructing new instance of SaveablePolicyNetwork');
  }
  return policyNet
}

const roomPassword = process.argv[5];
let _browser = null;

async function run () {
    bot.policy= await getPolicy();
    const {browser, page} = await connect({headless:true});
    _browser = browser;
    await new Promise((resolve) => setTimeout(resolve, 11000));
    await page.goto(roomLink);
    await page.waitForSelector("iframe");

    var frames = page.frames();
    var myframe = frames.find(f => f.url().indexOf("__cache_static__/g/game.html") > -1);

    const inputName = await myframe.$("input[type=text]");
    await inputName.type(bot.name,{delay: 200});
    const buttonName = await myframe.$("button");
    await buttonName.click({delay: 500});

    if(roomPassword) {
      const inputPassword = await myframe.$("input[data-hook=input]");
      await inputPassword.type(roomPassword,{delay: 200});
      const buttonPassword = await myframe.$("button[data-hook=ok]");
      await buttonPassword.click({delay: 500});
    }

    try {
      await myframe.waitForSelector(".icon-menu");
    } catch (error) {
      console.error(error);
      await page.screenshot({path: bot.name+'.png'});
    }
    await myframe.waitForSelector(".icon-menu", {timeout: (999999999)});

    await sendChat(page, "/avatar AI");
    await new Promise((page) => setTimeout(page, 1000));
    await sendChat(page, "!bot "+bot.adminToken+" "+bot.id);

    process.on('message', (message) => onServerMessage(message, page));
    while(await myframe.$(".icon-menu")) {
      await new Promise((page) => setTimeout(page, 10000));
    }
    cleanExit();
}

async function sendChat(page, message) {
  await page.keyboard.press('Tab');
  await page.keyboard.type(message);
  await page.keyboard.press('Enter');
}

async function onServerMessage(message, page) {
  if(message.callback in botCallbacks && typeof botCallbacks[message.callback] === "function") {
    botCallbacks[message.callback](message.data, bot, page);
  }
  else {
    console.error("The following client callback function doesn't exist : "+message.callback);
  }
}

function cleanExit() {
  _browser.close();
  console.log(bot.name+"'s process exited.");
  process.exit();
};
process.on('SIGINT', cleanExit); // catch ctrl-c
process.on('SIGTERM', cleanExit); // catch kill

run();