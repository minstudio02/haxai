

async function createHaxballRoom(serverName, password, recaptchaToken, adminToken, numberOfPlayersPerTeam) {
  if(numberOfPlayersPerTeam <= 0) {
    throw "numberOfPlayersPerTeam must be greather than 0";
  }
  const teamNames = ["Spectators", "Red team", "Blue team"];
  var tickNumber = 0;
  var announceNoOvertime = false;
  var gameEnded = false;
  var updateTeamsInProgress = false;
  var kickOffDuration = 0;
  var start_team=1
  var game_started=false

  var room = window.HBInit({
    roomName: serverName,
    password: password ? password : null,
    maxPlayers: 8,
    noPlayer: true,
    public: true,
    token: recaptchaToken
  });
  room.players = {};
  var stadiumFileText = `
  {
      "name": "G.Buffon Hot Small&[GB]",
      "width": 420,
      "height": 200,
      "spawnDistance": 130,
      "bg": {
          "type": "grass",
          "width": 320,
          "height": 130,
          "kickOffRadius": 70,
          "cornerRadius": 0
      },
      "vertexes": [{
          "x": -320,
          "y": 130,
          "cMask": ["wall"],
          "trait": "ballArea",
          "color": "C7E6BD",
          "curve": 0
      }, {
          "x": -320,
          "y": 55,
          "trait": "ballArea"
      }, {
          "x": -320,
          "y": -55,
          "trait": "ballArea"
      }, {
          "x": -320,
          "y": -130,
          "cMask": ["wall"],
          "trait": "ballArea",
          "color": "C7E6BD"
      }, {
          "x": 320,
          "y": 130,
          "cMask": ["wall"],
          "trait": "ballArea",
          "color": "C7E6BD",
          "curve": 0
      }, {
          "x": 320,
          "y": 55,
          "trait": "ballArea"
      }, {
          "x": 320,
          "y": -55,
          "trait": "ballArea"
      }, {
          "x": 320,
          "y": -130,
          "cMask": ["wall"],
          "trait": "ballArea",
          "color": "C7E6BD"
      }, {
          "x": 0,
          "y": 200,
          "trait": "kickOffBarrier"
      }, {
          "x": 0,
          "y": 70,
          "cMask": ["wall"],
          "trait": "kickOffBarrier",
          "curve": -180,
          "color": "C7E6BD"
      }, {
          "x": 0,
          "y": -70,
          "cMask": ["wall"],
          "trait": "kickOffBarrier",
          "curve": -180,
          "color": "C7E6BD"
      }, {
          "x": 0,
          "y": -200,
          "trait": "kickOffBarrier"
      }, {
          "x": -320,
          "y": -55,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": -10
      }, {
          "x": -355,
          "y": -40,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": -20
      }, {
          "x": -355,
          "y": 40,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": -20
      }, {
          "x": -320,
          "y": 55,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": -10
      }, {
          "x": 320,
          "y": -55,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": 10
      }, {
          "x": 355,
          "y": -40,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": 20
      }, {
          "x": 355,
          "y": 40,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": 20
      }, {
          "x": 320,
          "y": 55,
          "trait": "goalNet",
          "color": "C7E6BD",
          "curve": 10
      }, {
          "x": 320,
          "y": -55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": -320,
          "y": 55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": 320,
          "y": 55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": -320,
          "y": -55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": 0,
          "y": -130,
          "cMask": ["wall"],
          "curve": 0,
          "color": "C7E6BD"
      }, {
          "x": 0,
          "y": 130,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": -320,
          "y": -55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": -320,
          "y": 55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": 320,
          "y": -55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": 320,
          "y": 55,
          "cMask": ["wall"],
          "color": "C7E6BD"
      }, {
          "x": 322,
          "y": 60,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": 322,
          "y": 130,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": 324,
          "y": 60,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 324,
          "y": 130,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 326,
          "y": 60,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 326,
          "y": 130,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 328,
          "y": 60,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 328,
          "y": 130,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 330,
          "y": 60,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 330,
          "y": 130,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 332,
          "y": 60,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 332,
          "y": 130,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 332,
          "y": 59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 332,
          "y": 129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 334,
          "y": 59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 334,
          "y": 129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 336,
          "y": 59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 336,
          "y": 129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 338,
          "y": 59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 338,
          "y": 129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 340,
          "y": 59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 340,
          "y": 129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 320,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": 320,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": 322,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 322,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 324,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 324,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 326,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 326,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 328,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 328,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 330,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 330,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 330,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 330,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 332,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 332,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 334,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 334,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 336,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 336,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 338,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": 338,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -340,
          "y": 58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": -340,
          "y": 128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": -338,
          "y": 58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -338,
          "y": 128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -336,
          "y": 58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -336,
          "y": 128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -334,
          "y": 58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -334,
          "y": 128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -332,
          "y": 58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -332,
          "y": 128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": 58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": 128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": 57,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": 127,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -328,
          "y": 57,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -328,
          "y": 127,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -326,
          "y": 57,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -326,
          "y": 127,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -324,
          "y": 57,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -324,
          "y": 127,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -322,
          "y": 57,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -322,
          "y": 127,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -340,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": -340,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea",
          "curve": 0
      }, {
          "x": -338,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -338,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -336,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -336,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -334,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -334,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -332,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -332,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": -128,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": -58,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -330,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -328,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -328,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -326,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -326,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -324,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -324,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -322,
          "y": -129,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "x": -322,
          "y": -59,
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }],
      "segments": [{
          "v0": 0,
          "v1": 1,
          "trait": "ballArea"
      }, {
          "v0": 2,
          "v1": 3,
          "trait": "ballArea"
      }, {
          "v0": 4,
          "v1": 5,
          "trait": "ballArea"
      }, {
          "v0": 6,
          "v1": 7,
          "trait": "ballArea"
      }, {
          "v0": 12,
          "v1": 13,
          "curve": -10,
          "color": "C7E6BD",
          "trait": "goalNet"
      }, {
          "v0": 13,
          "v1": 14,
          "curve": -20,
          "color": "C7E6BD",
          "trait": "goalNet"
      }, {
          "v0": 14,
          "v1": 15,
          "curve": -10,
          "color": "C7E6BD",
          "trait": "goalNet"
      }, {
          "v0": 16,
          "v1": 17,
          "curve": 10,
          "color": "C7E6BD",
          "trait": "goalNet"
      }, {
          "v0": 17,
          "v1": 18,
          "curve": 20,
          "color": "C7E6BD",
          "trait": "goalNet"
      }, {
          "v0": 18,
          "v1": 19,
          "curve": 10,
          "color": "C7E6BD",
          "trait": "goalNet"
      }, {
          "v0": 8,
          "v1": 9,
          "trait": "kickOffBarrier"
      }, {
          "v0": 9,
          "v1": 10,
          "curve": 180,
          "cGroup": ["blueKO"],
          "trait": "kickOffBarrier"
      }, {
          "v0": 9,
          "v1": 10,
          "curve": -180,
          "cGroup": ["redKO"],
          "trait": "kickOffBarrier"
      }, {
          "v0": 10,
          "v1": 11,
          "trait": "kickOffBarrier"
      }, {
          "v0": 3,
          "v1": 7,
          "color": "C7E6BD",
          "bCoef": 0.5,
          "cMask": ["wall"]
      }, {
          "v0": 7,
          "v1": 20,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 4,
          "v1": 0,
          "curve": 0,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 0,
          "v1": 21,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 4,
          "v1": 22,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 3,
          "v1": 23,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 10,
          "v1": 24,
          "curve": 0,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 10,
          "v1": 9,
          "curve": 180,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 10,
          "v1": 9,
          "curve": -180,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 9,
          "v1": 25,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 26,
          "v1": 27,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 28,
          "v1": 29,
          "color": "C7E6BD",
          "cMask": ["wall"]
      }, {
          "v0": 30,
          "v1": 31,
          "curve": 0,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 32,
          "v1": 33,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 34,
          "v1": 35,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 36,
          "v1": 37,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 38,
          "v1": 39,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 40,
          "v1": 41,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 42,
          "v1": 43,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 44,
          "v1": 45,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 46,
          "v1": 47,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 48,
          "v1": 49,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 50,
          "v1": 51,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 52,
          "v1": 53,
          "curve": 0,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 54,
          "v1": 55,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 56,
          "v1": 57,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 58,
          "v1": 59,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 60,
          "v1": 61,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 62,
          "v1": 63,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 64,
          "v1": 65,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 66,
          "v1": 67,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 68,
          "v1": 69,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 70,
          "v1": 71,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 72,
          "v1": 73,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 74,
          "v1": 75,
          "curve": 0,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 76,
          "v1": 77,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 78,
          "v1": 79,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 80,
          "v1": 81,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 82,
          "v1": 83,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 84,
          "v1": 85,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 86,
          "v1": 87,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 88,
          "v1": 89,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 90,
          "v1": 91,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 92,
          "v1": 93,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 94,
          "v1": 95,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 96,
          "v1": 97,
          "curve": 0,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 98,
          "v1": 99,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 100,
          "v1": 101,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 102,
          "v1": 103,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 104,
          "v1": 105,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 106,
          "v1": 107,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 108,
          "v1": 109,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 110,
          "v1": 111,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 112,
          "v1": 113,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 114,
          "v1": 115,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }, {
          "v0": 116,
          "v1": 117,
          "vis": false,
          "color": "C7E6BD",
          "bCoef": 1,
          "cMask": ["ball"],
          "trait": "ballArea"
      }],
      "goals": [{
          "p0": [-320, -55],
          "p1": [-320, 55],
          "team": "red"
      }, {
          "p0": [320, 55],
          "p1": [320, -55],
          "team": "blue"
      }],
      "discs": [{
          "radius": 5.5,
          "pos": [-320, 55],
          "color": "FFFFFF",
          "bCoef": 1,
          "trait": "goalPost"
      }, {
          "radius": 5.5,
          "pos": [-320, -55],
          "color": "FFFFFF",
          "bCoef": 1,
          "trait": "goalPost"
      }, {
          "radius": 5.5,
          "pos": [320, 55],
          "color": "FFFFFF",
          "bCoef": 1,
          "trait": "goalPost"
      }, {
          "radius": 5.5,
          "pos": [320, -55],
          "color": "FFFFFF",
          "bCoef": 1,
          "trait": "goalPost"
      }],
      "planes": [{
          "normal": [0, 1],
          "dist": -130,
          "trait": "ballArea"
      }, {
          "normal": [0, -1],
          "dist": -130,
          "trait": "ballArea"
      }, {
          "normal": [0, 1],
          "dist": -200,
          "bCoef": 0.1
      }, {
          "normal": [0, -1],
          "dist": -200,
          "bCoef": 0.1
      }, {
          "normal": [1, 0],
          "dist": -420,
          "bCoef": 0.1
      }, {
          "normal": [-1, 0],
          "dist": -420,
          "bCoef": 0.1
      }],
      "traits": {
          "ballArea": {
              "vis": false,
              "bCoef": 1,
              "cMask": ["ball"]
          },
          "goalPost": {
              "radius": 8,
              "invMass": 0,
              "bCoef": 0.5
          },
          "goalNet": {
              "vis": true,
              "bCoef": 0.1,
              "cMask": ["ball"]
          },
          "kickOffBarrier": {
              "vis": false,
              "bCoef": 0.1,
              "cGroup": ["redKO", "blueKO"],
              "cMask": ["red", "blue"]
          }
      },
      "ballPhysics": {
          "radius": 6.4,
          "bCoef": 0.5,
          "damping": 0.99,
          "color": "7cfc00",
          "invMass": 1,
          "cMask": ["all"],
          "cGroup": ["ball"]
      },
      "playerPhysics": {
          "kickStrength": 5.7,
          "kickingDamping": 0.97,
          "kickingAcceleration": 0.073,
          "acceleration": 0.105,
          "damping": 0.96,
          "invMass": 0.3,
          "bCoef": 0.5
      }
  }
  `
  
  room.setTimeLimit(1);
  room.setScoreLimit(0);
  room.setCustomStadium(stadiumFileText)

  room.onPlayerJoin = async (player) => await onPlayerJoinListener(player);
  onPlayerJoinListener = async function(player) {
    let ip = decodeURIComponent(player.conn.replace(/(..)/g,'%$1'))
    const request = await fetch(`https://ipapi.co/${ip}/json/`);
    const response = await request.json();

    if(response.country != 'KR'){
      room.kickPlayer(player.id, 'Access Error', true); // kick
      return;
    }
    if(['Bucheon-si','Gyeyang-gu','Bupyeong-gu'].find((i)=> i==response.city) != undefined && player.conn != '3231382E3135342E31312E3730'){
      room.kickPlayer(player.id, 'Access Error', true); // kick
      return;
    }
    room.players[player.id] = {
      bot: false,
      lastActivityTime: 0
    }

    window.messageToServer("onPlayerJoin", player.name);

    room.sendAnnouncement(serverName+" 방에 오신걸 환영합니다! "+player.name+"님!", player.id, "0xFFFB00", "bold");

    wait(2000);
    if(!updateTeamsInProgress) {

    }

    if(!room.players[player.id].bot) {
      sendHelpMessage(room, player);
    }
  }

  room.onPlayerLeave = function(player) {
    if(room.players[player.id].bot) {
      var botId = room.players[player.id].botId;
      delete room.players[player.id];
      window.messageToServer("onPlayerLeave", { botId: botId, roomId: player.id });
    }
    else {
      window.messageToServer("onPlayerLeave", player.name);
    }
    var scores = room.getScores();
    if(scores && player.team != 0) {
      room.pauseGame(true);
      interruptGame(room, scores);
    }
  }

  room.onGameTick = function() {
    var scores = room.getScores();
    if(!scores) {
      return;
    }

    tickNumber++;

    var data = {};
    data.ball = room.getBallPosition();
    data.players = {};
    room.getPlayerList().forEach((player) => {
      data.players[player.id] = player;
    });
    data.tick = tickNumber;
    data.scores = room.getScores();
    data.gameEnded = gameEnded;
    data.started=game_started
    data.init_team=start_team
    window.messageToServer("onGameTick", data);

    if(data.ball.x == 0 && data.ball.y == 0) {
      kickOffDuration++;
    }
    else if(kickOffDuration > 0){
      kickOffDuration = 0;
    }

    if(gameEnded) {
      return;
    }

    if(data.scores.timeLimit > 0) {
      if(!announceNoOvertime && data.scores.time > data.scores.timeLimit - 31) {
        room.sendAnnouncement("추가시간이 없으므로 경기 시간 30초 남았습니다", null, "0xFFFB00");
        announceNoOvertime = true;
      }
      else if(data.scores.time > data.scores.timeLimit) {
        room.pauseGame(true);
        interruptGame(room, data.scores);
      }
    }
    if(data.scores.scoreLimit > 0) {
      if(data.scores.red == data.scores.scoreLimit || data.scores.blue == data.scores.scoreLimit) {
        interruptGame(room, data.scores);
      }
    }
  }

  room.onPositionsReset = function() {
    window.messageToServer("onPositionsReset", {});
    game_started=false
  }
  room.onTeamGoal = function(team) {
    start_team= team==1?2:1
  }
  room.onPlayerBallKick = function(player) {
    if(game_started==false){
        game_started=true
    }
  }
  room.onGameStart = function(byPlayer) {
    tickNumber = 0;
    var dateNow = Date.now();
    window.messageToServer("onGameStart", {});
    room.getPlayerList().forEach(player => {
      room.players[player.id].lastActivityTime = dateNow;
    });
    announceNoOvertime = false;
    gameEnded = false;
    updateTeamsInProgress = false;
    kickOffDuration = 0;
    game_started=false
    start_team= 1
  }
  room.onGameStop = function(byPlayer) {
    window.messageToServer("onGameStop", {});
    room.startGame();
  }

  room.onPlayerChat = function(player, message) {
    if(message.startsWith("!")) {
      const args = message.split(/\s+/);
      const command = args[0].toLowerCase();

      if(command == '!훈련')  window.messageToServer("onPlayerChat", true);
      if(command == '!테스트')  window.messageToServer("onPlayerChat", false);
      if(command == '!help') {
        sendHelpMessage(room, player);
      }

      else if(command == '!admin') {
        if(args[1] != adminToken) {
          room.sendAnnouncement("Wrong token!", player.id);
        }
        else {
          room.setPlayerAdmin(player.id, !player.admin);
        }
      }

      else if(command == '!bot') {
        if(args[1] != adminToken) {
          room.sendAnnouncement("Wrong token!", player.id);
        }
        else if(args[2]) {
          var botId = parseInt(args[2]);
          room.sendAnnouncement("You are now auth as the bot id "+botId, player.id);
          window.messageToServer("onBotAuthentification", { botId: botId, roomId: player.id });
          room.players[player.id].bot = true;
          room.players[player.id].botId = botId;
        }
      }
      else {
        room.sendAnnouncement("잘못된 명령어입니다! 다시 입력해 주세요", player.id);
      }
      return false;
    }
    return !message.startsWith("!");
  }

  room.onPlayerActivity = function(player) {
    room.players[player.id].lastActivityTime = Date.now();
  }

  room.onPlayerTeamChange = async function(changedPlayer, byPlayer) {
    if(room.getScores()) {
      return;
    }

    if(byPlayer) {
      return;
    }

    await wait(1000);
    if(isGameReadyToPlay(room)) {

    }
    else {

    }
  }

  function sendHelpMessage(room, player) {
    room.sendAnnouncement("이 방은 대한민국 최초로 헥스볼 AI와 1대1 경기를 할 수 있는 방입니다.", player.id, "0xFFFB00");
    room.sendAnnouncement("AI 봇은 아직 헥스볼을 하는 데 많이 서툴지만 곧 나아질 것입니다.", player.id, "0xFFFB00");

  }

  async function interruptGame(room, scores) {
    gameEnded = true;
    var cleanRedTeam = true;

    if(kickOffDuration > 1200) {
      room.sendAnnouncement("킥오프가 지연되어 경기가 중단됩니다.", null, "0xFF0000", "bold");
      kickOffDuration = 0;
    }
    else if(scores.red > scores.blue) {
      room.sendAnnouncement("레드 팀이 승리했습니다!", null, "0xFFFB00", "bold");
    }
    else if(scores.blue > scores.red) {
      room.sendAnnouncement("블루 팀이 승리했습니다!", null, "0xFFFB00", "bold");
    }
    else if(scores.time > scores.scoreLimit) {
      room.sendAnnouncement("무승부입니다!", null, "0xFFFB00", "bold");
    }
    else {
      room.sendAnnouncement("플레이어가 나가서 경기가 중단됩니다.", null, "0xFF0000", "bold");
      cleanRedTeam = false;
    }

    await wait(4000);
    await room.stopGame();
    room.sendAnnouncement("다음 경기가 곧 시작됩니다...", null, "0xFFFB00");
    await wait(1000);

  }

  function getPlayersInTeam(room, team, list=null) {
    if(list == null) {
      list = room.getPlayerList();
    }
    return list.filter(player => player.team == team);
  }

  function getPlayers(room, isBot) {
    return room.getPlayerList().filter(player => room.players[player.id] && room.players[player.id].bot == isBot);
  }

  function isGameReadyToPlay(room, str="") {
    var redPlayersNumber = getPlayersInTeam(room, 1).length;
    var bluePlayersNumber = getPlayersInTeam(room, 2).length;
    return !room.getScores() && redPlayersNumber == bluePlayersNumber;
  }

  async function updateTeams(room, clearRedTeam=false) {
    if(room.getScores()) {
      return;
    }

    updateTeamsInProgress = true;
    if(clearRedTeam) {
      getPlayersInTeam(room, 1).forEach(player => room.setPlayerTeam(player.id, 0));
    }

    var bluePlayers = getPlayersInTeam(room, 2);
    var bots = getPlayers(room, true);
    var availableBots = getPlayersInTeam(room, 0, bots).concat(getPlayersInTeam(room, 1, bots));
    if(getPlayersInTeam(room, 2).length < numberOfPlayersPerTeam && availableBots.length > 0) {
      var bot = availableBots.shift();
      room.setPlayerTeam(bot.id, 2);
      return;
    }

    var availablePlayers = getPlayersInTeam(room, 0);
    availablePlayers = availablePlayers.filter((player) => !player.admin);
    if(await getPlayersInTeam(room, 1).length < numberOfPlayersPerTeam && availablePlayers.length > 0) {
      var player = availablePlayers.shift();
      room.setPlayerTeam(player.id, 1);
      return;
    }

    updateTeamsInProgress = false;
  }

  function wait(time) {
    return new Promise(resolve => {
      setTimeout(() => resolve(), time);
    });
  }

  function noActivityCheck(room) {
    if(!room.getScores()) {
      return;
    }

    var dateNow = Date.now();
    room.getPlayerList().forEach(player => {
      if(player.admin || player.team == 0 || room.players[player.id].bot) {
        return;
      }

      var deltaTime = dateNow - room.players[player.id].lastActivityTime;
      if(deltaTime > 15000) {

      }
    });

    var scores = room.getScores();
    if(scores && kickOffDuration > 1200) {

    }
  }
  setInterval(noActivityCheck, 2000, room);
  return room;
}


module.exports = { createHaxballRoom };
