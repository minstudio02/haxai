const vec = require('../src/vectors.js');

function action(env) {
  let distanceWithBall = vec.normL2(env.ball.position);
  let distanceWithOpt = vec.normL2(env.opponents[0].position);
  let optdistanceWithBall = vec.normL2(vec.sub(env.ball.position,env.opponents[0].position));
  if(distanceWithBall < 21.41) {
    let clearPath= (env.bot.team==1?env.opponents[0].position.x<0:env.opponents[0].position.x>0)
    if(clearPath){
      let onOpponentSide= (env.bot.team==1?env.bot.position.x>0:env.bot.position.x<0)
      if(onOpponentSide){
        let goalAngle= (env.bot.team==1?
                        (vec.angle(vec.sub({x:320,y:-55},vec.add(env.bot.position,env.ball.position)))>vec.angle(env.ball.position)&&vec.angle(vec.sub({x:320,y:55},vec.add(env.bot.position,env.ball.position)))<vec.angle(env.ball.position)):
                        (vec.angle(vec.sub({x:-320,y:-55},vec.add(env.bot.position,env.ball.position)))<vec.angle(env.ball.position)||vec.angle(vec.sub({x:-320,y:55},vec.add(env.bot.position,env.ball.position)))>vec.angle(env.ball.position)))
        if(goalAngle){
          return "kick";
        }else{
          if(env.ball.position.y>0){
            return "right"
          }else{
            return "left"
          }
        }
      }else{
        return move()
      }
    }else{
      if(distanceWithOpt<100){
        return move()
      }else{
        return move()
      }
    }
  }else{
    if(optdistanceWithBall < 21.41){
      let onOurSide= (env.bot.team==1?env.opponents[0].position.x<0:env.opponents[0].position.x>0)
      if(onOurSide){
        return move()
      }else{
        return move()
      }
    }else{
      if(distanceWithBall<optdistanceWithBall){
        if(distanceWithOpt<200){
          return move()
        }else{
          return move()
        }
      }else{
        return move()
      }
    }
  }
  function move() {
    var angle = vec.angle(env.ball.position);

    if(approxEquals(angle, 0)) {
      return "forward";
    }
    else if(approxEquals(angle, 45)) {
      return "forward-left";
    }
    else if(approxEquals(angle, 90)) {
      return "left";
    }
    else if(approxEquals(angle, 135)) {
      return "backward-left";
    }
    else if(approxEquals(Math.abs(angle), 180)) {
      return "backward";
    }
    else if(approxEquals(angle, -135)) {
      return "backward-right";
    }
    else if(approxEquals(angle, -90)) {
      return "right";
    }
    else /*if(approxEquals(angle, -45))*/ {
      return "forward-right";
    }
  }
}

function approxEquals(v1, v2) {
  return Math.abs(v1-v2) <= 22.5;
}

module.exports = { action };
