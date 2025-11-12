function add(vec1, vec2) {
  return {
    x: vec1.x + vec2.x,
    y: vec1.y + vec2.y
   }
}

function sub(vec1, vec2) {
  return {
    x: vec1.x - vec2.x,
    y: vec1.y - vec2.y
   }
}

function div(vec, denominator) {
  return {
    x: vec.x / denominator,
    y: vec.y / denominator
   }
}

function mul(vec, factor) {
  return {
    x: vec.x * factor,
    y: vec.y * factor
   }
}

function normL2(vec) {
  return Math.sqrt(Math.pow(vec.x, 2) + Math.pow(vec.y, 2));
}

function angle(vec1, vec2 = {x: 1, y: 0})  {
    let rad = Math.atan2(vec2.y,vec2.x) - Math.atan2(vec1.y,vec1.x);
    let deg = rad*(180/Math.PI);
    return deg;
}

function shotangle(vec1, vec2)  {
  let rad = (vec1.x >=0) ? Math.atan2(vec2.y,vec2.x) - Math.atan2(vec1.y,vec1.x) : Math.atan2(vec2.y,-vec2.x) - Math.atan2(vec1.y,-vec1.x)
  let deg = Math.abs(rad*(180/Math.PI)) > 180 ? 360-Math.abs(rad*(180/Math.PI)):Math.abs(rad*(180/Math.PI));
  return deg;
}

function transformVectors(object, operation) {
  if(typeof object === 'object') {
    if("x" in object && "y" in object && Object.keys(object).length == 2) {
      return operation(object);
    }
    else {
      let newObject = {};
      Object.keys(object).forEach((key) => {
        newObject[key] = transformVectors(object[key], operation);
      });
      return newObject;
    }
  }
  else if(Array.isArray(object)) {
    return object.map((arrayObject) => {
      arrayObject = transformVectors(arrayObject, operation);
    });
  }
  else {
    return object;
  }
}

module.exports = { add, sub, div, mul, normL2, angle, transformVectors, shotangle };
