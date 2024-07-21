const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

// canvas.width = window.innerWidth - canvasOffsetX;
// canvas.height = window.innerHeight - canvasOffsetY;

let isPainting = false;
let lineWidth = 5;
let startX;
let startY;

const clearButton = document.getElementById("clear");
clearButton.onclick = function () {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
};

canvas.addEventListener("mousedown", (e) => {
  isPainting = true;
  startX = e.clientX - canvasOffsetX;
  startY = e.clientY - canvasOffsetY;
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
});

canvas.addEventListener("mouseup", () => {
  isPainting = false;
  ctx.stroke();
  ctx.beginPath();
});

canvas.addEventListener("mousemove", (e) => {
  if (!isPainting) return;
  ctx.lineTo(e.clientX - canvasOffsetX, e.clientY - canvasOffsetY);
  ctx.stroke();
});

// Convert the array to a 4D tensor
// const inputData = tf.tensor4d(inputDataArray, [1, 28, 28, 1]);

const getPixel = document.getElementById("predict");
console.log("getPixel", getPixel);

getPixel.onclick = function () {
  console.log("clicked");
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  console.log(imageData);
  // Now you can access the pixel data through imageData.data
  const pixelData = imageData.data;
  console.log(pixelData);
};

async function runModel() {
  const modelUrl =
    "https://doodle-classifier-mini.s3.ap-northeast-1.amazonaws.com/model/";
  const model = await tf.loadLayersModel(modelUrl + "model.json");

  model.summary();
  console.log("  model.summary()", model);
}

runModel();

// const canvas = document.getElementById("myCanvas");
// const ctx = canvas.getContext("2d");
