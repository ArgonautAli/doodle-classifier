const modelUrl =
  "https://doodle-classifier-mini.s3.ap-northeast-1.amazonaws.com/model/";

const classLabels = ["ambulance", "cat", "dog", "house", "tree"];

const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

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

async function runModel() {
  const model = await tf.loadLayersModel(modelUrl + "model.json");

  model.summary();
}

async function predict_doodle(reshaped_data) {
  const model = await tf.loadLayersModel(modelUrl + "model.json");
  const input_data = tf.tensor4d(reshaped_data, [1, 28, 28, 1]);
  const prediction = model.predict(input_data);
  const predictionValues = prediction.dataSync();
  // Find the predicted class and its confidence
  let predictedClass = -1;
  let maxConfidence = -1;
  for (let i = 0; i < predictionValues.length; i++) {
    if (predictionValues[i] > maxConfidence) {
      maxConfidence = predictionValues[i];
      predictedClass = i;
    }
  }
  const predictedClassIndex = prediction.argMax(-1).dataSync()[0];
  const predictedClassLabel = classLabels[predictedClassIndex];
  const predictedConfidence = predictionValues[predictedClassIndex];

  console.log(`Predicted class: ${predictedClassLabel}`);
  console.log(`Confidence: ${predictedConfidence}`);

  // Get top 5 predictions
  const top5 = Array.from(predictionValues)
    .map((confidence, index) => ({ confidence, label: classLabels[index] }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  console.log("Top 5 predictions:", top5);
}

const getPixel = document.getElementById("predict");

function processImage(imageData) {
  const imageWidth = 28;
  const imageHeight = 28;

  const temp_cvs = document.createElement("canvas");
  temp_cvs.width = imageWidth;
  temp_cvs.height = imageHeight;
  const temp_ctx = temp_cvs.getContext("2d");

  temp_ctx.putImageData(imageData, 0, 0);

  temp_ctx.drawImage(
    temp_cvs,
    0,
    0,
    imageData.width,
    imageData.height,
    0,
    0,
    imageWidth,
    imageHeight
  );

  const resized_image_data = temp_ctx.getImageData(
    0,
    0,
    imageWidth,
    imageHeight
  );

  for (let i = 0; i < resized_image_data.data.length; i += 4) {
    resized_image_data.data[i] = 255 - resized_image_data.data[i];
    resized_image_data.data[i + 1] = 255 - resized_image_data.data[i + 1];
    resized_image_data.data[i + 2] = 255 - resized_image_data.data[i + 2];
    resized_image_data.data[i + 3] = 255;
  }
  const resized_data = resized_image_data.data;

  // convert to greyscale & normalize
  const normalized_data = new Float32Array(imageWidth * imageHeight);
  for (let i = 0; i < resized_data.length; i++) {
    const greyscale =
      (resized_data[i] + resized_data[i + 1] + resized_data[i + 2]) / 3;
    normalized_data[i / 4] = greyscale / 255.0;
  }

  // reshaping as model's input
  const reshaped_data = new Float32Array(1 * imageWidth * imageHeight * 1);
  for (let i = 0; i < reshaped_data.length; i++) {
    reshaped_data[i] = normalized_data[i];
  }

  return reshaped_data;
}

getPixel.onclick = function () {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixelData = imageData.data;
  const reshaped_data = processImage(imageData);
  predict_doodle(reshaped_data);
};

runModel();

// const canvas = document.getElementById("myCanvas");
// const ctx = canvas.getContext("2d");
