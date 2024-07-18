// Sample input data for a grayscale image of size 28x28 pixels
const inputDataArray = new Float32Array(784);
for (let i = 0; i < 784; i++) {
  inputDataArray[i] = Math.random(); // Use random values between 0 and 1 for testing
}

// Convert the array to a 4D tensor
const inputData = tf.tensor4d(inputDataArray, [1, 28, 28, 1]);
console.log("inputData", inputData);

async function runModel() {
  const modelUrl =
    "https://doodle-classifier-mini.s3.ap-northeast-1.amazonaws.com/model/";
  const model = await tf.loadLayersModel(modelUrl + "model.json");

  // Log model summary for debugging
  model.summary();
  console.log("  model.summary()",  model.summary())

  // Perform prediction
  const output = model.predict(inputData);

  // Log the output
  output.print();
}

// Run the function
runModel();

// Changed batch_shape to input_shape

// For TensorFlow.js 4.20.0, which was released in early 2024, it would be compatible with TensorFlow (Python) versions that were current around that period, likely in the range of TensorFlow 2.11.x to TensorFlow 2.12.x.
