
async function loadModel() {
    try {
        const modelUrl =
        "https://doodle-classifier-mini.s3.ap-northeast-1.amazonaws.com/model/";   

        const model = await tf.loadLayersModel("./model/model.json");

        console.log("model", model)

    } catch (error) {
        console.error("Failed to load the model:", error);
        throw error;
    }
}

async function initializeModel() {
    try {
        const model = await loadModel();
        // Now you can use the model for predictions
        console.log("Model is ready for predictions");
        // You might want to store the model in a global variable or state for later use
        window.doodleModel = model;
    } catch (error) {
        console.error("Failed to initialize the model:", error);
    }
}

// Call initializeModel when the page loads or as needed
initializeModel();
