const express = require("express");
const cors = require("cors");
const ort = require("onnxruntime-node"); // ONNX Runtime for Node.js
const sharp = require("sharp"); // Image processing
const { Buffer } = require("buffer");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" })); // Allow large JSON payloads


async function startserver(){

// 🔹 Load ONNX model
const MODEL_PATH = path.join("plant-disease-model-complete.onnx");
let session;

async function loadModel() {
    session = await ort.InferenceSession.create(MODEL_PATH);
    console.log("✅ ONNX model loaded successfully!");
}

loadModel();

// 🔹 Class names 
const CLASS_NAMES = [
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Potato___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Tomato___Early_blight",
    "Tomato___Septoria_leaf_spot",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Strawberry___Leaf_scorch",
    "Peach___healthy",
    "Apple___Apple_scab",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Apple___Black_rot",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Apple___Cedar_apple_rust",
    "Tomato___Target_Spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Strawberry___healthy",
    "Apple___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Esca_(Black_Measles)",
    "Raspberry___healthy",
    "Tomato___Leaf_Mold",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Pepper,_bell___Bacterial_spot",
    "Corn_(maize)___healthy"
];

app.get("/", (req, res) => {
    res.send("✅ Plant Disease Prediction API");
});

// 🔹 Handle Image Prediction
app.post("/predict", async (req, res) => {
    try {
        const { image } = req.body;

        // 🔹 Convert Base64 to Buffer and Resize Image
        const imageBuffer = Buffer.from(image.split(",")[1], "base64");
        const resizedImage = await sharp(imageBuffer)
            .resize(256, 256)
            .toColourspace("srgb")
            .raw()
            .toBuffer();

        // 🔹 Convert to Float32 Array for ONNX
        const inputTensor = new Float32Array(resizedImage.length);
        for (let i = 0; i < resizedImage.length; i++) {
            inputTensor[i] = resizedImage[i] / 255.0; // Normalize pixel values
        }

        // 🔹 Run ONNX inference
        const tensor = new ort.Tensor("float32", inputTensor, [1, 3, 256, 256]);
        const results = await session.run({ input: tensor });
        const output = results.output.data; // Get output probabilities

        // 🔹 Find the class with the highest probability
        const predictedIndex = output.indexOf(Math.max(...output));
        const predictedLabel = CLASS_NAMES[predictedIndex];

        res.json({ prediction: predictedLabel });

    } catch (error) {
        console.error("Error:", error);
        res.status(500).json({ error: "Prediction failed" });
    }
});

// 🔹 Start Server
app.listen(3001, () => console.log("✅ Express server running on port 3001"));

}

startserver();
