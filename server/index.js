const express = require("express");
const cors = require("cors");
const ort = require("onnxruntime-node"); // ONNX Runtime for Node.js
const sharp = require("sharp"); // Image processing
const { Buffer } = require("buffer");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" })); // Allow large JSON payloads

async function startserver() {
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
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
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