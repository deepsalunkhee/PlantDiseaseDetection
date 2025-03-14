import React, { useState } from "react";
import axios from "axios";
import "./App.css"; 

const diseaseData = {
    "Tomato": ["Late Blight", "Healthy", "Early Blight", "Septoria Leaf Spot", "Yellow Leaf Curl Virus", "Bacterial Spot", "Target Spot", "Tomato Mosaic Virus", "Leaf Mold", "Spider Mites"],
    "Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
    "Corn": ["Northern Leaf Blight", "Cercospora Leaf Spot (Gray Leaf Spot)", "Common Rust", "Healthy"],
    "Grape": ["Healthy", "Leaf Blight (Isariopsis Leaf Spot)", "Black Rot", "Esca (Black Measles)"],
    "Strawberry": ["Leaf Scorch", "Healthy"],
    "Potato": ["Healthy", "Early Blight", "Late Blight"],
    "Peach": ["Healthy", "Bacterial Spot"],
    "Cherry": ["Healthy", "Powdery Mildew"],
    "Orange": ["Haunglongbing (Citrus Greening)"],
    "Blueberry": ["Healthy"],
    "Soybean": ["Healthy"],
    "Squash": ["Powdery Mildew"],
    "Raspberry": ["Healthy"],
    "Pepper Bell": ["Healthy", "Bacterial Spot"]
};

function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [prediction, setPrediction] = useState("");
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setPrediction("");
        const imageFile = event.target.files[0];
        if (imageFile) {
            const reader = new FileReader();
            reader.readAsDataURL(imageFile);
            reader.onloadend = () => {
                setFile(reader.result);
                setPreview(reader.result);
            };
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) return;

        setLoading(true); // Start loader
        try {
            const response = await axios.post("https://plantdiseasedetection-m3ar.onrender.com/predict", { image: file });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error("Error:", error);
        }
        setLoading(false); // Stop loader
    };

    return (
        <div className="container">
            <div className="left-section">
                <h1 className="heading">Plant Disease Detection</h1>
                <h2>🌿 Upload Plant Image</h2>
                <form onSubmit={handleSubmit}>
                    <div className="image-placeholder">
                        {preview ? <img src={preview} alt="Uploaded Preview" className="preview-img" /> : "Image Preview"}
                    </div>
                    <input type="file" accept="image/*" onChange={handleFileChange} required />
                    <button type="submit" disabled={loading}>
                        {loading ? "Predicting..." : "Predict"}
                    </button>
                </form>
                {loading && <p>🔄 Processing...</p>}
                {prediction && <h2 className="prediction">Prediction: {prediction}</h2>}
            </div>

            <div className="right-section">
                <h2>📝 Predictable Diseases</h2>
                <div className="scrollable-tables">
                    {Object.entries(diseaseData).map(([plant, diseases]) => (
                        <div key={plant} className="plant-table">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Sr.</th>
                                        <th>{plant}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {diseases.map((disease, index) => (
                                        <tr key={index}>
                                            <td>{index + 1}</td>
                                            <td>{disease}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default App;
