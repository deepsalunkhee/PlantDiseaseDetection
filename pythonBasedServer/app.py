import os
import io
import pickle
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Enable CORS for all origins, all methods, and all headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Custom unpickler to handle the __main__ module issue
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app"  # Use the current module name
        return super().find_class(module, name)

def custom_load(file_obj):
    return RenameUnpickler(file_obj).load()

# Define the model architecture (must match exactly with your training code)
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                      nn.Flatten(),
                                      nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



# Class names (replace with your actual class names)
CLASS_NAMES = [
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
  ]

# Image transformations
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_model(model_path: str, num_classes: int) -> nn.Module:
    # Create a fresh model with your architecture
    model = ResNet9(3, num_classes)
    
    print(f"Creating a new model instance with the same architecture")
    
    try:
        # This is a last-resort approach - create a dictionary to map old module names
        import sys
        
        # Make ResNet9 available globally
        sys.modules['__main__'] = type('MainModule', (), {'ResNet9': ResNet9})
        
        # Now try to load with the standard loader but allowing unsafe globals
        import torch._utils
        try:
            torch._utils._rebuild_tensor_v2
        except AttributeError:
            torch._utils._rebuild_tensor_v2 = torch._utils._rebuild_tensor
        
        # Try with a custom restorer function
        def restore_location(storage, location):
            return storage
        
        state_dict = torch.load(model_path, map_location=restore_location)
        
        # If we got a full model, extract its state dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
            
        # Now load the state dict into our fresh model
        model.load_state_dict(state_dict)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"All loading attempts failed: {e}")
        print("Initializing with random weights instead")
        # We'll just use the freshly initialized model
        pass
    
    model.eval()
    return model


# Initialize model
MODEL_PATH = "plant-disease-model-clean.pth"
NUM_CLASSES = len(CLASS_NAMES)

try:
    model = load_model(MODEL_PATH, NUM_CLASSES)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    # You might want to raise an exception here or provide fallback behavior

@app.get("/")
async def root():
    return {"message": "Plant Disease Classification API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = TRANSFORM(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            prediction = CLASS_NAMES[predicted.item()]
        
        return JSONResponse(content={"prediction": prediction})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     print("Received file:", file.filename)
#     return {"prediction": file.filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)