# save as convert_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Copy your model definition here
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

# Add the class to main module
sys.modules['__main__'].ResNet9 = ResNet9

# Try various ways to open the model
def convert_model(input_path, output_path, num_classes):
    print(f"Creating fresh model instance...")
    fresh_model = ResNet9(3, num_classes)
    
    # Try to load with weights_only=False (unsafe but necessary)
    try:
        print(f"Attempting to load model from {input_path}...")
        with torch.serialization.safe_globals(['__main__.ResNet9']):
            loaded_obj = torch.load(input_path, map_location='cpu', weights_only=False)
        
        print(f"Loaded object of type: {type(loaded_obj)}")
        
        # Extract state_dict if needed
        if isinstance(loaded_obj, nn.Module):
            state_dict = loaded_obj.state_dict()
            print("Extracted state_dict from loaded model")
        elif isinstance(loaded_obj, dict):
            state_dict = loaded_obj
            print("Loaded object is already a state_dict")
        else:
            print(f"Unexpected type: {type(loaded_obj)}")
            return
            
        # Save just the state_dict
        print(f"Saving clean state_dict to {output_path}...")
        torch.save(state_dict, output_path)
        print("Successfully converted model!")
        
        # Verify we can load it
        test_model = ResNet9(3, num_classes)
        test_model.load_state_dict(torch.load(output_path))
        print("Verified the new model file loads correctly!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    # Replace with your actual values
    INPUT_PATH = "plant-disease-model-complete.pth"
    OUTPUT_PATH = "plant-disease-model-clean.pth"
    NUM_CLASSES = 38  # Adjust based on your actual number of classes
    
    convert_model(INPUT_PATH, OUTPUT_PATH, NUM_CLASSES)