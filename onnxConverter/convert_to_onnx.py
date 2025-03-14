import torch
import torchvision.transforms as transforms
import onnx
import torch.nn as nn

# ✅ Define the base class for classification
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        epoch_accuracy = torch.stack(batch_accuracy).mean()  # Combine accuracies
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, "
              f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_accuracy']:.4f}")

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# ✅ Define ConvBlock function
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ✅ Define ResNet9 class
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
        
    def forward(self, xb):  # xb is the input batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out  # Residual connection
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out  # Residual connection
        out = self.classifier(out)
        return out

# ✅ Load your trained PyTorch model
MODEL_PATH = r"D:\Personal-Projects-2.0\PlantDiseasePrediction\models\plant-disease-model-complete.pth"
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"),weights_only=False)
model.eval()

# ✅ Convert to ONNX format
dummy_input = torch.randn(1, 3, 256, 256)  # Simulated image input
onnx_path = r"D:\Personal-Projects-2.0\PlantDiseasePrediction\models\plant-disease-model-complete.onnx"

torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"])

print(f"✅ Model converted and saved as {onnx_path}")
