# Plant Disease Classification using ResNet-9 🌱

This repository contains a deep learning model for classifying plant diseases from leaf images using a custom ResNet-9 architecture implemented in PyTorch. The model can identify 38 different classes comprising healthy and diseased leaves across 14 different plant types.

## Dataset Overview 📊

The model is trained on the PlantVillage dataset, which includes approximately 87,000 RGB images of healthy and diseased crop leaves. Key dataset characteristics:

- **Classes**: 38 different classes (plant + disease combinations)
- **Plants**: 14 unique plant types
- **Diseases**: 26 different plant diseases
- **Image size**: 256x256 RGB images
- **Split**: 80/20 training/validation ratio
- **Data distribution**: Fairly balanced across classes (~1,600-2,000 images per class)

## Model Architecture 🏗️

The implementation uses a custom ResNet-9 architecture with residual connections:

### Key Components

1. **Convolutional Blocks**: Each block consists of:
   - 2D Convolution
   - Batch Normalization
   - ReLU Activation
   - Optional MaxPooling (stride 4)

2. **Residual Blocks**: Two residual blocks that implement skip connections:
   - First residual block after the second conv layer (128 channels)
   - Second residual block after the fourth conv layer (512 channels)

3. **Classifier**: Final classification head consisting of:
   - MaxPooling
   - Flatten
   - Linear layer (512 to 38 classes)

### Network Structure

```
ResNet9(
  (conv1): Sequential(Conv2d, BatchNorm2d, ReLU)
  (conv2): Sequential(Conv2d, BatchNorm2d, ReLU, MaxPool2d) [Output: 128 x 64 x 64]
  (res1): Sequential(
    ConvBlock(128, 128),
    ConvBlock(128, 128)
  )
  (conv3): Sequential(Conv2d, BatchNorm2d, ReLU, MaxPool2d) [Output: 256 x 16 x 16]
  (conv4): Sequential(Conv2d, BatchNorm2d, ReLU, MaxPool2d) [Output: 512 x 4 x 4]
  (res2): Sequential(
    ConvBlock(512, 512),
    ConvBlock(512, 512)
  )
  (classifier): Sequential(MaxPool2d, Flatten, Linear(512, 38))
)
```

### Model Parameters
- Total parameters: 6,589,734
- Trainable parameters: 6,589,734
- Input size: 0.75 MB
- Forward/backward pass size: 343.95 MB
- Parameters size: 25.14 MB

## Training Methodology 🔄

The model was trained using several advanced techniques:

### Optimization Strategy

1. **One Cycle Learning Rate Policy**:
   - Starting with a low learning rate
   - Gradually increasing to a high rate (0.01) for ~30% of epochs
   - Gradually decreasing to a very low value for remaining epochs
   - Helps in faster convergence and better generalization

2. **Weight Decay**: 
   - Value: 1e-4
   - Prevents overfitting by penalizing large weights

3. **Gradient Clipping**:
   - Value: 0.1
   - Prevents exploding gradients by limiting their magnitude

4. **Optimizer**:
   - Adam optimizer with initial max_lr of 0.01

### Training Configuration

- Batch size: 32
- Epochs: 2 (achieved high accuracy very quickly)
- Random seed: 7 (for reproducibility)
- Loss function: Cross-Entropy Loss
- Device: CUDA (GPU acceleration)

## Performance Results 📈

The model achieved impressive results in a short training time:

- Final validation accuracy: **99.23%**
- Final validation loss: **0.0269**
- Training time: ~20 minutes on P100 GPU
- Test accuracy: **100%** (on a small test set of 33 images)

### Learning Curves

The model showed rapid convergence:
- Training loss decreased from 0.7466 to 0.1248
- Validation loss decreased from 0.5865 to 0.0269
- Validation accuracy increased from 83.19% to 99.23%

## Model Usage Guide 🚀

### Prerequisites

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
```

### Loading the Model

```python
# Method 1: Load state_dict (recommended)
model = ResNet9(3, 38)  # Create an instance of the model
model.load_state_dict(torch.load('plant-disease-model.pth'))

# Method 2: Load entire model
model = torch.load('plant-disease-model-complete.pth')

# Set to evaluation mode
model.eval()
```

### Prediction Function

```python
def predict_image(img, model):
    """
    Predicts class for a single image
    
    Args:
        img (torch.Tensor): Image tensor of shape [3, 256, 256]
        model: Trained model
        
    Returns:
        str: Predicted class name
    """
    # Convert to batch of size 1
    xb = img.unsqueeze(0)
    # Get predictions
    yb = model(xb)
    # Get index of highest probability
    _, preds = torch.max(yb, dim=1)
    # Return class name
    return classes[preds[0].item()]
```

### Example Usage

```python
# Load and preprocess image
transform = transforms.ToTensor()
img = Image.open('leaf_image.jpg')
img = transform(img)

# Make prediction
predicted_class = predict_image(img, model)
print(f"Predicted disease: {predicted_class}")
```

## Implementation Details 💻

### Base Class for Training

```python
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
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
        
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
```

### Helper Functions

```python
# Convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Accuracy calculation
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
```

## Advantages of This Architecture 🌟

1. **Efficient Design**: Achieves high accuracy with fewer parameters than standard ResNets
2. **Fast Training**: Converges in just 2 epochs due to effective training strategies
3. **High Accuracy**: 99%+ validation accuracy and perfect test set performance
4. **Residual Connections**: Help in preventing the vanishing gradient problem
5. **Batch Normalization**: Accelerates training and improves stability

## Potential Applications 🌾

- Automated disease diagnosis in agricultural settings
- Mobile applications for farmers to identify plant diseases in the field
- Integration with IoT devices for continuous crop monitoring
- Early warning systems for disease outbreak prevention
- Research tool for plant pathologists

## Future Improvements 🔍

1. **Data Augmentation**: Apply more aggressive augmentation techniques
2. **Transfer Learning**: Compare with pre-trained models like ResNet-50
3. **Model Pruning**: Reduce model size for mobile deployment
4. **Grad-CAM Visualization**: Implement for better interpretability of decisions
5. **Balanced Dataset**: Ensure equal representation across all classes
6. **Deployment**: Create a web or mobile application interface

## Citation ✍️

If you use this model or code, please cite the original PlantVillage dataset:

```
Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.
```

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.