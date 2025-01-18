import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Device configuration ??????????????????
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# number of our classes (plants in our db)
num_classes = 1
class_names = ['plant1 name'] # class names in the same order as training  

# build the same ResNet50 architecture used in training
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# load our trained model
model.load_state_dict(torch.load('houseplant_resnet50_finetuned.pth', map_location=device))
model = model.to(device)
model.eval() # always in evalutation mode during inference

# transform images to fit the model format
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])

# classify image
def predict_image(model, image_path, transform):
    
    # prepare image format and attributes to the model
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(): # prevent grad calculation to save memorand time
        outputs = model(img_t)
        _, pred_idx = torch.max(outputs, 1) # infere the higiest propebillity class
    return pred_idx.item(), outputs

# test images from folder (change to your folder or single image)
test_folder = 'test_images'
for filename in os.listdir(test_folder):
    full_path = os.path.join(test_folder, filename) # get file
    # classify image and print class
    pred_idx, outputs = predict_image(model, full_path, inference_transform) 
    pred_class = class_names[pred_idx]
    print(f"Image: {filename} => Predicted: {pred_class}")
    print()

