import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os


# check if gpu is available for faster calculations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform the images: crop, flip, stretch ... to get better training
data_transforms = {
'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
]),
}

# classify training images based on folders
data_dir = 'houseplants_data'
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}


# create dataloader to load images with labesls during training
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], 
        batch_size=16,  # how many images train each itearation
        shuffle=True, 
        num_workers=4 # images calssified simutanesly (:
    )
    for x in ['train', 'val']
}

# save the size of each dataset, will be used for computing accuracy
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# classes to classify the images to 
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print("Classes found:", class_names)

# for this well use ResNet50 model pretrained on ImageNet
model = models.resnet50(pretrained=True)

# to train the model to work well with our data while preventing overfitting well train only the last 4 
# layers and the final layer of the neural network explenation from chat:
# In ResNet, layer4 is the final stack of convolutional layers before the classification head. 
# Fine-tuning this portion plus the final layer often yields a good balance between efficient training time and leveraging pretrained features.


# freeze all layers of the neural network to prevent overfitting of the model to our training images
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last block (layer4) + final FC layer
for name, param in model.named_parameters():
    # Layer4 is the last main block in ResNet50
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

# replace the final classification layer to be trained from scratch to classify our classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# move model to the GPU if available for faster calculations
model = model.to(device)



# the standard loss function for multi-class classification tasks. 
# it measures the difference between the model’s predicted probability distribution and the true distribution 
criterion = nn.CrossEntropyLoss()

# we dont want to train the whole network so well only train layers where: `requires_grad=True` (unfreezed layers)
# this optimizer use gradient of the loss function in respect to the weights, to find the direction of the error 
# and how much to adjust each layer connection weight of the model, to minimize that loss.
# update weight by subtracting the gradient of the loss function * lr
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

# after step_size number of ephocs, reduce learning rate (lr) by *gamma, allows subtler changes as the learning advances to help the network converge more smoothly 
# is it really needed ????????????????????????
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode
            
            # reset statistics
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # resets the gradients from the previous batches
                optimizer.zero_grad()
                
                # forward pass on layers, calculate gradient only if in the train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # get p for each class
                    _, preds = torch.max(outputs, 1) # the prediction is the class with the highest p
                    loss = criterion(outputs, labels) # calculate loss function using classifications 

                    # backprop (update gradient) only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # track the loss  add accuracy for statistics and tracking
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # calculate average loss and accuracy over the epoch for statistics and tracking 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # step the scheduler once per epoch in the training phase (when reach step size, lower the learning rate)
            if phase == 'train':
                scheduler.step()
            
        print()
    return model

# train and save the model
num_epochs = 10  # higher = more accurate?
print("Starting training...")
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

model_save_path = 'houseplant_resnet50_finetuned.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'.")
