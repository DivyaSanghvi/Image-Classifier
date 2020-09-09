import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse


def data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train' 
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms )
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =64,shuffle = True)
    return trainloader , validationloader, testloader,train_data

def setup(structure='densenet121',dropout=0.5,hidden_layer1=200,lr=0.001,power='gpu'):
    arch={'vgg16':25088,'densenet121':1024}
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16 or densenet121?".format(structure))
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        model.classifier = nn.Sequential(OrderedDict([('dropout', nn.Dropout(dropout)),
                                                  ('fc1', nn.Linear(arch[structure], hidden_layer1)),
                                                  ('relu', nn.ReLU()),
                                                  ('dropout', nn.Dropout(dropout)),
                                                  ('fc2', nn.Linear(hidden_layer1, 100)),
                                                  ('relu', nn.ReLU()),
                                                  ('fc3', nn.Linear(100, 102)),
                                                  ('output', nn.LogSoftmax(dim=1))]))

        criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        model.to('cuda')
        return model,optimizer,criterion
def train_network(model, criterion, optimizer,trainloader,validationloader, epochs = 3, print_every=20, power='gpu'):
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for ii,(inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                validation_loss=0
                accuracy = 0
                for ii, (inputs1,labels1) in enumerate(validationloader):
 
        
                    inputs1, labels1 = inputs1.to('cuda:0') , labels1.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():
                        logps = model.forward(inputs1)
                        validation_loss = criterion(logps, labels1)
                        ps = torch.exp(logps).data
                        equals = (labels1.data == ps.max(1)[1])
                        accuracy += equals.type_as(torch.FloatTensor()).mean()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(validationloader):.3f}")
                running_loss = 0
                
def save_checkpoint(train_data,epochs,structure,hidden_layer1,dropout,lr,path='checkpoint.pth'):
    model,_,_=setup(structure,dropout,hidden_layer1,lr)
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure' :structure,'hidden_layer1':hidden_layer1, 'dropout':dropout, 'lr':lr, 'nb_of_epochs':epochs, 'state_dict':model.state_dict(), 'class_to_idx':model.class_to_idx},path)
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model,_,_ = setup(structure , dropout,hidden_layer1,lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
def process_image(image):


    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=5,power='gpu'):   
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)