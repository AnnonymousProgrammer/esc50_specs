import torch
from torch.optim import optimizer
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import librosa
import librosa.display
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ESC50Spectrogram(Dataset):

  def __init__(self, path):
    files = Path(path).glob("*.wav")
    self.items = [(str(f), f.name.split("-")[-1].replace(".wav", "")) for f in files]
    self.length = len(self.items)
    self.img_transforms = transforms.Compose(transforms.ToTensor())

  def __getitem__(self, index):
    filename, label = self.items(index)
    audio_tensor, sample_rate = librosa.load(filename, sr=None)
    spectrogram = librosa.feature.melspectrogram(audio_tensor, sr=sample_rate)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis="time", y_axis="mel")
    fig = plt.gcf()
    fig.canvas.draw()
    audio_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.unint8)
    audio_data = audio_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return (self.img_transforms(audio_data), label)
  
  def __len__(self):
    return self.length

def move_shite():
    base = "ESC-50/audio"
    os.system("mkdir train")
    os.system("mkdir test")
    os.system("mkdir valid")
    os.system("mv " + base + "/1* ./train")
    os.system("mv " + base + "/2* ./train")
    os.system("mv " + base + "/3* ./train")
    os.system("mv " + base + "/4* ./valid")
    os.system("mv " + base + "/5* ./test")

def precompute_spectrograms(path, dpi=50):
  print("start conversion")
  os.mkdir(path + "_image")
  files = Path(path).glob("*.wav")
  file_count = len(os.listdir(path))
  counter = 0
  print("created dir")
  for filename in files:
    counter +=1
    print("finished : " + str(counter / file_count))
    audio_tensor, sr = librosa.load(filename, sr=None)
    spectrogram = librosa.feature.melspectrogram(audio_tensor, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
    fig = plt.gcf().savefig("{}{}_{}.png".format(str(path) + "_image" + os.sep + str(filename.parent), dpi, filename.name), dpi=dpi)

'''
class PrecomputedESC50(Dataset):
    def __init__(self,path,dpi=50, img_transforms=None):
        files = Path(path).glob('{}*.wav.png'.format(dpi))
        self.items = [(f,int(f.name.split("-")[-1].replace(".wav.png",""))) for f in files]
        self.length = len(self.items)
        print(self.length)
        if img_transforms == None:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = img_transforms
    
    def __getitem__(self, index):
        filename, label = self.items[index]
        img = Image.open(filename)
        return (self.transforms(img), label)
            
    def __len__(self):
        return self.length
'''

class PrecomputedESC50(Dataset):

  def __init__(self, path, dpi=50, img_transforms=None):
    #posix_path = Path(mypath)
    #files = posix_path.glob("{}{}*.wav.png".format(posix_path.name, dpi))
    files = os.listdir(path)
    self.items = []
    for file_name in files:
      label = file_name.split("-")[-1].replace(".wav.png", "")
      self.items.append((path + os.sep + file_name, int(label)))
    self.length = len(self.items)
    print("found " + str(self.length) + " images")
    if img_transforms is None:
      self.img_transforms = transforms.Compose([transforms.ToTensor()])
    else:
      self.img_transforms = img_transforms
  
  def __getitem__(self, index):
      filename, label = self.items[index]
      img = Image.open(filename).convert("RGB")
      return (self.img_transforms(img), label)

  def __len__(self):
    return self.length

def get_initial_model():
  spec_resnet = models.resnet50(pretrained=True)
  for _, param in spec_resnet.named_parameters():
      param.requires_grad = False
  spec_resnet.fc = nn.Sequential(nn.Linear(spec_resnet.fc.in_features, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500,50))
  return spec_resnet

def get_dataloader(path_to_images):
  dataset = PrecomputedESC50(path_to_images, img_transforms=transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
  return DataLoader(dataset, batch_size=64, shuffle=True)
  

def find_lr(model, loss_fn, optimizer, train_loader, init_value=1e-8, final_value=10.0, device="cpu"):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            if(len(log_lrs) > 20):
                return log_lrs[10:-5], losses[10:-5]
            else:
                return log_lrs, losses

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss.item())
        log_lrs.append((lr))

        # Do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    if(len(log_lrs) > 20):
        print("he")
        return log_lrs[10:-5], losses[10:-5]
    else:
        return log_lrs, losses

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

def detect_lr():
  train_dataloader = get_dataloader("train_image")
  test_image = get_dataloader("test_image")
  spec_resnet = get_initial_model()
  torch.save(spec_resnet.state_dict(), "spec_resnet.pth")
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(spec_resnet.parameters(), lr=10e-4)
  print("search learnrate")
  logs, losses = find_lr(spec_resnet, loss_fn, optimizer, train_dataloader, device="cpu")
  print("finished search")
  print(logs)
  print(losses)
  plt.plot(logs, losses)

def pretraining():
  train_dataloader = get_dataloader("train_image")
  valid_loader = get_dataloader("test_image")
  spec_resnet = get_initial_model()
  spec_resnet.load_state_dict(torch.load("spec_resnet.pth"))
  optimizer = optim.Adam(spec_resnet.parameters(), lr=1e-2)
  train(spec_resnet, optimizer, nn.CrossEntropyLoss(), train_dataloader, valid_loader, epochs=5)
  torch.save(spec_resnet.state_dict(), "spec_resnet_pre.pth")

def long_training():
  print("start acquire data")
  train_dataloader = get_dataloader("train_image")
  print("acquired training data")
  valid_loader = get_dataloader("test_image")
  print("acquired validation data")
  spec_resnet = get_initial_model()
  print("loaded model structure")
  spec_resnet.load_state_dict(torch.load("spec_resnet_pre.pth"))
  print("loaded parameters")
  for param in spec_resnet.parameters():
    param.requires_grad = True
  print("set trainable")
  optimizer = optim.Adam([
                        {'params': spec_resnet.conv1.parameters()},
                        {'params': spec_resnet.bn1.parameters()},
                        {'params': spec_resnet.relu.parameters()},
                        {'params': spec_resnet.maxpool.parameters()},
                        {'params': spec_resnet.layer1.parameters(), 'lr': 1e-4},
                        {'params': spec_resnet.layer2.parameters(), 'lr': 1e-4},
                        {'params': spec_resnet.layer3.parameters(), 'lr': 1e-4},
                        {'params': spec_resnet.layer4.parameters(), 'lr': 1e-4},
                        {'params': spec_resnet.avgpool.parameters(), 'lr': 1e-4},
                        {'params': spec_resnet.fc.parameters(), 'lr': 1e-8}
                        ], lr=1e-2)
  print("created optimizer and start to train")
  train(spec_resnet, optimizer, nn.CrossEntropyLoss(), train_dataloader, valid_loader, epochs=40)

long_training()
