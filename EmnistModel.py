"""
Hier wird das Model für Handschriftserkennung erzeugt.
By Armel Franck Djiongo
"""

# import pandas as pd
import torch
import torchvision
from time import time
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np

''' df = pd.read_csv("emnist.csv")
print(df.shape)
# print(df.describe())
print(df.values)'''

# Herunterladen und Aufbereiten der Daten
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), ]) # Die Daten zu tensor transformieren

# Training und Test Loaders wird erzeugt
train_set = datasets.EMNIST('Data2FürPytorch', split="balanced", download=True, train=True, transform=transform)
test_set = datasets.EMNIST('Data2FürPytorch', split="balanced", download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


# Model erstellen
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Definieren Sie die Schichten und ihre Parameter
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.conv3 = nn.Conv2d(50, 64, 3, 1)
        self.linear1 = nn.Linear(3 * 3 * 64, 128)
        self.linear2 = nn.Linear(128, 47)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        # die hidden Schichten definieren
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2, 2)
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(self.conv3(out))
        out = self.dropout1(out)
        out = out.view(-1, 3 * 3 * 64)
        out = F.relu(self.linear1(out))
        out = self.dropout2(out)
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)  # Output Layer


# Definition der Trainingsfunktion
def train(model, device, train_data, optimizer, epoch):
    model.train()  # training aktivieren
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # Kontrolle des Fortschritts und des Verlusts
        if batch_idx % 400 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# Gerätekonfiguration
device = torch.device("cpu")
model = MyModel()
model.to(device)

# Hyper Parameters
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
n_epoch = 15


# Definition der Testingsfunktion
def test(model, device, test_data):
    model.eval() # testing aktivieren
    # Variablen, die zur Speicherung der Informationen über die Genauigkeit verwendet werden
    test_loss = 0  # loss
    correct_1 = 0  # top 1 richtige number
    correct_5 = 0  # top 5 richtige number
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss/fehler
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # top 1 accuracy
            pred_1 = output.argmax(dim=1, keepdim=True)
            correct_1 += pred_1.eq(target.view_as(pred_1)).sum().item()
            # top 5 accuracy
            _, pred_5 = output.topk(5, dim=1)
            pred_5 = pred_5.t()
            correct = pred_5.eq(target.view(1, -1).expand_as(pred_5))
            correct_5 += correct[:5].view(-1).float().sum(0)
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Top1 accuracy: {}/{} ({:.0f}%), '
          'Top5 accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_1, len(test_loader.dataset),
        100. * correct_1 / len(test_loader.dataset),
        correct_5, len(test_loader.dataset),
        100. * correct_5 / len(test_loader.dataset)))


# Trainings- und Testschleife für die gesamten Daten
'''for epoch in range(n_epoch):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# model save
path = 'EMNISTModel.pt'
torch.save(model.state_dict(), path)'''


# Prädiktionstest, basierend auf den ersten zufällig 15 Tensoren im Test_Loader
# Das Modell sollte zuerst trainiert und gespeichert werden, bevor diese Methode angewendet wird.
def predict():
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    model.load_state_dict(torch.load("EMNISTModel.pt"))
    output = model(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    print("Actuel : ", labels[:15])
    print("erkannt : erkannt", preds[:15])

# Methode laufen lassen
print(predict())