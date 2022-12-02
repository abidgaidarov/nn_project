import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
from torchsummary import summary
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import *

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Нейронная сеть ResNet18

### Классификация изображений котов и собак""")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = ImageFolder('/Users/paantur/Documents/GitHub/nn_project/cats_and_dogs_filtered/train',
                            transform=T.Compose([
                                T.Resize((128, 128)),
                                T.RandomRotation(90),
                                T.ToTensor()
                            ])
                            )

valid_dataset = ImageFolder('/Users/paantur/Documents/GitHub/nn_project/cats_and_dogs_filtered/validation',
                            transform=T.Compose([
                                T.Resize((128, 128)),
                                T.ToTensor()
                            ]))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True
model.fc = nn.Linear(512, 1)

model.to(device)

summary(model, (3, 128, 128), batch_size=64)

#for p in model.parameters():
#    print(p.requires_grad)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()

# зададим функцию рисования графиков
def plot_history(history, grid=True):
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    
    ax[0].plot(history['train_losses'], label='train loss')
    ax[0].plot(history['test_losses'], label='val loss')
    ax[0].set_title(f'Loss on epoch {len(history["train_losses"])}')
    ax[0].grid(grid)
    ax[0].legend()
    
    ax[1].plot(history['train_accs'], label='train acc')
    ax[1].plot(history['test_accs'], label='val acc')
    ax[1].set_title(f'Accuracy on epoch {len(history["train_losses"])}')
    ax[1].grid(grid)
    ax[1].legend()
    
    plt.show()


# Определяем цикл обучения
def torch_train(model, epochs, optimizer, history=None):
    '''
    model: pytorch model - model to train
    epochs: int          - number of epochs
    plot_every: int      - plot every N iterations
    '''
    
    # будем сохранять значения точности и лосса в history
    history = history or {
        'train_accs': [],
        'train_losses': [],
        'test_accs': [],
        'test_losses': [],
    }
    
    # определяем текущую эпоху обучения
    start_epoch = len(history['train_accs'])
    for epoch in range(start_epoch+1, start_epoch+epochs+1):
        print(f'{"-"*13} Epoch {epoch} {"-"*13}')
        
        
        model.train()
        batch_accs = []
        batch_losses = [] 
        for x_train_batch, y_train_batch in train_dataloader: 
            
            x_train_batch = x_train_batch
            
            y_pred = model(x_train_batch)
            # Считаем лосс: передаем в функцию потерь предсказания и настоящие метки классов
            # long() для перевода в integer (так в torch называется целочисленный тип данных)
            #print(y_train_batch.float())
            #print(y_pred.squeeze(-1))
            loss = loss_fn(y_pred.squeeze(-1), y_train_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            
            # сравниваем предсказания с таргетом и добавляем в список значение точности
            batch_accs.append((torch.round(y_pred.cpu()) == y_train_batch).numpy().mean())

        history['train_losses'].append(np.mean(batch_losses))
        history['train_accs'].append(np.mean(batch_accs))

        # Validation
        model.eval()
    
        batch_accs = []
        batch_losses = []
        for batch, (x_test_batch, y_test_batch) in enumerate(valid_dataloader):
        
            y_pred = model(x_test_batch)
            batch_losses.append(loss_fn(y_pred.squeeze(-1), y_test_batch.float()).cpu())
            batch_accs.append((torch.round(y_pred.cpu()) == y_train_batch).numpy().mean())
        history['test_accs'].append(np.mean(batch_accs))
        history['test_losses'].append(np.mean([i.tolist() for i in batch_losses]))
        
        
        # печатаем результат
        print(f'train: accuracy {history["train_accs"][-1]:.4f}, loss {history["train_losses"][-1]:.4f}\n'
              f'test:  accuracy {history["test_accs"][-1]:.4f}, loss {history["test_losses"][-1]:.4f}')
        print(f'{"-"*35}')
        print()
    
    # печатаем графики
    plot_history(history)

    return history

history = None
history = torch_train(model, 3, optimizer, history)

torch.save(model.state_dict(), '/Users/paantur/Documents/GitHub/nn_project/savemodel.pt')

