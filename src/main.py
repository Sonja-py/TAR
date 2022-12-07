# import numpy as np
import matplotlib.pyplot as plt
import utils as U

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def model_train(model, train_data, test_data, n_epoch=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(dataset=train_data,
                    batch_size=20, shuffle=True,
                    collate_fn=lambda x: U.process_data(x, 'train'))
    test_loader = DataLoader(dataset=test_data,
                    batch_size=20, shuffle=True,
                    collate_fn=lambda x: U.process_data(x, 'test'))
    criterion = nn.CTCLoss(blank=0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    training_loss = [0]*n_epoch
    accuracy = [0]*n_epoch
      
    # Training the model
    for epoch in range(n_epoch):
    
      model.train()
      batch_loss = 0
    
      for i, data in enumerate(train_loader):
        
        spectrograms, labels, input_len, label_len = data
        spectrograms, labels = spectrograms.to(device), labels.to(device)
    
        optimizer.zero_grad()
        prediction = model(spectrograms)
        prediction = prediction.cpu()
        loss = criterion(prediction, labels, input_len, label_len)
        batch_loss += loss.item()
        
        # Update weights
        loss.backward()
        optimizer.step()
#        scheduler.step()
                  
      training_loss[epoch] = (batch_loss/len(train_loader))
      
      test_acc = model_test(model, test_loader, device)
      accuracy[epoch] = test_acc
      
      if epoch % 5 == 0:
        print(f'Epoch {epoch}: {training_loss[-1]}  {accuracy[-1]}')
          
              
    # Final loss and accuracy
    print(f'Epoch {n_epoch}:  {training_loss[-1]}  {accuracy[-1]}')
        
    plt.figure()
    plt.plot(training_loss, color='red')
    plt.plot(accuracy, color='blue')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


def model_test(model, test_loader, device):
  batch_accuracy = 0
  model.eval()

  for i, data in enumerate(test_loader):
    spectrograms, labels, input_len, label_len = data
    spectrograms, labels = spectrograms.to(device), labels.to(device)

    prediction = model(spectrograms)
    prediction = prediction.cpu()

    batch_accuracy += U.accuracy(prediction, labels)

  accuracy = batch_accuracy/len(test_loader)  
  return accuracy