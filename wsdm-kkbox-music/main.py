# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.data import KKboxRSDataset, ImplicitProcesser
from sklearn.metrics import roc_auc_score

import numpy as np
import time

# Hyper Parameters
input_size = 100
hidden_size = 20
num_classes = 2
num_epochs = 3
batch_size = 128
learning_rate = 0.1

# user = 'TJU0Gfvy7FB+r89bWovPKXTjuApTCiv3xg/tt5shR78='
# song = 't0aT90DlS1TGncgnKoL0SvfAWEr3Dl72QBVcokmKfLc='
# song = '2bj5oqCPPzY6E0TPgwySkfj8/l/c+DVQBqnABx0qPSk='

start = time.time()
preprocessor = ImplicitProcesser(root='./data',
                                 feature_size=input_size,
                                 real_test=False,
                                 calculate_training_loss=True,
                                 save_dir='./model')

song_list = preprocessor.get_song_list()
for i in song_list[:10]:
    preprocessor.print_similar_song(i, 10)

# for i in song_list[1549:1559]:
#     preprocessor.print_similar_song(i, 10)
#
# for i in song_list[-10:]:
#     preprocessor.print_similar_song(i, 10)

train_dataset = KKboxRSDataset(train=True, processor=preprocessor)
test_dataset = KKboxRSDataset(train=False, processor=preprocessor)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):

        # Convert torch tensor then to Variable
        features = Variable(features.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 200 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

print("train in %0.2fs" % (time.time() - start))
# Test the Model
y_true, y_pred, y_prob = np.array([]), np.array([]), np.array([])
# for features, labels in test_loader:
for features, labels in train_loader:
    features = Variable(features.float())
    outputs = net(features)
    prob, predicted = torch.max(outputs.data, 1)
    y_true = np.concatenate((y_true, labels.numpy()))
    y_pred = np.concatenate((y_pred, predicted.numpy()))
    y_prob = np.concatenate((y_prob, prob.numpy()))

print('ROC score of the network on the test datasets(prob): %0.2f%%' % (roc_auc_score(y_true, y_prob)))
print('ROC score of the network on the test datasets(pred): %0.2f%%' % (roc_auc_score(y_true, y_pred)))

# # Save the Model
# torch.save(net.state_dict(), 'model.pkl')
