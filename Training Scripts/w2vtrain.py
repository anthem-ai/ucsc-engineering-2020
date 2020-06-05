import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split as tts
from matplotlib import pyplot as plt
from zipfile import ZipFile
import gensim
from gensim.models import FastText, Word2Vec
from sklearn import svm
from sklearn.manifold import TSNE
import sys

os.chdir(sys.argv[1])
files = os.listdir()

dataArray = []
vectorArray = []
labelArray = []

for f in files:
    currentMatrix = []
    openedFile = np.load(f, allow_pickle=True)

    listOfLists = openedFile[0]

    for l in listOfLists:
        if len(l) == 0:
            currentMatrix.append(np.zeros(100))
        else:
            currentMatrix.append(l)
    
    currentMatrix = np.stack(currentMatrix)
    currentMatrix = torch.from_numpy(currentMatrix)
    currentMatrix = currentMatrix.type(torch.FloatTensor)
    vectorArray.append(currentMatrix.reshape(-1).numpy())
    dataArray.append(currentMatrix)

    labelArray.append(openedFile[1])


vecTrainX, vecTestX, vecTrainy, vecTesty = tts(vectorArray, labelArray, test_size = 0.3, stratify = labelArray)
trainX, testX, trainY, testY = tts(dataArray, labelArray, test_size = 0.3, stratify = labelArray)

train_set = []
for i in range(len(trainX)):
  # print(trainX[i], trainY[i])
  train_set.append((trainX[i].reshape(-1, 100, 24), trainY[i]))
  # train_set.append((train_array[i], label_array[i]))

test_set = []
for i in range(len(testX)):
  # print(trainX[i], trainY[i])
  test_set.append((testX[i].reshape(-1, 100, 24), testY[i]))
  # train_set.append((train_array[i], label_array[i]))
BATCH_SIZE = 10
train_set = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_set = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)


baseline = svm.LinearSVC()
baseline.fit(vecTrainX, vecTrainy)
print("SVM score:", baseline.score(vecTestX, vecTesty))

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, num_classes):
        super(FeedForward, self).__init__()
        self.fc1= nn.Linear(input_size, hidden1_size)
        self.ReLU1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.ReLU2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden1_size, hidden2_size)
        self.ReLU2 = nn.ReLU()
        self.fc4 = nn.Linear(hidden2_size, num_classes)
    
    def  forward(self,x):
        # print(x)
        x = x.view(-1, self.num_flat_features(x))
        out = self.fc1(x)
        out = self.ReLU1(out)
        # print(out)
        # r_out, (h_n, h_c) = self.rnn(out.view(-1,1,512), None)
        # print(r_out)
        # out = r_out[:, -1, :]
        # out = self.ReLU2(out)
        out = self.fc2(out)
        out = self.ReLU2(out)
        out = self.fc3(out)
        m = nn.Dropout(p=0.2)
        out = m(out)
        out = self.ReLU3(out)
        out = F.softmax(self.fc4(out))
        # print(out)
        return out

    def num_flat_features(self, x): # To see dimensions of layers
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(RNN, self).__init__()
        self.fc1= nn.Linear(input_size, hidden1_size)
        self.ReLU1 = nn.ReLU()
        self.rnn = nn.LSTM(
            input_size=hidden1_size,
            hidden_size=hidden2_size, 
            num_layers=1,           
            batch_first=True,       
        )
        self.ReLU2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)
    
    def forward(self,x):
        out = None
        cumOut = []
        for patient in x:
          for mat in patient:
            mat = mat.transpose(1, 0)
            # ordering = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            for row in mat:
              out = self.rowForward(row)[0]
          cumOut.append(out)
        return torch.stack(cumOut)

    def rowForward(self, x):
        out = self.fc1(x)
        out = self.ReLU1(out)
        r_out, (h_n, h_c) = self.rnn(out.view(-1,1,256), None)
        out = r_out[:, -1, :]
        out = self.ReLU2(out)
        out = F.softmax(self.fc3(out))
        return out

model = RNN(100, 256, 128, 2)


# print("========================START RNN TRAINING===========================")

# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
# loss_f = nn.CrossEntropyLoss()

# lossX = np.arange(30)
# lossY = []
# testingY = []
# model.train()
# numEpoch = 1
# for i in range(30):
#   cum_loss = 0
#   epoch_loss = []
#   loss_per_batch = []
#   for inputs, labels in train_set:
#     # print(inputs.shape)
#     # print(labels)
#     # break
#     optimizer.zero_grad()
#     outputs = model(inputs.float())
#     # print(outputs[0])
#     # print(outputs)
#     # print(outputs.squeeze(2))
#     loss = loss_f(outputs, labels.long())
#     # loss.requires_grad = True
#     loss.backward()
#     optimizer.step()
#     # cum_loss += loss.data.item()
#     loss_per_batch.append(loss.data.item())
#     #add min, max, avg, std of loss
#     # print("batch loss avg")
#   loss_per_batch = torch.Tensor(loss_per_batch)
#   print("Epoch", numEpoch, "Loss avg: ", torch.mean(loss_per_batch) / BATCH_SIZE)
#   print("Epoch", numEpoch, "Loss std: ", (torch.std(loss_per_batch) / BATCH_SIZE) ** 0.5)
#   print("Epoch", numEpoch, "Loss min: ", torch.min(loss_per_batch) / BATCH_SIZE)
#   print("Epoch", numEpoch, "Loss max: ", torch.max(loss_per_batch) / BATCH_SIZE)
#   with torch.no_grad():
#     total = 0
#     correct = 0
#     for inputs, labels in test_set:
#       out = model(inputs)
#       for i in range(len(out)):
#         prediction = out[i].argmax().item()
#         # print(prediction)
#         total += 1
#         if prediction == labels[i]:
#           # print("Correct: ", prediction)
#           correct += 1
#     accuracy = correct / total
#     print("Epoch", numEpoch, "Test Accuracy: ", accuracy)
#     testingY.append(accuracy)

#     for inputs, labels in train_set:
#       out = model(inputs)
#       for i in range(len(out)):
#         prediction = out[i].argmax().item()
#         # print(prediction)
#         total += 1
#         if prediction == labels[i]:
#           # print("Correct: ", prediction)
#           correct += 1
#     accuracy = correct / total
#     print("Epoch", numEpoch, "Train Accuracy: ", accuracy)

#   lossY.append(cum_loss)
#   numEpoch += 1

print("========================START FFN TRAINING===========================")

FFN = FeedForward(100 * 24, 128, 64, 2)

optimizer = optim.SGD(FFN.parameters(), lr=0.1, momentum=0.5)
loss_f = nn.CrossEntropyLoss()

lossX = np.arange(100)
lossY = []
testingY = []
FFN.train()
numEpoch = 1
for i in range(100):
  cum_loss = 0
  epoch_loss = []
  loss_per_batch = []
  for inputs, labels in train_set:
    # print(inputs.shape)
    # print(labels)
    # break
    optimizer.zero_grad()
    outputs = FFN(inputs.float())
    # print(outputs[0])
    # print(outputs)
    # print(outputs.squeeze(2))
    loss = loss_f(outputs, labels.long())
    # loss.requires_grad = True
    loss.backward()
    optimizer.step()
    # cum_loss += loss.data.item()
    loss_per_batch.append(loss.data.item())
    #add min, max, avg, std of loss
    # print("batch loss avg")
  loss_per_batch = torch.Tensor(loss_per_batch)
  print("Epoch", numEpoch, "Loss avg: ", torch.mean(loss_per_batch) / BATCH_SIZE)
  print("Epoch", numEpoch, "Loss std: ", (torch.std(loss_per_batch) / BATCH_SIZE) ** 0.5)
  print("Epoch", numEpoch, "Loss min: ", torch.min(loss_per_batch) / BATCH_SIZE)
  print("Epoch", numEpoch, "Loss max: ", torch.max(loss_per_batch) / BATCH_SIZE)
  with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in test_set:
      out = FFN(inputs)
      for i in range(len(out)):
        prediction = out[i].argmax().item()
        # print(prediction)
        total += 1
        if prediction == labels[i]:
          # print("Correct: ", prediction)
          correct += 1
    accuracy = correct / total
    print("Epoch", numEpoch, "Test Accuracy: ", accuracy)
    testingY.append(accuracy)

    for inputs, labels in train_set:
      out = FFN(inputs)
      for i in range(len(out)):
        prediction = out[i].argmax().item()
        # print(prediction)
        total += 1
        if prediction == labels[i]:
          # print("Correct: ", prediction)
          correct += 1
    accuracy = correct / total
    print("Epoch", numEpoch, "Train Accuracy: ", accuracy)

  lossY.append(cum_loss)
  numEpoch += 1

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, (100 - 1,4))
    self.conv2 = nn.Conv2d(6, 16, (1, 7))
    # self.conv3 = nn.Conv2d(16, 32, 3)
    # self.conv4 = nn.Conv2d(32, 64, 2)
    self.fc1 = nn.Linear(240, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 2)
  
  def forward(self, x):
    # print(x)
    x = self.conv1(x)
    # x = F.avg_pool2d(x, (2, 2))
    # x = nn.Dropout(x)
    x = self.conv2(torch.relu(x))
    x = F.avg_pool2d(torch.relu(x), (2,1))
    # x = self.conv3(torch.relu(x))
    # x = F.avg_pool2d(torch.relu(x), 2)
    # x = self.conv4(torch.relu(x))
    # x = F.avg_pool2d(torch.relu(x), (2,2))
    x = x.view(-1, self.num_flat_features(x))
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = F.softmax(self.fc3(x))
    return x
  
  def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


print("========================START CNN TRAINING===========================")
CoNN = CNN()
optimizer = optim.SGD(CoNN.parameters(), lr=0.1, momentum=0.5)
loss_f = nn.CrossEntropyLoss()



lossX = np.arange(100)
lossY = []
testingY = []
CoNN.train()
numEpoch = 1
for i in range(100):
  # cum_loss = 0
  loss_per_batch = []
  for inputs, labels in train_set:
    # print(inputs.shape)
    # print(labels)
    # break
    optimizer.zero_grad()
    outputs = CoNN(inputs)
    # print(outputs[0])
    # print(outputs)
    # print(outputs.squeeze(2))
    loss = loss_f(outputs, labels.long())
    # loss.requires_grad = True
    loss.backward()
    optimizer.step()
    loss_per_batch.append(loss.data.item())
  loss_per_batch = torch.tensor(loss_per_batch)
  print("Epoch", numEpoch, "Loss avg: ", torch.mean(loss_per_batch) / BATCH_SIZE)
  print("Epoch", numEpoch, "Loss std: ", (torch.std(loss_per_batch) / BATCH_SIZE) ** 0.5)
  print("Epoch", numEpoch, "Loss min: ", torch.min(loss_per_batch) / BATCH_SIZE)
  print("Epoch", numEpoch, "Loss max: ", torch.max(loss_per_batch) / BATCH_SIZE)
  with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in test_set:
      out = CoNN(inputs)
      for i in range(len(out)):
        prediction = out[i].argmax().item()
        # print(prediction)
        total += 1
        if prediction == labels[i]:
          # print("Correct: ", prediction)
          correct += 1
    accuracy = correct / total
    print("Epoch", numEpoch, "Test Accuracy: ", accuracy)
    testingY.append(accuracy)

    for inputs, labels in train_set:
      out = CoNN(inputs)
      for i in range(len(out)):
        prediction = out[i].argmax().item()
        # print(prediction)
        total += 1
        if prediction == labels[i]:
          # print("Correct: ", prediction)
          correct += 1
    accuracy = correct / total
    print("Epoch", numEpoch, "Train Accuracy: ", accuracy)
    # testingY.append(accuracy)

  lossY.append(cum_loss)
  numEpoch += 1
  # break
