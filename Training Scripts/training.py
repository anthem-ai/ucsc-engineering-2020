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

# VECTOR_LENGTH = 100

# #see results variable in tenFiles.py on Aman's laptop
# tenFileData = np.load("tenFiles.npy", allow_pickle=True)

# def text_preprocessing(phrase):
#     phrase = phrase.lower()
#     phrase = phrase.replace('&', ' ')
#     phrase = phrase.replace('-', '')
#     phrase = phrase.replace(',',' ')
#     phrase = phrase.replace('.',' ')
#     phrase = phrase.replace('/',' ')
#     phrase = phrase.replace('(',' ')
#     phrase = phrase.replace(')',' ')
#     phrase = phrase.replace('[',' ')
#     phrase = phrase.replace(']',' ')
#     phrase = phrase.replace(':',' ')
#     phrase = phrase.replace(';',' ')
#     sentence = phrase.split(' ')
#     return sentence

# #patient sentence arrays holds each of the sentences, seperated by patiend
# #master sentence arrays holds all of the patients and loses information
# # about which patient the sentence corresponds to
# #master sentence used only for training the word2vec model
# #patientConcatenatedSentences contains joined versions of each of the sentences in patient sentence array
# #same for masterConcatenatedSentences
# patientSentenceArrays = []
# masterSentences = []
# masterConcatenatedSentences = []
# patientContatenatedSentences = []
# labelArray = []
# for sentenceArray, label in tenFileData:
#   curPat = []
#   curPatConc = []
#   for sentence in sentenceArray:
#     concatSentence = ""
#     for word in sentence:
#       concatSentence += word + " "
#     concatSentence = text_preprocessing(concatSentence)
#     if not sentence == []:
#       # concatSentence = ""
#       masterConcatenatedSentences.append(concatSentence)
#       masterSentences.append(sentence)
#     curPatConc.append(concatSentence)
#     curPat.append(sentence)
#   patientSentenceArrays.append(curPat)
#   patientContatenatedSentences.append(curPatConc)
#   labelArray.append(label)


# # concat_word2Vec = Word2Vec(masterConcatenatedSentences, size= VECTOR_LENGTH, window=2, min_count=1, workers=2, sg=1)
# non_concat_word2vec = Word2Vec(masterSentences, size= VECTOR_LENGTH, window=2, min_count=1, workers=2, sg=1)


# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []

#     j = 0
#     for word in model.wv.vocab:
#         j += 1
#         # if j < 400:
#         #   continue
#         tokens.append(model[word])
#         labels.append(word)
        
    
    
#     tsne_model = TSNE(perplexity= 100, n_components=2, init='pca', n_iter=2500)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i][0:10],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()


# #takes each patient's sentences and turns them into one of two things: a matrix and a long vector (flattened matrix)
# def patientToMatrix(patientSentences, w2vmodel):
#   matrix = []
#   flattenedVector = []
#   for s in patientSentences:
#     matrixInsert = torch.zeros(VECTOR_LENGTH)
#     if len(s) == 0:
#       matrix.append(matrixInsert)
#       for el in matrixInsert:
#         flattenedVector.append(el)
#     else:
#       # print(s)
#       for word in s:
#         # print(word)
#         matrixInsert += torch.Tensor(w2vmodel[word])
#       matrixInsert /= len(s)
#       for el in matrixInsert:
#         flattenedVector.append(el)
#       matrix.append(matrixInsert)
#   return (torch.stack(matrix), torch.tensor(flattenedVector))

# #takes a dataset of patients and returns a 3D matrix with numerical values, and a reduced dimension 2D matrix with the same numerical values
# def modelReadyPatientData(patientSentences, w2vmodel):
#   # w2vmodel.wv.vocab
#   patientMatrices = []
#   flattenedPatients = []
#   for patient in patientSentences:
#     patMat, patVec = patientToMatrix(patient, w2vmodel)
#     patientMatrices.append(patMat)
#     flattenedPatients.append(patVec)
#   return(torch.stack(patientMatrices), torch.stack(flattenedPatients))

# print("started matrix conversion")
    
# nonConcatPatientMatrices, nonConcatPatientVectors = modelReadyPatientData(patientSentenceArrays, non_concat_word2vec)

# trainX, testX, trainY, testY = tts(nonConcatPatientVectors, labelArray)

# supportVectorMachine = svm.LinearSVC()

# supportVectorMachine.fit(trainX, trainY)

# print("SVM Score: ", supportVectorMachine.score(testX, testY))

# trainX, testX, trainY, testY = tts(nonConcatPatientMatrices, labelArray)

os.chdir(sys.argv[1])

files = os.listdir()

data = []
for f in files:
  try:
    data.append(pd.read_csv(f))
  except:
    continue
os.chdir("../")

numColumns = len(data[0].columns) - 2

train_array = []
svmArray = []
label_array = []
for df in data:
  label = df[sys.argv[2]][0]
  df = df.drop(columns = ['Unnamed: 0', sys.argv[2]])
  # print(df.values)
  matrix = torch.Tensor(df.values)
  train_array.append(matrix)
  svmArray.append(df.values.reshape(-1))
  label_array.append(label)

trainX, testX, trainY, testY = tts(train_array, label_array, test_size = 0.3, stratify = label_array)
trainXSvm, testXSvm, trainYSvm, testYSvm = tts(svmArray, label_array, test_size = 0.3, stratify = label_array)

train_set = []
for i in range(len(trainX)):
  # print(trainX[i], trainY[i])
  train_set.append((trainX[i].reshape(-1, numColumns, 12), trainY[i]))
  # train_set.append((train_array[i], label_array[i]))

test_set = []
for i in range(len(testX)):
  # print(trainX[i], trainY[i])
  test_set.append((testX[i][0:24].reshape(-1, numColumns, 12), testY[i]))
  # train_set.append((train_array[i], label_array[i]))
BATCH_SIZE = 10
train_set = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_set = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

baseline = svm.LinearSVC()
baseline.fit(trainXSvm, trainYSvm)
print("SVM score:", baseline.score(testXSvm, testYSvm))

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(FeedForward, self).__init__()
        self.fc1= nn.Linear(input_size, hidden1_size)
        self.ReLU1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.ReLU2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)
    
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
        # out = self.ReLU3(out)
        # print(out)
        return out

    def num_flat_features(self, x): # To see dimensions of layers
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

FFN = FeedForward(numColumns * 12, 128, 64, 2)

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
    outputs = FFN(inputs)
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
    self.conv1 = nn.Conv2d(1, 6, (numColumns - 1,4))
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
    x = torch.sigmoid(self.fc3(x))
    return x
  
  def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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