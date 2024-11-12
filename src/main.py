import argparse
import os
import warnings
from Model.timenet import TimeNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from Data.DataRelatedFunctions import preProcessData, preProcessLabels
from Model.classifier import GestureClassifier
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", help="HAR folder in the repository")
    parser.add_argument("--embeddings-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--classifier-input-folder", help="Gesture folder in the repository")
    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    embeddingsDim = args.embeddings_dim
    nEpochs = args.n_epochs
    batchSize = args.batch_size
    numLayers = 9
    modelName = 'model_1'
    dropout = args.dropout

    encoderInputFolder = args.input_folder
    trainData = torch.load(f'{encoderInputFolder}/train.pt')['samples']
    valData = torch.load(f'{encoderInputFolder}/val.pt')["samples"]

    trainData, valData = preProcessData(trainData), preProcessData(valData)
    trainData = trainData.to(device=device)
    valData = valData.to(device=device)

    trainDataset = TensorDataset(trainData)
    valDataset = TensorDataset(valData)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

    enc = TimeNet(size=embeddingsDim, num_layers=3, inputSize=3, batch_size=batchSize, dropout=dropout, model_name=modelName)
    enc.to(device=device)
    logDir = enc.train_model(trainLoader, nb_epoch=nEpochs, lr=args.learning_rate, validation_data=valLoader, early_stop=args.early_stop)
    enc.plotLosses(logDir)


    print("Finished training")



    # Classifier
    classifierInputFolder = args.classifier_input_folder
    trainData_c = torch.load(os.path.join(classifierInputFolder, 'train.pt'))
    xTrain_c, yTrain_c = trainData_c["samples"], trainData_c["labels"]

    valData_c = torch.load(os.path.join(classifierInputFolder, 'val.pt'))
    xVal_c, yVal_c = valData_c["samples"], valData_c["labels"]

    xTrain_c, yTrain_c = preProcessData(xTrain_c), preProcessLabels(yTrain_c)
    xVal_c, yVal_c = preProcessData(xVal_c), preProcessLabels(yVal_c)

    xTrain_c, yTrain_c = xTrain_c.to(device=device), yTrain_c.to(device=device)
    xVal_c, yVal_c = xVal_c.to(device=device), yVal_c.to(device=device)

    trainDataset_c = TensorDataset(xTrain_c.to(torch.float32), yTrain_c)
    valDataset_c = TensorDataset(xVal_c.to(torch.float32), yVal_c)

    trainLoader_c = DataLoader(trainDataset_c, batch_size=batchSize, shuffle=True)
    valLoader_c = DataLoader(valDataset_c, batch_size=batchSize, shuffle=True)

    gestureClassifier = GestureClassifier(logDir, embeddingsDim, num_layers=3 ,nClasses=len(yTrain_c.unique()), epochs=30, learning_rate=0.001, dropout=dropout)
    gestureClassifier.to(device=device)
    gestureClassifier.train_model(trainLoader=trainLoader_c, validationLoader=valLoader_c, earlyStop=args.early_stop)

    testData_c = torch.load(os.path.join(classifierInputFolder, 'test.pt'))
    xTest_c, yTest_c = testData_c["samples"], testData_c["labels"]
    xTest_c, yTest_c = preProcessData(xTest_c), preProcessLabels(yTest_c)

    xTest_c, yTest_c = xTest_c.to(device=device), yTest_c.to(device=device)
    testDataset_c = TensorDataset(xTest_c.to(torch.float32), yTest_c)
    testLoader_c = DataLoader(testDataset_c)
    predicted, actual, cm = gestureClassifier.predictTest(testLoader=testLoader_c)
    gestureClassifier.plotLosses(logDir)
    plot_confusion_matrix(cm, yTrain_c.unique().tolist())
    acc = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            acc += 1
    
    print("Training finished with {}".format(acc / len(predicted)))

    # print(predicted)
    # print(actual)


    
