import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


class GestureClassifier(nn.Module):
    def __init__(self, inputFile, featureSize , num_layers , nClasses=8, epochs=100, learning_rate=0.001, dropout=0.4):
        super().__init__()
        self.num_layers = num_layers
        self.size = featureSize
        self.dropout = dropout
        self.encoder = self.encoderBlock()
        self.encoder.load_state_dict(torch.load(os.path.join(inputFile, "model_encoder.pth")))
        self.classifier = nn.Sequential(
            nn.Linear(featureSize, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8)
        )
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_loss = []
        self.training_loss = []
        print(self.num_layers, self.size)


    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            for i in range(self.num_layers):
                x, _ = self.gru_layers[i](x)
                x = self.dropout_layers[i](x)
        
        x = x[:, -1, :]
        output = self.classifier(x)
        return output

    
    def train_model(self, trainLoader, validationLoader=None, earlyStop=5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        self.train()

        for param in self.encoder.parameters():
            param.requires_grad = False

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, data in enumerate(trainLoader):
                optimizer.zero_grad()
                output = self(data[0])
                loss = criterion(output, data[1])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(trainLoader)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss}')
            self.training_loss.append(avg_loss)

            if validationLoader is not None:
                self.validate(validationLoader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = earlyStop
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping triggered.")
                    break
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate * 0.1, weight_decay=0.01)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, data in enumerate(trainLoader):
                optimizer.zero_grad()
                output = self(data[0])
                loss = criterion(output, data[1])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(trainLoader)
            self.training_loss.append(avg_loss)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss}')

            if validationLoader is not None:
                self.validate(validationLoader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = earlyStop
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping triggered.")
                    break

    def predictTest(self, testLoader):
        self.eval()

        predictions = []
        true_labels = []

        with torch.no_grad(): 
            for data, labels in testLoader:
                outputs = self(data)
                # print(outputs, outputs.shape)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())  
                true_labels.extend(labels.cpu().numpy())     
        cm = confusion_matrix(predictions, true_labels)
        return predictions, true_labels, cm
    

    def validate(self, validation_data):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            criterion = nn.CrossEntropyLoss()
            for data, target in validation_data:
                output = self(data)
                loss = criterion(output, target)
                total_loss += loss.item()
            avg_loss = total_loss / len(validation_data)
            self.validation_loss.append(avg_loss)
            print(f'Validation Loss: {avg_loss}')


    def encoderBlock(self):
        layers = []
        self.gru_layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.gru_layers.append(nn.GRU(3, self.size, batch_first=True))
            else:
                self.gru_layers.append(nn.GRU(self.size, self.size, batch_first=True))
        
        self.gru_layers = nn.ModuleList(self.gru_layers)

        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_layers)])
        
        return nn.Sequential(*layers)
    
    def plotLosses(self, output_file):
        df = pd.DataFrame({'Validation loss': self.validation_loss, 'Training loss': self.training_loss})
        df.plot(kind='line', figsize=(8,6))
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title('Losses During Training')
        plt.savefig(os.path.join(output_file, 'Classifier_losses.png'))
        plt.show()