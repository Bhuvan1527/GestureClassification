import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

class TimeNet(nn.Module):
    def __init__(self, size, num_layers, inputSize=3 ,batch_size=32, dropout=0.0, model_name=None):
        super().__init__()
        self.size = size
        self.inputSize = inputSize
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.model_name = model_name
        self.validation_loss = []
        self.training_loss = []
        self.encoder = self.encoder_block()
        self.decoder = self.decoder_block()
    
    def encoder_block(self):
        layers = []
        
        self.gru_layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.gru_layers.append(nn.GRU(self.inputSize, self.size, batch_first=True))
            else:
                self.gru_layers.append(nn.GRU(self.size, self.size, batch_first=True))
        
        self.gru_layers = nn.ModuleList(self.gru_layers)

        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_layers)])
        
        return nn.Sequential(*layers)

    def decoder_block(self):
        layers = []
        
        self.gru_layers_decoder = nn.ModuleList([nn.GRU(self.size, self.size, batch_first=True)
                                                 for _ in range(self.num_layers)])
        self.dropout_layers_decoder = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_layers)])
        
        self.output_layer = nn.Linear(self.size, self.inputSize)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        for i in range(self.num_layers):
            x, _ = self.gru_layers[i](x)
            x = self.dropout_layers[i](x)
        
        for i in range(self.num_layers):
            x, _ = self.gru_layers_decoder[i](x)
            x = self.dropout_layers_decoder[i](x)

        x = self.output_layer(x)  
        return x

    def train_model(self, train_data, nb_epoch, lr=0.01, validation_data=None, early_stop=5):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss() 
        
        best_loss = float('inf')
        patience = early_stop
        log_dir = self.get_run_id()
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        for epoch in range(nb_epoch):
            self.train()
            total_loss = 0
            for batch_idx, data in enumerate(train_data):
                optimizer.zero_grad()
                output = self(data[0])
                loss = criterion(output, data[0])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            print(f'Epoch {epoch+1}/{nb_epoch}, Loss: {avg_loss}')
            self.training_loss.append(avg_loss)
            
            if validation_data is not None:
                self.validate(validation_data)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = early_stop
                self.save_model(os.path.join(log_dir, "model_best.pth"))
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping triggered.")
                    break
            
            
        
        self.save_encoder(os.path.join(log_dir, "model_encoder.pth"))
        return log_dir
    
    def validate(self, validation_data):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            criterion = nn.MSELoss()
            for data in validation_data:
                output = self(data[0])
                loss = criterion(output, data[0])
                total_loss += loss.item()
            avg_loss = total_loss / len(validation_data)
            self.validation_loss.append(avg_loss)
            print(f'Validation Loss: {avg_loss}')
    
    def get_run_id(self):
        run_id = f"{self.size}_x{self.num_layers}_drop{int(100 * self.dropout)}"
        if self.model_name:
            run_id += f"_{self.model_name}"
        return run_id

    def save_model(self, output_file):
        torch.save(self.state_dict(), output_file)
    
    def save_encoder(self, output_file):
        try:
            torch.save(self.encoder.state_dict(), output_file)
        except Exception as e:
            print(f"Error saving encoder: {e}")
    
    def load_model(self, input_file):
        self.load_state_dict(torch.load(input_file))
    
    def plotLosses(self, output_file):
        df = pd.DataFrame({'Validation loss': self.validation_loss, 'Training loss': self.training_loss})
        df.plot(kind='line', figsize=(8,6))
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title('Losses During Training')
        plt.savefig(os.path.join(output_file, 'losses.png'))
        plt.show()

