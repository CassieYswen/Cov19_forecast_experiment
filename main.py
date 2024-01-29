#TODO
# 1. Add more data
# minmax scale X variables
# long term? more than 13?
# test min max
# BS 32, 16 ...
# batch norm? activation? 
# argparse
#wandb


import numpy as np
from create_data import generate_data_true, generate_data_more_var, generate_data_ar2,cumulative_gamma
import pandas as pd
from scipy.stats import gamma, norm, poisson, logistic
from numpy.linalg import inv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from flexible_models import FlexibleRNNModel,custom_loss,TimeSeriesDataset
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
#set seed for reproducibility
np.random.seed(42)
# Define the common parameters
T = 120
NoCov = 2
R_0 = 3
I_0 = 500
bias_corr_const = np.exp(-0.001 / 2)
Omega = np.array([cumulative_gamma(i, 2.5, 3) - cumulative_gamma(i-1, 2.5, 3) 
                  for i in range(1, 26)]) / cumulative_gamma(25, 2.5, 3)
OraclePhi = np.array([0.5, 0.7])
OracleBeta = np.array([-0.02, -0.125]).reshape(NoCov, 1)
OracleBeta1 = np.array([-0.02, -0.125, -0.03]).reshape(NoCov + 1, 1)
OraclePhi1 = np.array([0.5, 0.5, 0.3])
Rmin1 = 2


# Generate data
df0 = generate_data_true(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta, bias_corr_const)
df = generate_data_true(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta, bias_corr_const)
df1 = generate_data_more_var(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta1, bias_corr_const)
df2 = generate_data_ar2(T, NoCov, R_0, I_0, Omega, OraclePhi1, OracleBeta, bias_corr_const, Rmin1)

# fig, ax =plt.subplots(1,1)
# ax.plot(df0['R'],'b',label = 'test')
# ax.plot(df['R'],'r',label = 'train')

# # Optionally save the DataFrames for datachecking
# #df0.to_csv('validate_data.csv', index=False)
# #df.to_csv('sampled_data.csv', index=False)
# #df1.to_csv('misspec_var.csv', index=False)
# df2.to_csv('misspec_ar2.csv', index=False)
#import pdb;pdb.set_trace()

#try RNN-LSTM
# Model initialization
model_lstm = FlexibleRNNModel(rnn_type='GRU', input_size=3, hidden_size=10, output_size=1, num_layers=2, dropout=0.2)

# Assuming R_0 is constant and set to 3
# R_0 = 3 as defined before
sequence_length=14
dataset1 = TimeSeriesDataset(df, sequence_length)
dataset2 = TimeSeriesDataset(df0, sequence_length)
dataset3 = TimeSeriesDataset(df, sequence_length)
train_loader = DataLoader(dataset1, batch_size=64, shuffle=True)

test_loader = DataLoader(dataset2, batch_size=120, shuffle=False)
test_loader2 = DataLoader(dataset3, batch_size=120, shuffle=False)
testdata = next(iter(test_loader))
#np.log(df0['R'].values[14:]), testdata[1]
#import pdb;pdb.set_trace()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
for epoch in tqdm(range(num_epochs)):
    model_lstm.train()
    total_loss = 0
    n_iter = 0
    for features, target in  train_loader :
        features = features.view(-1, sequence_length-1, 3)  # Reshape for RNN
        target = target.unsqueeze(-1)  # Add an extra dimension to the target
        #import pdb;pdb.set_trace()
        optimizer.zero_grad()
        output = model_lstm(features).squeeze()
        output = output.view(-1, 1)
        #import pdb;pdb.set_trace()
        # if epoch % 50 ==49:
        #     #import pdb;pdb.set_trace()
        #     print(output.view(-1), '\n', target.view(-1), '\n', features[:,-1].view(-1))
        loss = custom_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_iter += 1
    if epoch % 50 ==0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')


    if epoch % 500 ==0:
        # Add evaluation logic as needed
        model_lstm.eval()  # Set the model to evaluation mode
        mse_criterion = torch.nn.MSELoss()  # MSE loss function
        total_mse = 0

        with torch.no_grad():  # Disable gradient calculation
            for features, target in tqdm(test_loader):
                features = features.view(-1, sequence_length-1, 3)  # Reshape for RNN
                target = target.unsqueeze(-1)  # Add an extra dimension to the target
                output = model_lstm(features).squeeze()
                mse = mse_criterion(output, target)
                total_mse += mse.item()

        avg_mse = total_mse / len(test_loader)
        print(f'Average MSE on Test Set: {avg_mse}')

        predicted_R_test = []
        with torch.no_grad():  # Disable gradient calculation
            for features, label in test_loader:
                features = features.view(-1, sequence_length-1, 3)  # Reshape for RNN
                output = model_lstm(features).squeeze()
                predicted_R_test.extend(output.tolist())

        predicted_R_test1 = []
        with torch.no_grad():  # Disable gradient calculation
            for features, label in test_loader2:
                features = features.view(-1, sequence_length-1, 3)  # Reshape for RNN
                output = model_lstm(features).squeeze()
                predicted_R_test1.extend(output.tolist())


        # Convert predictions back to the original scale if necessary
        predicted_R_test = np.exp(predicted_R_test)  # Apply exponential if the log was taken
        predicted_R_test1 = np.exp(predicted_R_test1)
        # Extract actual R values from the training data
        actual_R_test = df0['R']
        actual_R_test1 = df['R']

        # import pdb; pdb.set_trace()
        # Prepare the plot
        fig,ax=plt.subplots(1,2,figsize=(20, 6))
        ax[0].plot(actual_R_test, label='Actual R (Test)', color='blue')
        ax[0].plot(np.arange(14,120), predicted_R_test, label='Predicted R (Test)', color='red')
        ax[0].set_xlabel('Time Step')
        ax[0].set_ylabel('R')
        ax[0].set_title('Actual and Predicted R')
        ax[0].legend()
        ax[1].plot(actual_R_test1, label='Actual R (Train)', color='blue')
        ax[1].plot(np.arange(14,120), predicted_R_test1, label='Predicted R (Train)', color='red')
        ax[1].set_xlabel('Time Step')
        ax[1].set_ylabel('R')
        ax[1].set_title('Actual and Predicted R')
        ax[1].legend()

        #plt.show() #
        fig.savefig('prediction_{}.png'.format(epoch))