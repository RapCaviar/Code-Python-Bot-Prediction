import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


data_df = pd.read_csv('Ethereum.csv')  


data_df['Date'] = pd.to_datetime(data_df['Date'])

data_df = data_df[(data_df['Date'] >= '2019-01-01') & (data_df['Date'] <= '2019-12-31')]

prices = data_df['Price'].values
dates = data_df['Date'].values


in_scaler = MinMaxScaler(feature_range=(-1, 1))
prices = in_scaler.fit_transform(prices.reshape(-1, 1))


train_size = int(len(prices) * 0.8)  
test_size = len(prices) - train_size
train_data, test_data = prices[0:train_size], prices[train_size:len(prices)]
test_dates = dates[train_size:len(prices)]  

# LSTM thingy model 
ws = 9 # window size
def data_loader(train, test, dates=None):
    train_seq, train_tar = [], []
    test_seq, test_tar = [], []
    test_seq_dates = []

    for i in range(len(train) - ws):
        train_seq.append(train[i:i + ws])
        train_tar.append(train[i + ws])
    for i in range(len(test) - ws):
        test_seq.append(test[i:i + ws])
        test_tar.append(test[i + ws])
        test_seq_dates.append(dates[i + ws] if dates is not None else None)
        
    return np.array(train_seq), np.array(train_tar), np.array(test_seq), np.array(test_tar), np.array(test_seq_dates)

train_seq, train_tar, test_seq, test_tar, test_seq_dates = data_loader(train_data, test_data, test_dates)


np.save('in_scaler.npy', in_scaler)
np.save('test_dates.npy', test_seq_dates)
