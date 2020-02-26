#importing all important packages
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import datetime as dt
from datetime import date
import pandas_datareader.data as web

def date_prep():
    today=date.today()
    d1 = today.strftime("%d/%m/%Y")
    d=d1.split('/')
    day=int(d[0])
    month=int(d[1])
    year=int(d[2])
    year_19=year-19
    end = dt.datetime(year, month, day)
    if month==2 and day==29 and not (((year_19 % 4 == 0) and (year_19 % 100 != 0)) or (year_19 % 400 == 0)):
        start = dt.datetime(year_19, month, day-1)
    else:
        start = dt.datetime(year_19, month, day)
    return start,end

def prepare_csv(comp_name):
    start,end=date_prep()
    try:
        df = web.DataReader(comp_name, 'yahoo', start, end)
        df.to_csv(comp_name+ '.csv')
    except Exception as e:
        print(e)

def create_model(x_train):
    ####MODEL cretion #####
    model = Sequential()  # define the Keras model
    model.add(LSTM(units=240, return_sequences=True,input_shape=(x_train.shape[1], 4)))  # 120 neurons in the hidden layer
    ##return_sequences=True makes LSTM layer to return the full history including outputs at all times
    model.add(Dropout(0.2))
    model.add(LSTM(units=240, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=240, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=240, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=240, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=240, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    # adding optimizer and loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(comp_name):
    #reading a csv file
    prepare_csv(comp_name)
    df= pd.read_csv(comp_name + '.csv')
    df.dropna(how='any', inplace=True)
    trainingd = df.iloc[:, 1:5].values
    sc = MinMaxScaler()
    training_set_scaled = sc.fit_transform(trainingd)
    x_train = []
    y_train = []
    timestamp = 60
    future_predict = 1
    length = len(trainingd)
    for i in range(timestamp, length, future_predict):
        if i + future_predict < length:
            x_train.append(training_set_scaled[i - timestamp:i, ])
            l = list(training_set_scaled[i:i + future_predict, 3])
            y_train.extend(l)
        else:
            break

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (-1, future_predict))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))

    model = create_model(x_train)
    # training the model
    model.fit(x_train, y_train, epochs=25, batch_size=32)  # ,callbacks=[tensorboard])
    # storing the model
    model.save('StockPredictor'+ comp_name +'.model')
    os.remove(comp_name+'.csv')


def total_train():
    comp_name=input("Enter the company name GOOG and AAPL is betteer ")
    comp_name=comp_name.upper()
    train_model(comp_name)
#### Server code
'''today = date.today()
d1 = today.strftime("%d/%m/%Y")
d2 = today.strftime("%d/%m/%Y")
while True:
    t.sleep(86400)
    d2=today.strftime("%d/%m/%Y")
    if d1 !=d2:
        d1 = today.strftime("%d/%m/%Y")
        d2 = today.strftime("%d/%m/%Y")
        total_train() #comp name-->aapl
        break

'''
total_train()