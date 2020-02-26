
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import datetime as dt
from datetime import date
import pandas_datareader.data as web


def plot_stock(predicted, y, comp_name):
    plt.figure(figsize=(20, 10))
    plt.plot(predicted, color='red', label='Predicted Stock Price of last few days')
    plt.plot(y, color='blue', label='Actual stock price of last few days')
    plt.title('Stock Price of ' + comp_name)
    plt.xlabel('Time days')
    plt.ylabel('US dollars')
    plt.legend()
    plt.grid(color='black', linestyle='--', linewidth=1)
    plt.show()


def plot_difference(predicted, y, comp_name):
    plt.figure(figsize=(20, 10))
    difference = abs(predicted - y)
    plt.plot(difference, color='black', label="difference")
    plt.title('Stock Price diff of ' + comp_name)
    plt.xlabel('Time days')
    plt.ylabel('US dollars')
    plt.legend()
    plt.grid(color='black', linestyle='--', linewidth=1)
    plt.show()


def plot_relation(predicted, y, compname):
    plt.figure(figsize=(20, 10))
    plt.plot(y, predicted, '-r', label='actual=f(predicted)')
    plt.title('Graph of actual=f(predicted)')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def date_prep():
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    d = d1.split('/')
    day = int(d[0])
    month = int(d[1])
    year = int(d[2])
    year_1 = year - 1
    end = dt.datetime(year, month, day)
    if month == 2 and day == 29 and not (((year_1 % 4 == 0) and (year_1 % 100 != 0)) or (year_1 % 400 == 0)):
        start = dt.datetime(year_1, month, day - 1)
    else:
        start = dt.datetime(year_1, month, day)
    return start, end


def prepare_csv(comp_name):
    start, end = date_prep()
    try:
        df = web.DataReader(comp_name, 'yahoo', start, end)
        df.to_csv(comp_name + '.csv')
    except Exception as e:
        print(e)


def predict_next_day_closing_price():
    comp_name = input("Enter the company name ")
    comp_name = comp_name.upper()
    prepare_csv(comp_name)
    df = pd.read_csv(comp_name + '.csv')
    model = keras.models.load_model('StockPredictorAAPl.model')

    # new part
    df1 = pd.read_csv(comp_name + '.csv')
    closing1 = df1.iloc[-300:, 1:5].values  # closing values
    closing_unchanged1 = closing1[:, 3]
    sc = MinMaxScaler()
    closing1 = sc.fit_transform(closing1)

    span = 60
    jump = 1
    x_10 = []
    y_10 = []
    length = len(closing1)
    for i in range(span, length + 1, jump):
        x_10.append(closing1[i - span:i, ])
        if i < length:
            l = list(closing_unchanged1[i:i + jump, ])
            y_10.extend(l)

    x_10 = np.array(x_10)
    y_10 = np.array(y_10)
    y_10 = np.reshape(y_10, (-1, jump))
    x_10 = np.reshape(x_10, (-1, x_10.shape[1], 4))
    p_new = model.predict(x_10)

    trainingd2 = df1.iloc[-300:, 4:5].values
    k = sc.fit(trainingd2)
    predicted_new = sc.inverse_transform(p_new)

    print("todays actual closing price : " + str([closing_unchanged1[-1]]) +
          "\n next day predicted closing price : " + str(predicted_new[-1]))

    print("last 300 day")

    print(predicted_new[-3:])
    print(y_10[-2:])

    plot_stock(predicted_new, y_10, comp_name)


    os.remove(comp_name + '.csv')


predict_next_day_closing_price()