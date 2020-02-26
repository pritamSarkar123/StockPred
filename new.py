import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import datetime as dt
from datetime import date
import pandas_datareader.data as web

def plot_stock(predicted, y,comp_name):
    plt.figure(figsize=(20, 10))
    plt.plot(predicted, color='red', label='Predicted Stock Price of last few days')
    plt.plot(y,color='blue',label='Actual stock price of last few days')
    plt.title('Stock Price of ' + comp_name)
    plt.xlabel('Time days')
    plt.ylabel('US dollars')
    plt.legend()
    plt.grid(color='black', linestyle='--', linewidth=1)
    plt.show()

def plot_difference(predicted, y,comp_name):
    plt.figure(figsize=(20, 10))
    difference = abs(predicted - y)
    plt.plot(difference, color='black', label="difference")
    plt.title('Stock Price diff of ' + comp_name)
    plt.xlabel('Time days')
    plt.ylabel('US dollars')
    plt.legend()
    plt.grid(color='black', linestyle='--', linewidth=1)
    plt.show()

def plot_relation(predicted,y,compname):
    plt.figure(figsize=(20,10))
    plt.plot(y,predicted,'-r', label='actual=f(predicted)')
    plt.title('Graph of actual=f(predicted) of '+compname)
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

    closing = df.iloc[-60:, 1:5].values  # closing values
    closing_unchanged = closing[:,3]
    sc = MinMaxScaler()
    closing = sc.fit_transform(closing)

    span = 60
    jump = 1
    length = len(closing)
    x = []
    y = []
    y_p = []
    x1 = []
    i = span
    x2=closing
    x2=np.array(x2)
    x2 = x2.reshape(-1, span, 4)
    p = model.predict(x2)

    trainingd1 = df.iloc[-60:, 4:5].values
    k = sc.fit(trainingd1)
    predicted = sc.inverse_transform(p)


    print("todays actual closing price : " + str([closing_unchanged[-1]])+
          "\n next day predicted closing price : " + str(predicted[0]))

    #new part
    df1 = pd.read_csv(comp_name + '.csv')
    closing1 = df1.iloc[-100:, 1:5].values  # closing values
    closing_unchanged1 = closing1[:,3]
    sc = MinMaxScaler()
    closing1 = sc.fit_transform(closing1)

    x_10=[]
    y_10=[]
    length = len(closing1)
    for i in range(span, length,jump):
        if i + jump < length:
            x_10.append(closing1[i - span:i, ])
            l = list(closing_unchanged1[i:i + jump,])
            y_10.extend(l)
        else:
            break
    x_10 = np.array(x_10)
    y_10 = np.array(y_10)
    y_10 = np.reshape(y_10, (-1, jump))
    x_10 = np.reshape(x_10, (-1, x_10.shape[1], 4))
    p_new=model.predict(x_10)

    trainingd2 = df1.iloc[-100:, 4:5].values
    k = sc.fit(trainingd2)
    predicted_new=sc.inverse_transform(p_new)

    print("last 30 day")
    count = 0
    count_w = 0
    for bal in range(30,1,-1):

        #print("predicted : " + str(predicted_new[-bal+1]))
        #print("actual : " + str(y_10[-bal]))
        if predicted_new[-bal+1]>predicted_new[-bal]:
            if y_10[-bal+1]>y_10[-bal]:
                #print("p in a in")
                count+=1
            elif y_10[-bal+1]==y_10[-bal]:
                #print("p in a unchanged")
                count_w += 1
            else:
                #print("p in a dec")
                count_w += 1
        elif predicted_new[-bal+1]==predicted_new[-bal]:
            if y_10[-bal+1]>y_10[-bal]:
                #print("p unchanged a in")
                count_w += 1
            elif y_10[-bal+1]==y_10[-bal]:
                #print("p unchanged a unchanged")
                count += 1
            else:
                #print("p unchanged a dec")
                count_w += 1
        else:
            if y_10[-bal+1]>y_10[-bal]:
                #print("p dec a in")
                count_w += 1
            elif y_10[-bal+1]==y_10[-bal]:
                #print("p dec a unchanged")
                count_w += 1
            else:
                #print("p dec a dec")
                count += 1

    #print(count,count_w)
    correct=(float(count/30)*100)
    incorrect=(float(count_w/30)*100)
    print("correct : "+str(correct)+"% incorrect : "+str(incorrect)+"% ")
    plot_stock(predicted_new,y_10,comp_name)
    plot_difference(predicted_new,y_10,comp_name)
    plot_relation(predicted_new,y_10,comp_name)


    os.remove(comp_name + '.csv')

predict_next_day_closing_price()