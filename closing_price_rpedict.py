# importing valuable libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import datetime as dt
from datetime import date
import pandas_datareader.data as web
max_obj=3000
obj_counter=0
class Predict_Stock:
    def __init__(self,id):
        self.id=id

    def date_prep(self):
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

    def prepare_csv(self,comp_name):
        start, end = self.date_prep()
        try:
            df = web.DataReader(comp_name, 'yahoo', start, end)
            df.to_csv(comp_name + '.csv')
        except Exception as e:
            print(e)

    def plot_stock(self,jump, predicted, comp_name):
        plt.figure(figsize=(20, 10))
        plt.plot(predicted, color='red', label='Predicted Stock Price for next ' + str(jump) + ' days')
        plt.title('Stock Price of ' + comp_name)
        plt.xlabel('Time days')
        plt.ylabel('US dollars')
        plt.legend()
        plt.grid(color='black', linestyle='--', linewidth=1)
        plt.show()

    def predict_next_five_closing_price(self):
        comp_name = input("Enter the company name ")
        comp_name = comp_name.upper()
        self.prepare_csv(comp_name)
        df = pd.read_csv(comp_name + '.csv')
        model = keras.models.load_model('StockPredictorAAPl.model')
        closing = df.iloc[-60:, 4:5].values  # closing values
        closing_unchanged = closing
        sc = MinMaxScaler()
        closing = sc.fit_transform(closing)
        span = 60
        jump = 5
        length = len(closing)
        x = []
        y = []
        y_p = []
        x1 = []
        i = span
        x = np.append(x, closing[i - span:i, ])
        x1 = np.append(x1, closing[i - span:i, ])
        for j in range(jump):
            x2 = []
            x2 = np.append(x2, x1[-span:, ])
            x2 = x2.reshape(1, span, 1)
            p = model.predict(x2)
            y_p = np.append(y_p, p)
            x1 = np.append(x1, p)
        predicted = sc.inverse_transform([y_p])
        y = np.array(closing_unchanged)
        y = y.reshape(-1, )
        predicted = np.array(predicted)
        predicted = predicted.reshape(-1, )
        print(predicted)
        self.plot_stock(jump, predicted, comp_name)
        os.remove(comp_name + '.csv')

    def predict_next_ten_closing_price(self):
        comp_name = input("Enter the company name ")
        comp_name = comp_name.upper()
        self.prepare_csv(comp_name)
        df = pd.read_csv(comp_name + '.csv')
        model = keras.models.load_model('StockPredictorAAPl.model')
        closing = df.iloc[-60:, 4:5].values  # closing values
        closing_unchanged = closing
        sc = MinMaxScaler()
        closing = sc.fit_transform(closing)
        span = 60
        jump = 10
        length = len(closing)
        x = []
        y = []
        y_p = []
        x1 = []
        i = span
        x = np.append(x, closing[i - span:i, ])
        x1 = np.append(x1, closing[i - span:i, ])
        for j in range(jump):
            x2 = []
            x2 = np.append(x2, x1[-span:, ])
            x2 = x2.reshape(1, span, 1)
            p = model.predict(x2)
            y_p = np.append(y_p, p)
            x1 = np.append(x1, p)
        predicted = sc.inverse_transform([y_p])
        y = np.array(closing_unchanged)
        y = y.reshape(-1, )
        predicted = np.array(predicted)
        predicted = predicted.reshape(-1, )
        print(predicted)
        self.plot_stock(jump, predicted, comp_name)
        os.remove(comp_name + '.csv')

    def predict_next_day_closing_price(self):
        comp_name = input("Enter the company name ")
        comp_name = comp_name.upper()
        self.prepare_csv(comp_name)
        df = pd.read_csv(comp_name + '.csv')
        model = keras.models.load_model('StockPredictorAAPl.model')
        closing = df.iloc[-60:, 4:5].values  # closing values
        closing_unchanged = closing
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
        x = np.append(x, closing[i - span:i, ])
        x1 = np.append(x1, closing[i - span:i, ])
        for j in range(jump):
            x2 = []
            x2 = np.append(x2, x1[-span:, ])
            x2 = x2.reshape(1, span, 1)
            p = model.predict(x2)
            y_p = np.append(y_p, p)
            x1 = np.append(x1, p)
        predicted = sc.inverse_transform([y_p])
        predicted = np.array(predicted)
        predicted = predicted.reshape(-1, )
        print("Today's actual closing price : " + str(closing_unchanged[-1]))
        print("next day predicted closing price : " + str(predicted[0]))
        os.remove(comp_name + '.csv')

    def switch_function(self):
        while True:
            option=int(input('0,1,2,3 for 1 day 5 days 10 days prediction and STOP respectively : '))
            if option==0:
                self.predict_next_day_closing_price()
            elif option==1:
                self.predict_next_five_closing_price()
            elif option==2:
                self.predict_next_ten_closing_price()
            else:
                break

#predict_next_day_closing_price('GOOG')

def create_obj():
    global obj_counter
    global max_obj
    obj_counter += 1
    if obj_counter <=max_obj:
        serve_obj=Predict_Stock(id=id)
        serve_obj.switch_function()
        if obj_counter > 0:
            obj_counter -=1
        else:
            return
    else:
        print("Max server reached !")
        return

create_obj()