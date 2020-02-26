
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates


style.use('ggplot')
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()
df = web.DataReader('TSLA', 'yahoo', start, end)
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)
df.to_csv('tsla.csv')

df= pd.read_csv('tsla.csv',parse_dates=True,index_col=0)

df['Adj Close'].plot(figsize=(20, 10))
plt.show()

#100 moving average
df['100ma']=df['Adj Close'].rolling(window=100,min_periods=0).mean()

#matplotlib.pyplot.subplot2grid(shape of grid, loc, rowspan=1, colspan=1, fig=None, **kwargs) !!!!
plt.figure(figsize=(20, 10))
ax1=plt.subplot2grid((6,1),(0,0),rowspan=5, colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=5, colspan=1,sharex=ax1)#sharing the x axis of ax1
ax1.plot(df.index,df['Adj Close'])
ax1.plot(df.index,df['100ma'])
ax2.bar(df.index,df['Volume'])
plt.show()

df_ohlc=df['Adj Close'].resample('10D').ohlc()
df_volume=df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)

print(df_ohlc.head())

plt.figure(figsize=(20, 10))
ax1=plt.subplot2grid((6,1),(0,0),rowspan=5, colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=5, colspan=1,sharex=ax1)#sharing the x axis of ax1
ax1.xaxis_date()

  #h
  #c
#fill##
  #o
  #l
candlestick_ohlc(ax1,df_ohlc.values,width=2,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
plt.show()