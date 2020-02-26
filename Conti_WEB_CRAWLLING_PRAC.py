import bs4
from bs4 import BeautifulSoup
import requests
company_names=['GOOG','FB','RELIANCE.NS','SBI','TCS','AAPL','WIPRO.NS']

def get_stock_info(company_name):
    while True:
        m = []
        r=requests.get('https://in.finance.yahoo.com/quote/'+company_name+'/')
        soup=BeautifulSoup(r.text,"lxml")
        l=soup.find('div',{'class':'My(6px) Pos(r) smartphone_Mt(6px)'}).find_all('span') #The outermost division
        m.append(l[0].text)
        m.append(l[1].text)
        print(company_name+' : '+m[0],m[1])

get_stock_info(company_names[1])

