import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

pd.set_option('display.max_colwidth', 25)

symbol = input("Enter a ticker: ")
print ('Getting data for ' + symbol + "...\n")

url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
req = Request(url, headers={"User-Agent": 'Mozilla/5.0'})
webpage = urlopen(req).read()
html = soup(webpage, "html.parser")


def get_fundamentals():
    try:
        # Find fundamentals table
        fundamentals = pd.read_html(str(html), attrs={'class': 'snapshot-table2'})[0]
        fundamentals = fundamentals[fundamentals.columns[4:6]]
        # Clean up fundamentals dataframe
        fundamentals.columns = ['4', '5',]
        colOne = []
        colLength = len(fundamentals)
        for k in np.arange(int(fundamentals.columns[0]), colLength, 2):
            if k > int(fundamentals.columns[-1]):
              break
            colOne.append(fundamentals[f'{k}'])
        attrs = pd.concat(colOne, ignore_index=True)

        colTwo = []
        colLength = len(fundamentals)
        for k in np.arange(int(fundamentals.columns[0])+1, colLength, 2):
            if k > int(fundamentals.columns[-1]):
              break
            colTwo.append(fundamentals[f'{k}'])
        vals = pd.concat(colTwo, ignore_index=True)

        fundamentals = pd.DataFrame()
        fundamentals['Attributes'] = attrs
        fundamentals['Values'] = vals
        fundamentals = fundamentals.set_index('Attributes')
        fundamentals.to_csv('fundamentals.csv')
        return fundamentals

    except Exception as e:
        return e


def get_news():
    try:
        # Find news table
        news = pd.read_html(str(html), attrs={'class': 'fullview-news-outer'})[0]
        links = []
        for a in html.find_all('a', class_="tab-link-news"):
            links.append(a['href'])

        # Clean up news dataframe
        news.columns = ['Date', 'News Headline']
        news['Article Link'] = links
        news = news.set_index('Date')
        return news

    except Exception as e:
        return e


def get_insider():
    try:
        # Find insider table
        insider = pd.read_html(str(html), attrs={'class': 'body-table'})[0]

        # Clean up insider dataframe
        insider = insider.iloc[1:]
        insider.columns = ['Trader', 'Relationship', 'Date', 'Transaction', 'Cost', '# Shares', 'Value ($)',
                           '# Shares Total', 'SEC Form 4']
        insider = insider[
            ['Date', 'Trader', 'Relationship', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total',
             'SEC Form 4']]
        insider = insider.set_index('Date')
        return insider

    except Exception as e:
        return e

print(get_fundamentals())
