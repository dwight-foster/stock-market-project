
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
def get_html(symbol):
    pd.set_option('display.max_colwidth', 25)

    print('Getting data for ' + symbol + "...\n")

    url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
    req = Request(url, headers={"User-Agent": 'Mozilla/5.0'})
    try:
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        return html
    except:
        return 0
def get_fundamentals(symbol):
    html = get_html(symbol)
    try:
        # Find fundamentals table
        fundamentals = pd.read_html(str(html), attrs={'class': 'snapshot-table2'})[0]

        # Clean up fundamentals dataframe
        fundamentals.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        colOne = []
        colLength = len(fundamentals)
        for k in np.arange(0, colLength, 2):
            colOne.append(fundamentals[f'{k}'])
        attrs = pd.concat(colOne, ignore_index=True)

        colTwo = []
        colLength = len(fundamentals)
        for k in np.arange(1, colLength, 2):
            colTwo.append(fundamentals[f'{k}'])
        vals = pd.concat(colTwo, ignore_index=True)

        fundamentals = pd.DataFrame()
        fundamentals['Attributes'] = attrs
        fundamentals['Values'] = vals
        fundamentals.to_csv('fundamentals.csv')
        return fundamentals, 1

    except:
        return 0, 0


def get_news(symbol):
    html = get_html(symbol)

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


def get_insider(symbol):
    html = get_html(symbol)

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

