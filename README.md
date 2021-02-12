# This is a simple DQN based trading bot I created. 
### WARNING USE AT YOUR OWN RISK. I DO NOT TAKE RESPONSIBILITY FOR ANY MONEY LOST USING THIS BOT. THIS IS JUST A STARTER BOT. TRADE AT YOUR OWN RISK
It is not that great but it is a cool starter program if you are trying to do something like this. It uses a on policy dqn to make trading decisions.
It is passed data scraped from finviz.com. WARNING ALL DATA FROM FINVIZ IS DELAYED BY 15 MINUTES. The model is a combination of an lstm and a linear layer.
The LSTM is passed the price data for the stock while the linear layers are passed information about the company and the amount of money it has. 
