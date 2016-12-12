# --------------- Bitcoin Price & Twitter Sentiment Analysis ----------------------

This project is to predict Bitcoin stock prices based on Twitter tweets using sentiment analysis and Bitcoin historical stock prices.

The main project directory consist of assets such as:

. bitcoin_stock directory - CSV Bitcoin stock price files

. Data directory
	-training_results - Data results of the test we did with our Bitcoin predictor program.
	-tweets - A collection of csv files with all of the tweets we've pulled from Twitter using Tweepy.

. img - Graphs of Bitcoin predictions. X is the price of Bitcoin. Y is the price of Bitcoin at the next time step(t + 1).

.bitcoin_predictor - The program used to get tweets from Twitter store them and predict Bitcoin stock prices based on the CSV file used. To test the program make sure all the dependencies are installed locally and run it. The output will show up in your terminal or IDE.
