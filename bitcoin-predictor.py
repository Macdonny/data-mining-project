import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tweepy
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

# Step 1 - Insert your API keys
CONSUMER_KEY = '2ikoi79lK5IUgmRWEe5aE3O36'
CONSUMER_SECRET = 'f4KursqvuCXk1Kjf8BVKnflSnXR0xC3BORlUzWggd8ExEQGMmg'

ACCESS_TOKEN = '716412406247723009-nb1ClrdbnjPfxPo8og2cGZltOg4ECvw'
ACCESS_TOKEN_SECRET = 'ZdXEKJBXMcY0YTFWkUbJGDT18UxooJcBfHSOTUk1GPSSM'

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dates = []
prices = []
bitcoin_sentiment = []


# Twitter authentication and search
def twitter_search(text):
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    api = tweepy.API(auth)
    
    # Step 2 - Search for your text on Twitter
    public_tweets = api.search(text)
    print(public_tweets[2].text)

    for tweet in tweepy.Cursor(api.search,
                               q=text,
                               rpp=100,
                               count=20,
                               result_type="recent",
                               include_entities=True,
                               lang="en").items(200):
        print(tweet)
    
    # Step 3 - Define a threshold for each sentiment to classify each
    # as positive or negative. If the majority of tweets you've collected are positive
    # then use your neural network to predict a future price
    threshold = 0.5
    pos_tweet = 0
    pos_sentiment = ()
    neg_tweet = 0
    neg_sentiment = ()

    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        bitcoin_sentiment.append(analysis.sentiment)
        print(analysis.sentiment)
        if analysis.sentiment.polarity >= threshold:
            pos_tweet += 1
        else:
            neg_tweet += 1
    if pos_tweet > neg_tweet:
        print("Overall Positive")
    else:
        print("Overall Negative")


# data collection
def get_data(filename):
    df = pd.read_csv(filename, header=0, delimiter=",", quoting=3)

    for row in df['Date']:
        dates.append(int(row.split('/')[1]))

    for row in df['Close']:
        prices.append(float(row))

    print('Received all of the data!')
    return


# Convert an array of values into a dataset matrix
def create_datasets(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        data_x.append(dataset[i:(i+look_back), 0])
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


# Built Long Short-Term Memory Network model using Keras for regreesion, trained it, then have it predict the price
# on a given day. We'll later print the price out to terminal.
def predict_prices(in_dates, in_prices):
    # in_dates = np.reshape(in_dates, (len(in_dates), 1))
    print(in_prices)
    
    # Normalize the price data set
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_dataset = scaler.fit_transform(in_prices)
    
    # Split into train and test sets
    train_size = int(len(prices_dataset) * 0.67)    # 20
    test_size = len(prices_dataset) - train_size    # 11
    
    train, test = prices_dataset[0:train_size, :], prices_dataset[train_size:len(prices_dataset), :]
    
    # Reshape into X=t and Y=t+1
    look_back = 3
    train_x, train_y = create_datasets(train, look_back)
    test_x, test_y = create_datasets(test, look_back)
    
    print('Before Reshape\n')
    print(train_x.shape)
    print(test_x.shape)
    
    # Reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    
    print('After Reshape\n')
    print(train_x.shape)
    print(train_x)
    print(test_x.shape)
    print(test_x)
    
    # Create and fit the LSTM network
    print('Build model...')
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(100):
        model.fit(train_x, train_y, nb_epoch=100, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    
    # Make predictions
    train_prediction = model.predict(train_x, batch_size=batch_size)
    model.reset_states()
    test_prediction = model.predict(test_x, batch_size=batch_size)
    
    # Invert predictions
    train_prediction = scaler.inverse_transform(train_prediction)
    train_y = scaler.inverse_transform([train_y])
    
    test_prediction = scaler.inverse_transform(test_prediction)
    test_y = scaler.inverse_transform([test_y])
    
    # Calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(train_y[0], train_prediction[:, 0]))
    print('Train Score: {:10.4f} RMSE'.format(train_score))
    
    test_score = math.sqrt(mean_squared_error(test_y[0], test_prediction[:, 0]))
    print('Test Score: {:10.4f} RMSE'.format(test_score))
    
    # Shift train predictions for plotting
    train_prediction_plot = np.empty_like(prices_dataset)
    train_prediction_plot[:, :] = np.nan
    train_prediction_plot[look_back:len(train_prediction)+look_back, :] = train_prediction
    
    # Shift test predictions for plotting
    test_prediction_plot = np.empty_like(prices_dataset)
    test_prediction_plot[:, :] = np.nan
    test_prediction_plot[len(train_prediction) + (look_back * 2) + 1: len(prices_dataset) - 1, :] = test_prediction
    
    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(prices_dataset))
    plt.plot(train_prediction_plot)
    plt.plot(test_prediction_plot)
    plt.show()
    
    return train_score, test_score


def main():
    # twitter_search('Bitcoin')

    # Reference CSV file here
    # get_data('coindesk-bpi-USD-close2.csv')

    # plt.plot(prices)
    # plt.show()
    
    # predict_prices(dates, prices)

    # load the dataset
    dataframe = pd.read_csv('coindesk-bpi-USD-close2.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    predict_prices(dates, dataset)
    
if __name__ == '__main__':
    main()
