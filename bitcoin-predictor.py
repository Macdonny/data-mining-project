import tweepy
from textblob import TextBlob
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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


# Twitter authentication and search
def twitter_search(text):
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    api = tweepy.API(auth)
    
    # Step 2 - Search for your text on Twitter
    public_tweets = api.search(text)

    # Step 3 - Define a threshold for each sentiment to classify each
    # as positive or negative. If the majority of tweets you've collected are positive
    # then use your neural network to predict a future price
    try:
        for tweet in public_tweets:
            analysis = TextBlob(tweet.text)
            print(analysis.sentiment)
    except UnicodeEncodeError as uee:
        print(uee)


# data collection
def get_data(filename):
    df = pd.read_csv(filename, header=0, delimiter=",", quoting=3)
    # print(df)
    # print(df.columns)
    # print(df['Date'])
    # print(df['Close'])

    for row in df['Date']:
        dates.append(int(row.split('/')[1]))

    for row in df['Close']:
        prices.append(float(row))

    print('Received all of the data!')
    return


# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


# Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price
# on a given day. We'll later print the price out to terminal.
def predict_prices(in_dates, in_prices, x):
    in_dates = np.reshape(in_dates, (len(in_dates), 1))
    in_prices = np.reshape(in_prices, (-1, 1))
    print(in_prices)
    
    # print(in_dates, sep='\n')
    # print(in_prices, sep='\n')
    
    # Normalize the price data set
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_dataset = scaler.fit_transform(in_prices)
    print(prices_dataset)
    
    # Split into train and test sets
    train_size = int(len(prices_dataset) * 0.67)
    test_size = len(prices_dataset) - train_size
    train, test = prices_dataset[0:train_size, :], prices_dataset[train_size:len(prices_dataset), :]
    print(len(train), len(test))
    print('train \n{} \ntest \n{}'.format(train, test))
    
    # Reshape into X=t and Y=t+1
    look_back = 1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    
    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, nb_epoch=100, batch_size=1, verbose=2)
    
    # print('support vector regression linear')
    # svr_lin = SVR(kernel='linear', C=1e3)   # C=1e3 Scientific Notation for 1000
    # print('support vector regression ploy')
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2, cache_size=500)
    # print('support vector regression rbf')
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # print('svr linear fit')
    # svr_lin.fit(in_dates, in_prices)
    # print('svr poly fit')
    # svr_poly.fit(in_dates, in_prices)
    print('svr rbf fit')
    # svr_rbf.fit(in_dates, in_prices)

    # plt.scatter(in_dates, in_prices, color='black', label='Data')
    # plt.plot(in_dates, svr_lin.predict(in_dates), color='green', label='Linear model')
    # plt.plot(in_dates, svr_poly.predict(in_dates), color='blue', label='Polynomial model')
    # plt.plot(in_dates, svr_rbf.predict(in_dates), color='red', label='RBF model')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('Support Vector Regression')
    # plt.legend()
    # plt.show()

    # return svr_lin.predict(x)[0], svr_rbf.predict(x)[0]
    return


def main():
    # twitter_search('Bitcoin')

    # Step 5 reference your CSV file here
    get_data('coindesk-bpi-USD-close2.csv')

    # predicted_price = predict_prices(dates, prices, 29)
    # print(predicted_price)
    
    predict_prices(dates, prices, 29)

    # load dataset
    # dataframe = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
    # dataset = dataframe.values
    # split into input (X) and output (Y) variables
    # X = dataset[:, 0:13]
    # Y = dataset[:, 13]
    # print('X{} Y{}'.format(X, Y))

if __name__ == '__main__':
    main()
