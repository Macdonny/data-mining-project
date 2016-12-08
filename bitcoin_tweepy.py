import tweepy
from textblob import TextBlob
import csv
import time


CONSUMER_KEY = '2ikoi79lK5IUgmRWEe5aE3O36'
CONSUMER_SECRET = 'f4KursqvuCXk1Kjf8BVKnflSnXR0xC3BORlUzWggd8ExEQGMmg'

ACCESS_TOKEN = '716412406247723009-nb1ClrdbnjPfxPo8og2cGZltOg4ECvw'
ACCESS_TOKEN_SECRET = 'ZdXEKJBXMcY0YTFWkUbJGDT18UxooJcBfHSOTUk1GPSSM'

tweets = []
bitcoin_sentiment = []


def twitter_search(text, limit=20):
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    api = tweepy.API(auth)
    
    # Step 3 - Define a threshold for each sentiment to classify each
    # as positive or negative. If the majority of tweets you've collected are positive
    # then use your neural network to predict a future price
    threshold = 0.2
    pos_tweet = 0
    pos_sentiment_x = []
    neg_tweet = 0
    neg_sentiment_y = []
    
    for tweet in limit_handled(tweepy.Cursor(api.search, q=text, result_type="recent", lang="en").items(limit=limit)):
        
        analysis = TextBlob(tweet.text)
        bitcoin_sentiment.append(analysis.sentiment)
        
        if analysis.sentiment.polarity >= threshold:
            pos_tweet += 1
            polarity_flag = 1
            pos_sentiment_x.append(analysis.sentiment)
        else:
            neg_tweet += 1
            polarity_flag = 0
            neg_sentiment_y.append(analysis.sentiment)
            
        tweets.append({'created_at': tweet.created_at,
                       'polarity': analysis.sentiment.polarity,
                       'subjectivity': analysis.sentiment.subjectivity,
                       'sentiment': polarity_flag,
                       'tweet': tweet.text.encode('utf-8')})

    if pos_tweet > neg_tweet:
        print("Overall Positive")
    else:
        print("Overall Negative")
    
    # write tweets to cvs file
    write_tweets(tweets)
    print(bitcoin_sentiment)
    print(pos_sentiment_x)
    print(neg_sentiment_y)


# Maintains twitter api threshold
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)
            

def write_tweets(tweets_data):
    filename = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Open/Create a file to append data
    with open('assets/data/tweets/' + filename + '.csv', 'w', newline='') as tweets_csv:

        # Use CSV writer
        fieldnames = ['created_at', 'polarity', 'subjectivity', 'sentiment', 'tweet']
        writer = csv.DictWriter(tweets_csv, fieldnames=fieldnames)

        writer.writeheader()
        # Write a row to the csv file
        for tweet in tweets_data:
            writer.writerow(tweet)


def main():
    twitter_search('Bitcoin', 25)


if __name__ == '__main__':
    main()
