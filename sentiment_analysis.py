# Import the pandas package, then use the "read_csv" function
# the labeled training data
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from nltk.corpus import stopwords       # Import the stop word list
import nltk
# nltk.download()                         # Download text data sets, including stop words


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result
    return " ".join(meaningful_words)


def main():
    train = pd.read_csv("labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)
    train.shape
    train.columns.values
    
    # clean_review = review_to_words(train["review"][0])
    # print(clean_review)
    
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size
    
    # Initialize an empty list to hold the clean reviews
    print("Cleaning and parsing the training set movie reviews...\n")
    clean_train_reviews = []
    
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    for i in range(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if (i+1) % 1000 == 0:
            print("Review {} of {}\n".format(i+1, num_reviews))

        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_train_reviews.append(review_to_words(train["review"][i]))
    
    print("Creating the bag of words...\n")
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
    
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    
    # Numpy arrays are easy to work with, so convert the results to an
    # array
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)
    
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print(vocab)
    
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print(count, tag)
    
    print("Training the random forest...")
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)
    
    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit(train_data_features, train["sentiment"])
    
    # Read the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t",
                       quoting=3)
    
    # Verify that there are 25,000 rows and 2 columns
    print(test.shape)
    
    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = []
    
    print("Cleaning and parsing the test set movie reviews... \n")
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print("Review {} of {}\n".format(i+1, num_reviews))
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)
        
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)
    
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    
    # Use pandas to write the comma-separated output file
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
    
    # Initialize the BeautifulSoup object on a single movie review
    # example1 = BeautifulSoup(train["review"][0], "lxml")
    
    # Print the raw review and then the output of get_text(), for
    # comparison
    # print(train["review"][0])
    # print(example1.get_text())
    
    # Use regular expressions to do a find-and-replace
    # letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
    #                       " ",                   # The pattern to replace it with
                          # example1.get_text())   # The text to search
    # print(letters_only)
    
    # lower_case = letters_only.lower()   # Convert to lower case
    # words = lower_case.split()          # Split into words
    
    # print(stopwords.words("english"))
    
    # Remove stop words from "words"
    # words = [w for w in words if not w in stopwords.words("english")]
    # print(words)


if __name__ == '__main__':
    main()
