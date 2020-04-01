import os
import re
import nltk
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from collections import Counter

pjoin = os.path.join

nltk.data.path.append(os.path.abspath('./data/nltk_data'))

class TweetProcessor:
    def __init__(self, csv_file):
        '''Initializer function for processor. Reads the CSV file and creates a dataframe ready for preprocessing.'''
        # Read the CSV file, construct a dataframe
        csv_file = os.path.abspath(csv_file)
        self.df = pd.read_csv(csv_file, delimiter=';', parse_dates=[1], error_bad_lines=False)
        # Drop the "ID" and "geo" columns (unused)
        del self.df['id']
        del self.df['geo']

        self._get_date_and_time()
    
        self.tweets = self.df['text']
        self.num_tweets = len(self.tweets)

    def _get_date_and_time(self):
        '''From the datetime stamps in the original dataframe, get date and time separately.'''
        self.df.rename(columns={'date' : 'DateTime'}, inplace=True)
        self.df['Date'] = self.df['DateTime'].dt.date
        self.df['Time'] = self.df['DateTime'].dt.time
        self.df['Hour'] = self.df['DateTime'].dt.hour
        
        # Set the date as index
        self.df.set_index('Date', inplace=True)
    
    def _preprocess(self):
        '''Pre-process words in tweets. Returns a list of words for each tweet, passed through Gensim preprocessor.'''
        self.processed_tweets = []

        def remove_links(tweet):
            '''Remove links in tweet text as a part of pre-processing.'''
            # Regular expressions to be matched with links
            to_be_matched = [
                'pic.twitter.com/.*',
                'imza.la.*http://.*fb.me/.*',
                'http.*'
            ]

            for exp in to_be_matched:
                matches = re.findall(exp, tweet)
                for match in matches:
                    tweet = tweet.replace(match, '')
            
            return tweet

        for idx, tweet in enumerate(self.tweets):
            print(f'Processing tweet: {idx}/{self.num_tweets}', end='\r')
            tweet = remove_links(tweet)
            processed_tweet = simple_preprocess(tweet, min_len=3)
            self.processed_tweets.append(processed_tweet)

        print(f'Processing tweet: {idx+1}/{self.num_tweets}')

    def get_most_common_words(self, extra_words_to_remove, remove_stopwords=True, num_words=10):
        '''Count number of words and get the most common words out of the given tweets.'''
        self.counter = Counter()

        # Do the pre-processing
        self._preprocess()

        # Count number of words from processed tweets
        for tweet in self.processed_tweets:
            self.counter.update(tweet)
        
        # Remove stopwords, taken from NLTK library
        if remove_stopwords:
            for word in nltk.corpus.stopwords.words('english'):
                self.counter[word] = 0

        for word in extra_words_to_remove:
            self.counter[word] = 0

        # Return as a numpy array data type
        return np.array(self.counter.most_common(num_words))




