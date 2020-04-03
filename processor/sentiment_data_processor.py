import os
import re
import nltk
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from tqdm import tqdm

pjoin = os.path.join

nltk.data.path.append(os.path.abspath('./data/nltk_data'))

class SentimentDataProcessor:
    def __init__(self, csv_file, num_tweets='all'):
        '''Initializer function for the processor. Reads the CSV file into a processed dataframe.'''
        csv_file = os.path.abspath(csv_file)
        # Load stopwords from NLTK library
        self.stopwords = stopwords.words('english')
        if num_tweets == 'all':
            self.df = pd.read_csv(csv_file, error_bad_lines=False)
        else:
            self.df = pd.read_csv(csv_file, error_bad_lines=False, nrows=num_tweets)
        print(f'Read {num_tweets} tweets')
        # Store tweet texts and sentiments
        self.tweets = self.df['Text']
        self.sentiments = self.df['Sentiment']
        self.num_tweets = len(self.tweets)

    def _prepocess(self):
        '''Preprocessor for tweet text.'''
        self.processed_tweets = []
        for tweet in tqdm(self.tweets):
            # Set text to lowercase
            tweet = tweet.lower()
            # Remove numbers
            tweet = re.sub(r'\d+', '', tweet)
            # Remove whitespaces
            tweet = tweet.strip()
            # Remove links
            tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
            # Remove usernames
            tweet = re.sub(r'@[^\s]+', '', tweet) 
            processed_tweet = simple_preprocess(tweet, min_len=3)
            # Remove stopwords
            stopword_mask = lambda word: word not in self.stopwords 
            processed_tweet = list(filter(stopword_mask, processed_tweet))
            processed_tweet = ' '.join(processed_tweet) 
            self.processed_tweets.append(tweet)

    def get_processed_df(self):
        '''Get processed dataframe for analysis.'''
        self._prepocess()
        processed_df = pd.DataFrame(
            {
                'Sentiment' : self.sentiments,
                'Tweet' : self.processed_tweets,
            }
        )

        return processed_df