#!/bin/bash

# Install the stopwords dataset if not installed already
if [ ! -d "./data/nltk_data" ]
then
    python -m nltk.downloader -d ./nltk_data stopwords
else
    echo "./data/nltk_data directory already exists, moving on!"
fi

# Set the env variable so that NLTK can find the data
export NLTK_DATA="${PWD}/data/nltk_data"