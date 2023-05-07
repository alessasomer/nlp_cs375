import tweepy
import config
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer

class Twitter:
    def __init__(self):
        self._vocab2indx = None
        self._embed_array = None
        self._len_idx2vocab = int

    def create_tweetcsv(self, file_name):
        trainingList =[]
        with open(file_name) as file_object:
            for jsonObj in file_object:
                trainingDict = json.loads(jsonObj)
                trainingList.append(trainingDict)

        api_key = 'YEpNeLOhlYEL5OId1bCKenHxD'
        api_key_secret = 'AdWfTqltmr7qKSOkOjujkipCGs08UuMjegYRLqpHkLT6Fv362L'
        BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAOJnAEAAAAAyVKYaq6rfDF7NeHLThQ86hVJibk%3DatFtfDHbOAYzA3IcsLvfECssqqrk8uS4d8967dJFgD7Cs4dlCU'

        access_token = "1651328402245185538-7yv4hOJshKAnLroHWEmq0T47TRdFML"
        access_token_secret = "aWpwR6zzZsdqU6GixgTC8nuLDkr73jcGL83vKNDh259JM"

        #authentication
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(access_token, access_token_secret)

        #call Api
        api = tweepy.API(auth)

        #create dataframe
        columns = ['Label', 'Tweet']
        data = []
        for t in trainingList:
            try:
                status = api.get_status(t["id"])

                #get text
                text = status.text
                #print(text)
                data.append([t["label"], text])
            except:
                #print("Tweet with ID" , t["id"] , "does not exist")
                continue
        df = pd.DataFrame(data, columns=columns)
        label_examples = df['Label'].tolist()
        tweet_examples = df['Tweet'].tolist()
        #df.to_csv('tweets.csv')
        #NOTE: here tweet_examples has not been tokenized yet, and is a list of tweet texts NOT a list of words
        return tweet_examples, label_examples

    def create_tokens(self, tweet_examples)-> list[list[str]]:
        """
        Takes in a list of tweets (str) and uses the nltk tweet tozenizer to tokenize each text string 
        RETURNS: A list of lists, where the inner list are the string tokens from one tweet
        """
        tknzr = TweetTokenizer()
        for t in tweet_examples:
            tweet_examples[tweet_examples(t)] = tknzr.tokenize(t)
        return tweet_examples

    def create_embeddings(self) -> np.ndarray:
        """
        Create emebeddings and add OOV (initialze as "the) to vocabulary, and pad + truncate embeddings 
        """
        embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)
        #vocab2indx is a dictionary where keys are words in vocab
            #values are the "index" of a word- an int correspinding to the row index for the embedding in embed_array
        self._vocab2indx = dict(embeddings.key_to_index)
        idx2vocab = list(embeddings.index_to_key)
        self._len_idx2vocab = len(idx2vocab)
        self._embed_array = embeddings.vectors # matrix of dense word embeddings 
                                    # rows: a word 
                                    # columns: dimensions (50) of the dense embeddings
        # Add <OOV> symbol 
        new_oov_entry = len(embeddings)
        idx2vocab += ["<OOV>"]
        self._vocab2indx["<OOV>"] = new_oov_entry
        self._embed_array = self.add_the_embedding()
        # Add <PAD> symbol (also as embedding for the word type "the")
        new_pad_entry = len(idx2vocab)
        idx2vocab += ["<PAD>"]
        self._vocab2indx["<PAD>"] = new_pad_entry
        self._embed_array = self.add_the_embedding()
        return self._embed_array
    
    def add_the_embedding(self, embedding_dim=50): 
        """
        Adds "the" embedding to the embed_array matrix
        """
        the_embedding = self._embed_array[self._vocab2indx["the"]]
        out = np.vstack((self._embed_array, the_embedding))
        return out

    def create_X_train(self, tweet_examples) -> list[int]: 
        """
        For each example, translate each token into its corresponding index from vocab2indx 
                           
        Returns: 
        - (List[int]): list of integers
        """ 
        MAXIMUM_LENGTH = 100
        indexes = []
        X_list = []
        for example in tweet_examples:
            for token in example:
                if self._vocab2indx.get(token) is not None:
                    indexes.append(self._vocab2indx[token])
                else:
                    indexes.append(self._vocab2indx['<OOV>'])
            #Truncate
            indexes = indexes[:MAXIMUM_LENGTH]
            #pad
            while len(indexes) < MAXIMUM_LENGTH:
                indexes.append(len(self._len_idx2vocab))
        return indexes

#call deepaveragingmodel 
def main():
    twitter = Twitter()
    test_data = twitter.create_tweetcsv('training.txt')
    print(test_data[0])
    print(type(test_data[0]))
    #tweet_examples_test = twitter.create_tokens(test_data[0])
    #tweet_examples_dev , label_examples_dev = twitter.create_tweetcsv('dev.txt')
    embedding_array = twitter.create_embeddings()
    #hey = twitter.create_X_train(tweet_examples_test)
    #print(hey[0:5])

if __name__ == '__main__':
    main()
