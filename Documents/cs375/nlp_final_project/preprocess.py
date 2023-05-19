import tweepy
import config
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
import pickle


class Twitter:
   def __init__(self):
       self._vocab2indx = dict
       self._embed_array = None
       self._len_idx2vocab = int
       self._train_data = ([], [])
       self._dev_data = ([], [])
       self._pred_data = ([], [])
       self._X_train_list = []
       self._X_dev_list = []


   def create_tweetcsv_test(self):
       trainingList =[]
       with open('coronavirus_train.txt') as file_object:
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
               data.append([t["label"], text])
           except:
               #print("Tweet with ID" , t["id"] , "does not exist")
               continue
       df = pd.DataFrame(data, columns=columns)
       test_label_examples = df['Label'].tolist()
       test_tweet_examples = df['Tweet'].tolist()
       #NOTE: here tweet_examples has not been tokenized yet, and is a list of tweet texts NOT a list of words
       return test_tweet_examples, test_label_examples
   
   def create_tweetcsv_dev(self):
       trainingList =[]
       with open('coronavirus_dev.txt') as file_object:
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
       dev_label_examples = df['Label'].tolist()
       dev_tweet_examples = df['Tweet'].tolist()
       #df.to_csv('tweets.csv')
       #NOTE: here tweet_examples has not been tokenized yet, and is a list of tweet texts NOT a list of words
       return dev_tweet_examples, dev_label_examples

   def create_tweets_predict(self, tweet_examples)-> list[list[str]]:
        trainingList =[]
        with open('covid_dev.txt') as file_object:
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
        pred_label_examples = df['Label'].tolist()
        pred_tweet_examples = df['Tweet'].tolist()
        #df.to_csv('tweets.csv')
        #NOTE: here tweet_examples has not been tokenized yet, and is a list of tweet texts NOT a list of words
        return pred_tweet_examples, pred_label_examples

   def create_tokens(self, tweet_examples)-> list[list[str]]:
       """
       Takes in a list of tweets (str) and uses the nltk tweet tozenizer to tokenize each text string
       RETURNS: A list of lists, where the inner list are the string tokens from one tweet
       """
       tokenized_tweet_examples = [[]]
       tknzr = TweetTokenizer()
       for t in tweet_examples:
           tokenized_tweet_examples.append(tknzr.tokenize(t))
       return tokenized_tweet_examples[1:]


   def create_embeddings(self) -> np.ndarray:
       """
       Create emebeddings and add OOV (initialze as "the) to vocabulary, and pad + truncate embeddings
       """
       embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove.twitter.27B.50d.txt", binary=False)
       #vocab2indx is a dictionary where keys are words in vocab
           #values are the "index" of a word- an int correspinding to the row index for the embedding in embed_array
       self._vocab2indx = dict(embeddings.key_to_index)
       if self._vocab2indx is not None:
           print("vocab not none")
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
       the = "the"
       the_embedding = self._embed_array[self._vocab2indx[the]]
       out = np.vstack((self._embed_array, the_embedding))
       return out


   def create_train(self, tweet_examples) -> list[int]:
       """
       For each example, translate each token into its corresponding index from vocab2indx
                         
       Returns:
       - (List[int]): list of integers
       """
       MAXIMUM_LENGTH = 50
       X_list = []
       indexes = []
       for example in tweet_examples:
           for token in example:
               indexes = []
               if self._vocab2indx.get(token) is not None:
                   print("self vocab token", self._vocab2indx[token])
                   indexes.append(self._vocab2indx[token])
               else:
                   oov = "<OOV>"
                   indexes.append(self._vocab2indx[oov])
           #Truncate

           indexes = indexes[:MAXIMUM_LENGTH]
           #pad
           while len(indexes) < MAXIMUM_LENGTH:
               indexes.append(self._len_idx2vocab)
           if len(indexes) != 50:
               print(len(indexes))
           print("X_list while in create train", X_list)
           X_list.append(indexes)
       print("X_list right before return", X_list)
       return X_list
   
   def create_word_indices(self, tweet_example):
       """
       GIVEN ONE TWEET - a list of tokens - return list of indexes 
       """
       indexes = []
       for token in tweet_example:
            print("token", token)
            if self._vocab2indx.get(token) is not None:
                print("self vocab token", self._vocab2indx[token])
                indexes.append(self._vocab2indx[token])
            else:
                oov = "<OOV>"
                print("token not found")
                indexes.append(self._vocab2indx[oov])
       return indexes
   
   def convert_X(self, examples):
        MAXIMUM_LENGTH = 50
        X_list = []
        for one_train_example in examples: 
            one_train_indices = self.create_word_indices(one_train_example)
            print("one train indices before modify", one_train_indices)
            one_train_indices = one_train_indices[:MAXIMUM_LENGTH]
            #pad
            while len(one_train_indices) < MAXIMUM_LENGTH:
               one_train_indices.append(self._len_idx2vocab)
            if len(one_train_indices) != 50:
               print(len(one_train_indices))
            X_list.append(one_train_indices)
            if len(one_train_indices) != 50:
                print(len(one_train_indices))
        print("X before convert", X_list[0:5])
        X = torch.LongTensor(X_list)
        return X
  
   def get_Y_train(self):
        return torch.LongTensor(self._train_data[1])
  
   def get_Y_dev(self):
       return torch.LongTensor(self._dev_data[1])
  
   def create_Xtrain_tensor(self):
       return torch.LongTensor(self._X_train_list)
  
   def create_Xdev_tensor(self):
       return torch.LongTensor(self._X_dev_list)


#call deepaveragingmodel
def main():
   twitter = Twitter()
   twitter._train_data = twitter.create_tweetcsv_test()
   tweet_examples_train = twitter.create_tokens(twitter._train_data[0])
   #print("create_tokens output", tweet_examples_test[0:5])
   #print("create_tokens output first list", tweet_examples_test[0])
   twitter._dev_data = twitter.create_tweetcsv_dev()
   tweet_examples_dev = twitter.create_tokens(twitter._dev_data[0])
   print(tweet_examples_dev)
   twitter._pred_data = twitter.create_tweetcsv_test()
   tweet_examples_pred = twitter.create_tokens(twitter._pred_data[0])
   twitter._embed_array = twitter.create_embeddings()
   X_train = twitter.convert_X(tweet_examples_train)
   X_dev = twitter.convert_X(tweet_examples_dev)
   X_pred = twitter.convert_X(tweet_examples_pred)
   #twitter._X_train_list = twitter.create_train(tweet_examples_test)
   #twitter._X_dev_list = twitter.create_train(tweet_examples_dev)
   #X_train = twitter.create_Xtrain_tensor()
   #X_dev = twitter.create_Xdev_tensor()
   Y_train = twitter._train_data[1]
   Y_dev = twitter._dev_data[1]
   Y_pred = twitter._pred_data[1]
   # open a file, where you ant to store the data
   data = [X_train, Y_train, X_dev, Y_dev, X_pred, Y_pred]
   file = open('mypickle.pickle', 'wb')
   # dump information to that file
   pickle.dump(data, file)
   # close the file
   file.close()

if __name__ == '__main__':
   main()