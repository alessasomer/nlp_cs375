import tweepy
import config
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
#import configparser


class Twitter:
    def __init__(self):
        self._label_examples = []
        self._tweet_examples = []
        self._vocab2indx = None

    def create_tweetcsv(self):
        trainingList =[]
        with open('training.txt') as file_object:
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
        self._label_examples = df['Label'].tolist()
        self._tweet_examples = df['Tweet'].tolist()
        #df.to_csv('tweets.csv')
        #NOTE: here tweet_examples has not been tokenized yet, and is a list of tweet texts NOT a list of words
        return self._tweet_examples, self._label_examples

    def create_tokens(self)-> list[list[str]]:
        """
        Takes in a list of tweets (str) and uses the nltk tweet tozenizer to tokenize each text string 
        RETURNS: A list of lists, where the inner list are the string tokens from one tweet
        """
        tknzr = TweetTokenizer()
        for t in self._tweet_examples:
            self._tweet_examples[self._tweet_examples(t)] = tknzr.tokenize(t)

    #CREATE EMBEDDINGS
    def create_embeddings(self) -> np.ndarray:
        embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)
        self._vocab2indx = dict(embeddings.key_to_index)
        idx2vocab = list(embeddings.index_to_key)
        embed_array = embeddings.vectors # matrix of dense word embeddings 
                                    # rows: a word 
                                    # columns: dimensions (50) of the dense embeddings
        # Add <OOV> symbol 
        new_oov_entry = len(embeddings)
        idx2vocab += ["<OOV>"]
        self._vocab2indx["<OOV>"] = new_oov_entry
        #add_the_embedding
        the_embedding = embed_array[self._vocab2indx["the"]]
        embed_array_w_oov = np.vstack((embed_array, the_embedding))
        # Add <PAD> symbol (also as embedding for the word type "the")
        new_pad_entry = len(idx2vocab)
        idx2vocab += ["<PAD>"]
        self._vocab2indx["<PAD>"] = new_pad_entry
        #the_embedding = embed_array_w_oov[self._vocab2indx["<PAD>"]]
        #embed_array_w_oov_pad = np.vstack((embed_array_w_oov, the_embedding))
        #pretrained_embedding_matrix = embed_array_w_oov_pad
        #print(type(pretrained_embedding_matrix))
        #return pretrained_embedding_matrix
        return embed_array_w_oov

    def create_X_train(self) -> list[int]: 
        """
        For each example, translate each token into its corresponding index from vocab2indx
    
        Replace words not in the vocabulary with the symbol "<OOV>" 
        which stands for 'out of vocabulary'
        - vocab2indx (dict): each vocabulary word as strings and its corresponding int index 
                           for the embeddings 
                           
        Returns: 
        - (List[int]): list of integers
        """ 
        MAXIMUM_LENGTH = 25
        indexes = []
        X_list = []
        for example in self._tweet_examples:
            for token in example:
                if self._vocab2indx.get(token) is not None:
                    indexes.append(self._vocab2indx[token])
                else:
                    indexes.append(self._vocab2indxocab2indx['<OOV>'])
            #Truncate
            indexes = indexes[:MAXIMUM_LENGTH]
            #pad

embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)
vocab2indx = dict(embeddings.key_to_index)
idx2vocab = list(embeddings.index_to_key)
embed_array = embeddings.vectors # matrix of dense word embeddings 
                                 # rows: a word 
                                 # columns: dimensions (50) of the dense embeddings

def add_the_embedding(embed_array, vocab2indx, embedding_dim=50): 
    """
    Adds "the" embedding to the embed_array matrix
    """
    the_embedding = embed_array[vocab2indx["the"]]
    out = np.vstack((embed_array, the_embedding))
    return out

# Add <OOV> symbol 
new_oov_entry = len(embeddings)
idx2vocab += ["<OOV>"]
vocab2indx["<OOV>"] = new_oov_entry
embed_array_w_oov = add_the_embedding(embed_array, vocab2indx)
pretrained_embedding_matrix = embed_array_w_oov

trainingList =[]
with open('training.txt') as file_object:
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
    df.to_csv('tweets.csv') 
    print(df)

#call deepaveragingmodel 
def main():
    twitter = Twitter()
    twitter.create_tweetcsv()
    embedding_array = twitter.create_embeddings()
    from deepaverage import DeepAveragingNetwork
    model = DeepAveragingNetwork(2,embedding_array,50, 20, 20, 0.01,0.1)
    #Random inputs to check syntax
    #X_batch = torch.randint(low=0, high=len(vocab2indx), size=(100, 10))
    #print(X_batch.shape)
    #log_probs_out = model.forward(X_batch)
    
    loss_fn= nn.NLLLoss() #For binary logistic regression 
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-1) #stochastic

if __name__ == '__main__':
    main()
