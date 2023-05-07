import tweepy
import config
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer


#call deepaveragingmodel 
def main():
    from preprocess import Twitter
    twitter = Twitter()
    embed_array = twitter.create_embeddings()
    from deepaverage import DeepAveragingNetwork
    model = DeepAveragingNetwork(2,embed_array,50, 20, 20, 0.01,0.1)
    #Random inputs to check syntax
    #X_batch = torch.randint(low=0, high=len(vocab2indx), size=(100, 10))
    #print(X_batch.shape)
    #log_probs_out = model.forward(X_batch)
    
    loss_fn= nn.NLLLoss() #For binary logistic regression 
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-1) #stochastic

if __name__ == '__main__':
    main()
