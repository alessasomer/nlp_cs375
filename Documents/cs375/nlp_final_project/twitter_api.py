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
    LEARNING_RATE = 1e-1
    HIDDEN_DIM1 = 200 
    HIDDEN_DIM2 = 100
    LEAKY_RELU_NEG_SLOPE = 0.01
    DROPOUT_PROB = 0.4 
    from preprocess import Twitter
    twitter = Twitter()
    embed_array = twitter.create_embeddings()
    from deepaverage import DeepAveragingNetwork
    model = DeepAveragingNetwork(2,embed_array,50, 20, 20, 0.01,0.1)
    loss_fn= nn.NLLLoss() #For binary logistic regression 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    #Random inputs to check syntax
    #X_batch = torch.randint(low=0, high=len(vocab2indx), size=(100, 10))
    #print(X_batch.shape)
    #log_probs_out = model.forward(X_batch)
    NUMBER_ITERATIONS = 300
    X_train = twitter.create_Xtrain_tensor()
    X_dev = twitter.create_Xdev_tensor()
    loss_history, train_accuracy, dev_accuracy = model.train_model(X_train, twitter.get_Y_train(), X_dev, twitter.get_Y_dev(), 
                                                             loss_fn, optimizer, NUMBER_ITERATIONS, 
                                                             batch_size = 200,
                                                             check_every=50, verbose=False)
    print(loss_history)
    print(train_accuracy)
    print(dev_accuracy)

if __name__ == '__main__':
    main()
