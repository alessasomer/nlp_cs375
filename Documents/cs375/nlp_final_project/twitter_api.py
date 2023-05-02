import tweepy
import config
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
#import configparser


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

def pad(original_indices_list: list, pad_index: int, maximum_length=100) -> list: 
    """
    Given original_indices_list, concatenates the pad_index enough times 
    to make the list to maximum_length. 
    """
    #NOT SURE IF IM DOING THIS RIGHT LOL
    while len(original_indices_list) < maximum_length:
        original_indices_list.append(pad_index)
    return original_indices_list

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

#for t in trainingList:
    #try:
        #status = api.get_status(t["id"])

        #get text
        #text = status.text
        #print(text)
    #except:
        #print("Tweet with ID" , t["id"] , "does not exist")
        #continue

# create dataframe
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
    from deepaverage import DeepAveragingNetwork
    model = DeepAveragingNetwork(2, embed_array_w_oov,50, 20, 20, 0.01,0.1)
    # Random inputs to check syntax
    X_batch = torch.randint(low=0, high=len(vocab2indx), size=(100, 10))
    print(X_batch.shape)
    log_probs_out = model.forward(X_batch)
    
    #loss_fn= nn.NLLLoss() #For binary logistic regression 
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-1) #stochastic

if __name__ == '__main__':
    main()
