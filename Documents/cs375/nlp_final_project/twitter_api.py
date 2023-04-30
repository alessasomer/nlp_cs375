import tweepy
import config
import json
#import configparser

trainingList =[]
with open('training.txt') as file_object:
    for jsonObj in file_object:
        trainingDict = json.loads(jsonObj)
        trainingList.append(trainingDict)
#for t in trainingList:
    #print(t["id"], t["label"])

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
#cli = tweepy.Client(bearer_token=config.BEARER_TOKEN)

#queryy = 'covid'

#cli.search_recent_tweets(query=queryy,max_result=100)

for t in trainingList:
    try:
        status = api.get_status(t["id"])

        #get text
        text = status.text
        print(text)
    except:
        print("Tweet with ID" , t["id"] , "does not exist")
        continue

