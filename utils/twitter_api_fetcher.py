"""
Wrappers for Twitter API
"""
import tweepy

auth = tweepy.OAuth1UserHandler(
    consumer_key='KVv9FbA49ICeBSRfufo340pC2',
    consumer_secret='fSAfaeck4l4hvYojmIVYYOZ1Ya6ecUqkEBEFOqRwjuROyebhOC',
    access_token='1387483664355794947-dfVCgJghYwLGqHaGKkl9xKbxrVmeJZ',
    access_token_secret='TWxbTjXPgXNXLm0BvUDKnHL7BhJwo2rLEMLv2CiwHRK9w'
)

api = tweepy.API(auth)
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAJNykgEAAAAAemZaLDxcoCtX7w07%2FompjdqRfNQ%3DfOunWQuCjA10GYbpb2igZs27Fsooowc3ylzj7RvH0L71Jd85ki')

# x = client.get_user(username='elonmusk')
x = client.search_all_tweets(query='from:elonmusk', max_results=10)
# print(api.search_tweets(q='from:elonmusk', max_results=10))