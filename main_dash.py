#######
# This script will make regular API calls to http://data-live.flightradar24.com
# to obtain updated total worldwide flights data.
# ** This version continuously updates the number of flights worldwide,
#    AND GRAPHS THOSE RESULTS OVER TIME! **
######
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import random
import re
import numpy as np
import pandas as pd
import requests
import json
from textblob import TextBlob
import praw # import  Python Reddit API Wrapper
# from dashapp import server as application
app = dash.Dash(__name__)

#_____________________________________________________________________________________________________
# Stream the data from reddit and show the sentiment over time + sentiment comparison bw. BTC and ETH
#_____________________________________________________________________________________________________
client_id = "2BKbi2rOzcWy5w"
secret = "GOa9xfMkea62qn6U7yHdVonrF-g"
reddit = praw.Reddit(client_id = client_id, client_secret = secret, password = "******************",
                     user_agent='praw_test', username='***********************') # Reddit instance
list_of_items = []
fields = ('title', 'created_utc', 'num_comments', 'ups', 'downs')
# fields attribute has the relevant "key" names that we want from each submission
# we deliberately dispose of selftext (long form), comments and url for consistency bw. reddits,
# since only some of them have selftext and the comments are nested, making their storage in a single table impossible
for submission in reddit.subreddit('CryptoCurrency').new(limit=None):
    to_dict = vars(submission) # vars returns a dictionary with the attributes of the object.
    sub_dict = {field:to_dict[field] for field in fields} # we grab specific values such as title, ups/downs from the dictionary
    list_of_items.append(sub_dict) # to_dict is just a variable that is the dictionary form of each submission
## clean the reddits
reddit_df = pd.DataFrame(list_of_items)
reddit_df["timestamp"] = pd.to_datetime(reddit_df["created_utc"], unit='s') # clean the date format: up to seconds, without miliseconds

# Regexp: split df into BTC & ETH¶
reddit_df.drop_duplicates(subset='title', keep='first', inplace=True) # remove duplicates
titles = reddit_df.title.tolist()
btc_titles = []
for i in range(len(titles)):
    if len(re.findall(r"(bitcoin+|Bitcoin+|BTC+|BitCoin+|bitCoin+|BitcoinClassic+|Bitcoin Classic+|bitcoinclassic+|bitcoinClassic+|XBT+)",
                                titles[i])) > 0:
        btc_titles.append(titles[i])
eth_titles = []
for i in range(len(titles)):
    if len(re.findall(r"(ethereum+|Ethereum+|ETH+|ETC+|Ethereum Classic+|EthereumClassic+|ether+)", titles[i])) > 0:
        eth_titles.append(titles[i])
# Standardize to have the same lenght:
indices = random.sample(range(len(btc_titles)), len(eth_titles))
btc_titles = [btc_titles[i] for i in sorted(indices)]

# Sentiment analysis
def get_sentiment(sentence):
    analysis = TextBlob(sentence)
    return(analysis.sentiment.polarity) # > 0 positive, < 0 negative

btc = [get_sentiment(sentence) for sentence in btc_titles]
btc_df = pd.DataFrame({"title":btc_titles, "SA_score":btc})
btc_df = pd.merge(btc_df, reddit_df, on="title")

eth = [get_sentiment(sentence) for sentence in eth_titles]
eth_df = pd.DataFrame({"title":eth_titles, "SA_score":eth})
eth_df = pd.merge(eth_df, reddit_df, on="title")

def get_short_sentiment(sentence):
    '''function to classify sentiment of passed SA score'''
    if sentence > 0.05:
        return 'positive'
    elif sentence <= 0.05 and sentence > -0.005:
        return 'neutral'
    else:
        return 'negative'

short_btc = [get_short_sentiment(t) for t in btc_df.SA_score] # btc
short_eth = [get_short_sentiment(t) for t in eth_df.SA_score] # eth
btc_grouped = pd.DataFrame({"title":btc_df.title, "short":short_btc}).groupby("short")
eth_grouped = pd.DataFrame({"title":eth_df.title, "short":short_eth}).groupby("short")
btc_grouped = btc_grouped.count()
eth_grouped = eth_grouped.count()
btc_grouped["short"] = btc_grouped.index
eth_grouped["short"] = eth_grouped.index

#_____________________________________________________________________________________________________
# BTC and ETH values over time
#_____________________________________________________________________________________________________

url = 'https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=500'
r = requests.get(url) # Decode the JSON data into a dictionary: json_data
json_data = r.json()
btc_values_df = pd.DataFrame(json_data["Data"]) # dictionary of currency values is stored as a list under the key "Data"
# Example:
# 'Aggregated': False,
#  'ConversionType': {'conversionSymbol': '', 'type': 'direct'},
#  'Data': [{'close': 821.17,
#    'high': 826.43,
#    'low': 808.63,
#    'open': 819.63,
#    'time': 1484438400,
#    'volumefrom': 24401.41,
#    'volumeto': 20004735.1},
btc_values_df["timestamp"] = pd.to_datetime(btc_values_df["time"], unit='s') # clean the date format: up to seconds, without miliseconds

url = 'https://min-api.cryptocompare.com/data/histoday?fsym=ETH&tsym=USD&limit=500'
r = requests.get(url)
json_data = r.json() # Decode the JSON data into a dictionary: json_data
eth_values_df = pd.DataFrame(json_data["Data"])
eth_values_df["timestamp"] = pd.to_datetime(eth_values_df["time"], unit='s') # clean the date format: converts the unix timestamp to pandas date data type
# unit='s' means: up to seconds, without miliseconds

#__________________________________________________________________________________________________
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize

def generate_word_list(text_col, nr_words = 5):
    tokens = word_tokenize(text_col.to_string()) # tokenize
    lower_tokens = [t.lower() for t in tokens] # Convert the tokens into lowercase: lower_tokens
    alpha_only = [t for t in lower_tokens if t.isalpha()] # Retain alphabetic words: alpha_only
    stopwords = nltk.corpus.stopwords.words('english') # Remove all stop words: no_stops
    newStopWords = ["rt", "bitcoin", "crypto", "cryptocurrency", "blockchain", "btc", "eth", "ethereum", "price", "bitcoi", "bitcoins", "classic"]
    stopwords.extend(newStopWords)
    no_stops = [t for t in alpha_only if t not in stopwords]
    wordnet_lemmatizer = WordNetLemmatizer() # Instantiate the WordNetLemmatizer
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops] # Lemmatize all tokens into a new list: lemmatized
    bow = Counter(lemmatized) # Create the bag-of-words: bow
    word = []
    if len(lemmatized) < 5:
        word.append("no results")
    else:
        for i in range(nr_words):
            word.append(bow.most_common(nr_words)[i][0])
    return(word) # return the 5 most common tokens

def generate_word_counts(text_col, nr_words = 5):
    tokens = word_tokenize(text_col.to_string()) # tokenize
    lower_tokens = [t.lower() for t in tokens] # Convert the tokens into lowercase: lower_tokens
    alpha_only = [t for t in lower_tokens if t.isalpha()] # Retain alphabetic words: alpha_only
    stopwords = nltk.corpus.stopwords.words('english') # Remove all stop words: no_stops
    newStopWords = ["rt", "bitcoin", "crypto", "cryptocurrency", "blockchain", "btc", "eth", "ethereum", "price", "bitcoi", "bitcoins", "classic"]
    stopwords.extend(newStopWords)
    no_stops = [t for t in alpha_only if t not in stopwords]
    wordnet_lemmatizer = WordNetLemmatizer() # Instantiate the WordNetLemmatizer
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops] # Lemmatize all tokens into a new list: lemmatized
    bow = Counter(lemmatized) # Create the bag-of-words: bow
    word_count = []
    if len(lemmatized) < 5:
        word_count = 1
    else:
        for i in range(nr_words):
            word_count.append(bow.most_common(nr_words)[i][1])
    return(word_count) # return count of 5 most common tokens
#__________________________________________________________________________________________________
# Twitter:
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import pandas as pd

class TwitterClient(object):
    '''Generic Twitter Class for sentiment analysis.'''
    def __init__(self):
        '''Class constructor or initialization method.'''
        # keys and tokens from the Twitter Dev Console
        consumer_key = '******************'
        consumer_secret = '***************************************'
        access_token = '*********************************************'
        access_token_secret = '*************************************'

        # attempt authentication
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret) # create OAuthHandler object
            self.auth.set_access_token(access_token, access_token_secret) # set access token and secret
            self.api = tweepy.API(self.auth) # create tweepy API object to fetch tweets
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''Utility function to clean tweet text by removing links, special characters using simple regex statements.'''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''Utility function to classify sentiment of passed tweet using textblob's sentiment method'''
        analysis = TextBlob(self.clean_tweet(tweet)) # create TextBlob object of passed tweet text
        return(analysis.sentiment.polarity)

    def get_tweets(self, query, count = 5):
        '''Main function to fetch tweets and parse them.'''
        tweets = [] # empty list to store parsed tweets
        try:
            fetched_tweets = self.api.search(q = query, count = count, lang = "en") # call twitter api to fetch tweets
            # parsing tweets one by one
            for tweet in fetched_tweets:
                parsed_tweet = {} # empty dictionary to store required params of a tweet
                parsed_tweet['text'] = tweet.text # saving text of tweet
                parsed_tweet['created_at'] = tweet.created_at
                #parsed_tweet['lang'] = tweet.lang
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) # saving sentiment of tweet

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets # return parsed tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e)) # print error (if any)

def get_btc_df():
    api = TwitterClient()    # creating object of TwitterClient Class
    tweets = api.get_tweets(query = ['Bitcoin', 'bitcoin', 'BitCoin', 'BTC'], count = 1000)    # calling function to get tweets
    text = [] # text of the tweet
    for index, value in enumerate(tweets):
        text.append(tweets[index]["text"])

    tweet_created = [] # UTC time when this Tweet was created.
    for index, value in enumerate(tweets):
        tweet_created.append(tweets[index]["created_at"])

    sentiment = []
    for index, value in enumerate(tweets):
        sentiment.append(tweets[index]["sentiment"])

    twitter_btc_df = pd.DataFrame({"text":text, "tweet_created":tweet_created, "SA_score":sentiment})
    return(twitter_btc_df)

twitter_btc_df = get_btc_df()
#_____________________________________________________________________________________________
def get_eth_df():
    api = TwitterClient()    # creating object of TwitterClient Class
    tweets = api.get_tweets(query = ['Ether', 'Ethereum', 'Ethereum Classic', 'EthereumClassic'], count = 1000)    # calling function to get tweets
    text = [] # text of the tweet
    for index, value in enumerate(tweets):
        text.append(tweets[index]["text"])

    tweet_created = [] # UTC time when this Tweet was created.
    for index, value in enumerate(tweets):
        tweet_created.append(tweets[index]["created_at"])

    sentiment = []
    for index, value in enumerate(tweets):
        sentiment.append(tweets[index]["sentiment"])

    twitter_eth_df = pd.DataFrame({"text":text, "tweet_created":tweet_created, "SA_score":sentiment})
    return(twitter_eth_df)

twitter_eth_df = get_eth_df()
#_____________________________________________________________________________________________________
def get_short_sentiment(sentence):
    '''function to classify sentiment of passed SA score'''
    if sentence > 0.05:
        return 'positive'
    elif sentence <= 0.05 and sentence > -0.005:
        return 'neutral'
    else:
        return 'negative'

short_twitter_btc = [get_short_sentiment(t) for t in twitter_btc_df.SA_score] # twitter_btc
short_twitter_eth = [get_short_sentiment(t) for t in twitter_eth_df.SA_score] # twitter_eth
twitter_btc_grouped = pd.DataFrame({"text":twitter_btc_df.text, "short":short_twitter_btc}).groupby("short")
twitter_eth_grouped = pd.DataFrame({"text":twitter_eth_df.text, "short":short_twitter_eth}).groupby("short")
twitter_btc_grouped = twitter_btc_grouped.count()
twitter_eth_grouped = twitter_eth_grouped.count()
twitter_btc_grouped["short"] = twitter_btc_grouped.index
twitter_eth_grouped["short"] = twitter_eth_grouped.index


#_____________________________________________________________________________________________________
app.layout = html.Div([html.H1('This dashboard shows current trends about Bitcoin and Ethereum in order to help you to make an informed decision for your investment',
                        id='h1-element'),
                        html.P("You can analyze the sentiment on social media and in the news regarding the two most popular cryptocurrencies: Bitcoin (BTC) and Ethereum (ETH).\n \
                        You can choose the source you are interested in by selecting from the dropdown-menu on the left."),
                    dcc.Graph(id='scatterplot1',
                    figure = {'data':[
                            go.Scatter(
                            x = btc_df.timestamp,
                            y = btc_df.SA_score,
                            name = "BTC Sentiment on Reddit",
                            visible=True,
                            marker=dict(color='#f2a900'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = eth_df.timestamp,
                            y = eth_df.SA_score,
                            name = "ETH Sentiment on Reddit",
                            visible=True,
                            marker=dict(color='#4d4d4e'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = twitter_btc_df.tweet_created,
                            y = twitter_btc_df.SA_score,
                            name = "BTC Sentiment on Twitter",
                            visible=False,
                            marker=dict(color='#f2a900'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = twitter_eth_df.tweet_created,
                            y = twitter_eth_df.SA_score,
                            name = "ETH Sentiment on Twitter",
                            visible=False,
                            marker=dict(color='#4d4d4e'),
                            mode = 'markers+lines'
                            )
                            ],
                            'layout':go.Layout(title = 'BTC and ETH sentiment over time', showlegend=True,
                                                updatemenus = list([
                                                    dict(active=-1,
                                                         buttons=list([
                                                            dict(label = 'BTC Sentiment on Reddit',
                                                                 method = 'update',
                                                                 args = [{'visible': [True, False, False, False]},
                                                                         {'title': 'BTC sentiment over time on Reddit'}]),
                                                            dict(label = 'ETH Sentiment on Reddit',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, True, False, False]},
                                                                         {'title': 'ETH sentiment over time on Reddit'}]),
                                                            dict(label = 'Both: Sentiment on Reddit',
                                                                 method = 'update',
                                                                 args = [{'visible': [True, True, False, False]},
                                                                         {'title': 'BTC and ETH sentiment over time on Reddit'}]),
                                                            dict(label = 'BTC Sentiment on Twitter',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, True, False]},
                                                                         {'title': 'BTC sentiment over time on Twitter'}]),
                                                            dict(label = 'ETH Sentiment on Twitter',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, False, True]},
                                                                         {'title': 'ETH sentiment over time on Twitter'}]),
                                                            dict(label = 'Both: Sentiment on Twitter',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, True, True]},
                                                                         {'title': 'BTC and ETH sentiment over time on Twitter'}])

                                                        ]),
                                                    )
                                                ])
                                                ,
                                                xaxis = {'title':'Time'},
                                                yaxis = {'title':'Sentiment'}
                                        )}
                                        ),
# Sentiment grouped
                    dcc.Graph(id='pie2',
                        figure = {'data':[
                        go.Pie(
                            labels=btc_grouped.short,
                            values=btc_grouped.title,
                            name = 'BTC Sentiment on Reddit',
                            visible=True,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686']) # set the marker color to gold
                        ),
                        go.Pie(
                            labels=eth_grouped.short,
                            values=eth_grouped.title,
                            name = 'ETH Sentiment on Reddit',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686']) # set the marker color to silver
                        ),
                        go.Pie(
                            labels=twitter_btc_grouped.short,
                            values=twitter_btc_grouped.text,
                            name = 'BTC Sentiment on Twitter',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686']) # set the marker color to gold
                        ),
                        go.Pie(
                            labels=twitter_eth_grouped.short,
                            values=twitter_eth_grouped.text,
                            name = 'ETH Sentiment on Twitter',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686']) # set the marker color to silver
                        )
                        ],
                        'layout':go.Layout(title = 'BTC vs. ETH sentiment on Reddit and Twitter', showlegend=True,
                                            updatemenus = list([
                                                dict(active=-1,
                                                     buttons=list([
                                                        dict(label = 'BTC sentiment on Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [True, False, False, False]},
                                                                     {'title': 'BTC sentiment on Reddit'}]),
                                                        dict(label = 'ETH sentiment on Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [False, True, False, False]},
                                                                     {'title': 'ETH sentiment on Reddit'}]),
                                                         dict(label = 'BTC sentiment on Twitter',
                                                              method = 'update',
                                                              args = [{'visible': [False, False, True, False]},
                                                                      {'title': 'BTC sentiment on Twitter'}]),
                                                         dict(label = 'ETH sentiment on Twitter',
                                                              method = 'update',
                                                              args = [{'visible': [False, False, False, True]},
                                                                      {'title': 'ETH sentiment on Twitter'}])
                                                    ]),
                                                )
                                            ])
                                        )}
                                        ),
# BTC/ETH values over time
html.P("You can also look at the recent development in the currency values. If you are interested in a specific time interval, \
you can zoom in by selecting the desired period. If you click at the small house icon, you can reset the axis again."),
                    dcc.Graph(id='scatterplot3',
                    figure = {'data':[
                            go.Scatter(
                            x = btc_values_df.timestamp,
                            y = btc_values_df.close,
                            name = 'BTC',
                            mode = 'markers+lines'
                            ),
                            	go.Scatter(
                                x = btc_values_df.timestamp,
                                y = [btc_values_df.close.mean()]*len(btc_values_df.timestamp),
                                name = 'BTC Average',
                                visible = False,
                                line=dict(color='#33CFA5', dash='dash')
                                ),
                            go.Scatter(
                            x = eth_values_df.timestamp,
                            y = eth_values_df.close,
                            name = 'ETH',
                            mode = 'markers+lines'
                            ),
                            	go.Scatter(
                                x = eth_values_df.timestamp,
                                y = [eth_values_df.close.mean()]*len(eth_values_df.timestamp),
                                name = 'ETH Average',
                                visible = False,
                                line=dict(color='#33CFA5', dash='dash')
                                )
                            ],
                    'layout':go.Layout(title = 'BTC/ETH values over time', showlegend=True,
                                        updatemenus = list([
                                            dict(active=-1,
                                                 buttons=list([
                                                    dict(label = 'BTC',
                                                         method = 'update',
                                                         args = [{'visible': [True, True, False, False]},
                                                                 {'title': 'BTC values over time',
                                                                 'annotations': [
                                                                 dict(x=btc_values_df.iloc[btc_values_df.close.idxmax()]["timestamp"],
                                                                   y=btc_values_df.close.max(),
                                                                   xref='x', yref='y',
                                                                   text='Max value:<br>'+str(btc_values_df.close.max()),
                                                                   ax=0, ay=-40),
                                                                dict(x='2017-05-01 00:00:00',
                                                                     y=btc_values_df.close.mean(),
                                                                     xref='x', yref='y',
                                                                     text='Average value in the displayed time period:<br>'+str(btc_values_df.close.mean()),
                                                                     ax=0, ay=-40)
                                                                 ]},
                                                                 ]),
                                                    dict(label = 'ETH',
                                                         method = 'update',
                                                         args = [{'visible': [False, False, True, True]},
                                                                 {'title': 'ETH values over time',
                                                                 'annotations': [
                                                                 dict(x=eth_values_df.iloc[eth_values_df.close.idxmax()]["timestamp"],
                                                                   y=eth_values_df.close.max(),
                                                                   xref='x', yref='y',
                                                                   text='Max value:<br>'+str(eth_values_df.close.max()),
                                                                   ax=0, ay=-40),
                                                                dict(x='2017-05-01 00:00:00',
                                                                     y=eth_values_df.close.mean(),
                                                                     xref='x', yref='y',
                                                                     text='Average value in the displayed time period:<br>'+str(eth_values_df.close.mean()),
                                                                     ax=0, ay=-40)
                                                                 ]}])
                                                ]),
                                            )
                                        ]),
                                        xaxis = {'title':'Time'},
                                        yaxis = {'title':'Value (in USD)'}
                                        )}
                                        ),
# BoW plot
html.P("Additionally, you can see the most common words that are used in all discussions around Bitcoin and Ethereum on diverse channels. \
You can select the channel and the currency you are interested in from the dropdown menu on the left."),
                    dcc.Graph(id='barplot4',
                        figure = {'data':[
                        go.Bar(
                            x=generate_word_list(text_col= btc_df.title),
                            y=generate_word_counts(text_col= btc_df.title),
                            name = 'BTC words on Reddit',
                            visible=True,
                            marker=dict(color='#f2a900') # set the marker color to gold
                        ),
                        go.Bar(
                            x=generate_word_list(text_col= eth_df.title),
                            y=generate_word_counts(text_col= eth_df.title),
                            name = 'ETH words on Reddit',
                            visible=True,
                            marker=dict(color='#4d4d4e') # set the marker color to silver
                        ),
                        go.Bar(
                            x=generate_word_list(text_col= twitter_btc_df.text),
                            y=generate_word_counts(text_col= twitter_btc_df.text),
                            name = 'BTC words on Twitter',
                            visible=False,
                            marker=dict(color='#f2a900') # set the marker color to gold
                        ),
                        go.Bar(
                            x=generate_word_list(text_col= twitter_eth_df.text),
                            y=generate_word_counts(text_col= twitter_eth_df.text),
                            name = 'ETH words on Twitter',
                            visible=False,
                            marker=dict(color='#4d4d4e') # set the marker color to silver
                        )
                        ],
                        'layout':go.Layout(title = '5 most common words currently used in Bitcoin/Ethereum discussions', showlegend=True,
                                            updatemenus = list([
                                                dict(active=-1,
                                                     buttons=list([
                                                        dict(label = 'BTC words on Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [True, False, False, False]},
                                                                     {'title': '5 most common words currently used about Bitcoin on Reddit'}]),
                                                        dict(label = 'ETH words on Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [False, True, False, False]},
                                                                     {'title': '5 most common words currently used about Ethereum on Reddit'}]),
                                                        dict(label = 'Both Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [True, True, False, False]},
                                                                     {'title': '5 most common words currently used about Bitcoin and Ethereum on Reddit'}]),
                                                        dict(label = 'BTC words on Twitter',
                                                             method = 'update',
                                                             args = [{'visible': [False, False, True, False]},
                                                                     {'title': '5 most common words currently used about Bitcoin on Twitter'}]),
                                                        dict(label = 'ETH words on Twitter',
                                                             method = 'update',
                                                             args = [{'visible': [False, False, False, True]},
                                                                     {'title': '5 most common words currently used about Ethereum on Twitter'}]),
                                                        dict(label = 'Both Twitter',
                                                             method = 'update',
                                                             args = [{'visible': [False, False, True, True]},
                                                                     {'title': '5 most common words currently used about Bitcoin and Ethereum on Twitter'}])
                                                    ]),
                                                )
                                            ])
                                            ,
                                            xaxis = {'title':'Word'},
                                            yaxis = {'title':'Word count'}
                                        )}
                                        )
                                        ])
if __name__ == '__main__':
    app.run_server()
