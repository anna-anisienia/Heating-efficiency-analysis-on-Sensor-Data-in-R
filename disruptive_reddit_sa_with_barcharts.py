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
import pandas as pd
import random
import re
from textblob import TextBlob
import praw # import  Python Reddit API Wrapper
# from dashapp import server as application

app = dash.Dash(__name__)

client_id = "2BKbi2rOzcWy5w"
secret = "GOa9xfMkea62qn6U7yHdVonrF-g"
reddit = praw.Reddit(client_id = client_id, client_secret = secret, password = "6d%v44w7J8Xn",
                     user_agent='praw_test', username='dziewczynaaa') # Reddit instance
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
    if len(re.findall(r"(bitcoin+|Bitcoin+|BTC+|BitCoin+|bitCoin+|BitcoinClassic+|Bitcoin Classic+|bitcoinclassic+|bitcoinClassic+|XBT+)", titles[i])) > 0:
        btc_titles.append(titles[i])
eth_titles = []
for i in range(len(titles)):
    if len(re.findall(r"(ethereum+|Ethereum+|ETH+|ETC+|Ethereum Classic+|EthereumClassic+|ether+)", titles[i])) > 0:
        eth_titles.append(titles[i])
# Standardize:
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

app.layout = html.Div([
                    dcc.Graph(id='scatterplot',
                    figure = {'data':[
                            go.Scatter(
                            x = btc_df.timestamp,
                            y = btc_df.SA_score,
                            mode = 'markers+lines'
                            )],
                            'layout':go.Layout(title = 'BTC sentiment over time',
                                        xaxis = {'title':'Time'},
                                        yaxis = {'title':'Sentiment'}
                                        )}
                                        ),
                    dcc.Graph(id='scatterplot2',
                    figure = {'data':[
                            go.Scatter(
                            x = eth_df.timestamp,
                            y = eth_df.SA_score,
                            mode = 'markers+lines'
                            )],
                            'layout':go.Layout(title='ETH sentiment over time',
                                                xaxis = {'title':'Time'},
                                                yaxis = {'title':'Sentiment'}
                                        )}
                                        ),
# Next two graphs:
                    dcc.Graph(id='barplot',
                        figure = {'data':[
                        go.Bar(
                            x=btc_grouped.short,  # NOC stands for National Olympic Committee
                            y=btc_grouped.title,
                            name = 'BTC',
                            marker=dict(color='#f2a900') # set the marker color to gold
                        ),
                        go.Bar(
                            x=eth_grouped.short,  # NOC stands for National Olympic Committee
                            y=eth_grouped.title,
                            name = 'ETC',
                            marker=dict(color='#4d4d4e') # set the marker color to silver
                        )
                        ],
                        'layout':go.Layout(title = 'BTC vs. ETH sentiment on Reddit' #,
                                            # xaxis = {'title':'Sentiment'},
                                            # yaxis = {'title':'Proportion (%)'}
                                        )}
                                        )
                                        ])
if __name__ == '__main__':
    app.run_server()
