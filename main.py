# This is a sample Python script.
import json
import nltk
from tqdm import tqdm
from textblob import TextBlob
import pandas as pd
import plotly.express as px
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


data = {}
messages = []
textblob_dataframe = pd.DataFrame
vader_dataframe = pd.DataFrame
analyzer = SentimentIntensityAnalyzer()


# function to calculate vader sentiment
def vader_sentiment_analysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']


# function to analyse
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'


def textblob_get_polarity(message):
    return TextBlob(message).sentiment.polarity


def textblob_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def generate_textblob_df():
    global textblob_dataframe
    global messages
    textblob_dataframe = pd.DataFrame(messages)
    textblob_dataframe['Polarity'] = textblob_dataframe['message'].apply(textblob_get_polarity)
    textblob_dataframe['Analysis'] = textblob_dataframe['Polarity'].apply(textblob_analysis)


def generate_vader_df():
    global vader_dataframe
    global messages
    vader_dataframe = pd.DataFrame(messages)
    vader_dataframe['Polarity'] = vader_dataframe['message'].apply(vader_sentiment_analysis)
    vader_dataframe['Analysis'] = vader_dataframe['Polarity'].apply(vader_analysis)


def generate_number_of_messages_chart():
    global textblob_dataframe
    fig = px.bar(textblob_dataframe.groupby('date').size().reset_index(name='counts'),
                 x="date", y="counts", title="Number of messages per day")
    fig.show()


def generate_plotly_graph_textblob():
    global textblob_dataframe
    fig = px.line(textblob_dataframe.groupby('date').mean(),
                title="Average Sentiment(Polarity) per day for TextBlob Analyzer")
    fig.show()


def generate_plotly_graph_vader():
    global vader_dataframe
    fig = px.line(vader_dataframe.groupby('date').mean(),
                  title="Average Sentiment(Polarity) per day for Vader Analyzer")
    fig.show()


def load_json():
    global data
    f = open('result.json')
    data = json.load(f)
    f.close()


def preprocess_data():
    global data
    global messages
    nltk.download('words')
    words = set(nltk.corpus.words.words())
    for m in tqdm(data["messages"]):
        message = m["text"]
        if isinstance(message, list):
            msg = ""
            for i in message:
                if isinstance(i, str):
                    msg += i
                if isinstance(i, dict):
                    msg += i["text"]
            message = msg

        message = " ".join(w for w in nltk.wordpunct_tokenize(message) if w.lower() in words or not w.isalpha())
        if is_word_present(message):
            f = "%Y-%m-%dT%H:%M:%S"
            out = datetime.strptime(m['date'], f)
            out = out.strftime("%m/%d/%Y")
            messages.append({'date': out, 'message': message})


def is_word_present(sentence):
    sentence = sentence.upper()
    lis = sentence.split()
    if lis.count("SHIB") > 0 or lis.count("DOGE") > 0:
        return True
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_json()
    preprocess_data()
    generate_textblob_df()
    generate_vader_df()
    generate_number_of_messages_chart()
    generate_plotly_graph_textblob()
    generate_plotly_graph_vader()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
