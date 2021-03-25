import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import sys
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import sys
from six.moves import cPickle as pickle
import pandas as pd
import gzip
import seaborn as sns
import string
from time import time
import nltk
stops = set(stopwords.words("english"))
import streamlit as st
import urllib.request as tr
import re
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

#st.title('Amazon Musical Instruments Reviews')
st.markdown("<h1 style='text-align: center; color: red;'>Amazon Musical Instruments Reviews</h1>", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_data():
    data=pd.read_csv('Musical_instruments_reviews.csv')
    data_js = pd.read_json('Musical_Instruments_5.json',lines=True,orient='columns')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data
    
    
@st.cache(allow_output_mutation=True)    
def load_json():
    df = pd.read_json('Musical_Instruments_5.json',lines=True,orient='columns')
    return df

@st.cache(allow_output_mutation=True)
def load_few(nrows):
    df2 = pd.read_csv('Musical_instruments_reviews.csv',nrows=nrows)
    return df2

    
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! Data Loaded ")
df = load_json()

if st.checkbox('Show raw data'):
    #st.subheader('Raw data')
    st.markdown("<h3 style='text-align: left; color: blue;'>Raw Data</h3>", unsafe_allow_html=True)
    st.write('Total number of reviews are %s .' %len(data.index))
    st.dataframe(data)
st.write("")
st.write("")
st.write("")

overall = st.slider('Overall Ratings' , 1 , 5 , 3)
filtered_data = data[data['overall'] == overall]

st.subheader('All the reviews of %s star ratings' % overall)
st.write('Total number of reviews are %s .' %len(filtered_data.index))
st.dataframe(filtered_data)
st.write("")
st.write("")
st.write("")


df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
data.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
df['helpful_numerator'] = df['helpful'].apply(lambda x: x[0])
df['helpful_denominator'] = df['helpful'].apply(lambda x: x[1])
df['Ratio'] = round(df['helpful_numerator']/df['helpful_denominator']*100,0)
df['Ratio'] = df['Ratio'].fillna(0)

#st.subheader('% Distribution of Helpful ratings')
st.markdown("<h3 style='text-align: left; color: blue;'>% Distribution of Helpful ratings </h3>", unsafe_allow_html=True)
ratio_values = np.histogram(df['Ratio'], bins=range(100), range=(0,100))[0]
st.line_chart(ratio_values)
st.write("So we can see from almost 10261 reviews , 7189 reviews don't have a feedback in the form of helpful or not-helpful ! i.e [0,0] --> 0%")
st.write("")
st.write("")
st.write("")
st.write("")

#st.title("Test Genuinity of 'reviewText' vs 'Score'")
st.markdown("<h1 style='text-align: center; color: red;'>Test Genuinity of 'reviewText' vs 'Score'</h3>", unsafe_allow_html=True)

df2 = load_few(47)
df2 = df2.iloc[:,[4,5]]

conditions = [
    (df2['overall'] <= 2),
    (df2['overall'] == 3),
    (df2['overall'] > 3)
    ]

# create a list of the values we want to assign for each condition
values = ['Negative', 'Neutral', 'Positive']

# create a new column and use np.select to assign values to it using our lists as arguments
df2['User_Sentiment'] = np.select(conditions, values)
df2['reviewText'] = df2['reviewText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df2['reviewText'] = df2['reviewText'].str.replace('[^\w\s]','')

stop = stopwords.words('english')
df2['reviewText'] = df2['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Lemmatization

# Let's define our sentiment analyzer function:
polarity_score =[]
def analyze_sentiment(cleaned_verified_reviews):
    analysis = TextBlob(cleaned_verified_reviews)
    if analysis.sentiment.polarity > 0.2:
        polarity_score.append(analysis.sentiment.polarity)
        return 'Positive'
    elif analysis.sentiment.polarity >= -0.2 and analysis.sentiment.polarity <= 0.2 :
        polarity_score.append(analysis.sentiment.polarity)
        return 'Neutral'
    else:
        polarity_score.append(analysis.sentiment.polarity)
        return 'Negative'
df2['Review_Sentiment'] = df2['reviewText'].apply(lambda x: analyze_sentiment(x))

df2['Polarity_Score'] = polarity_score


df2 = df2[df2.User_Sentiment != 'Neutral']
df2 = df2[df2.Review_Sentiment != 'Neutral']
comparison_column = np.where(df2["User_Sentiment"] == df2["Review_Sentiment"], True, False)
df2["result"] = comparison_column
df2 = df2[df2.result != True]
#st.header('Example of Non-Genuine Reviews !')
st.markdown("<h3 style='text-align: left; color: blue;'>Example of Non-Genuine Reviews !</h3>", unsafe_allow_html=True)
st.table(df2)
st.write("")
st.write("")
st.write("")

#st.header('Test Genuinity of your review here !')
st.markdown("<h2 style='text-align: left; color: blue;'>Test Genuinity of your review here !</h2>", unsafe_allow_html=True)

content = st.text_area('Enter your product review here ...')
rate = st.selectbox('Star Ratings',[1,2,3,4,5])

if(rate == 1 or rate == 2):
    user_senti = "Negative"
elif(rate == 3):
    user_senti = "Neutral"
elif(rate == 4 or rate == 5):
    user_senti = "Positive"
#df3['User_Sentiment'] = user_senti


def processRow(row):
 comment = row
#Lower case
 comment.lower()

#Removes unicode strings like "\u002c"  -> ,(comma)
 comment = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', comment)
    
# Removes non-ascii characters. note : \x00 to \x7f is 00 to 255
# non-ascii characters like copyrigth symbol, trademark symbol
 comment = re.sub(r'[^\x00-\x7f]',r'',comment)
               
#convert any url to URL
 comment = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',comment)
               
#Convert any @Username to "AT_USER"
 comment = re.sub('@[^\s]+','AT_USER',comment)

#Remove additional white spaces
 comment = re.sub('[\s]+', ' ', comment)
 comment = re.sub('[\n]+', ' ', comment)

#Remove not alphanumeric symbols white spaces
 comment = re.sub(r'[^\w]', ' ', comment)

#Removes hastag in front of a word """
 comment = re.sub(r'#([^\s]+)', r'\1', comment)

#Replace #word with word
 comment = re.sub(r'#([^\s]+)', r'\1', comment)

 comment = re.sub(':\)|:\(|:\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', comment)

#trim
 comment = comment.strip('\'"')
     
#lemma
 comment =" ".join([Word(word).lemmatize() for word in comment.split()])

 row = comment
 return row
               
#call the function to process your data
review = processRow(content)

def analyze_sentiment(cleaned_verified_reviews):
    analysis = TextBlob(cleaned_verified_reviews)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0 :
        return 'Neutral'
    else:
        return 'Negative'

Review_Senti = analyze_sentiment(review)
def fun():
    if st.button('Submit Review'):
        st.write('')
        st.write('')
        if(content) == '':
            st.error('Please Enter some Review !')
            return
        elif(Review_Senti == 'Neutral' or user_senti == 'Neutral'):
            st.success('Review Submitted ! Genuine.')
            return
        elif(Review_Senti == user_senti):
            st.success('Review Submitted ! Genuine.')
        else:
            st.warning('Not Genuine ! Please recheck your Review or Rating Stars.')
fun()
