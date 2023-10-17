
import streamlit as st
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

#from textblob import TextBlob

import pandas as pd
import numpy as np
import html
import os
import matplotlib.pyplot as plt
import joblib
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns





#api_key = os.environ.get("TWITTER_API_KEY")
#api_secret_key = os.environ.get("TWITTER_API_SECRET_KEY")
#access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
#access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")


# Creating the authentication object
#authenticate = tweepy.OAuthHandler(api_key, api_secret_key) 

# Set the access token and access token secret
#authenticate.set_access_token(access_token, access_token_secret) 

# Creating the API object while passing in auth information
#api = tweepy.API(authenticate, wait_on_rate_limit = True)



# Preprocessing functions
def remove_contractions(text):
    contraction_mapping = {
        "you've": "you have",
        "didn't": "did not",
        "can't": "cannot",
        "it's": "it is",
        "I'm": "I am",
        "we're": "we are",
        "that's": "that is",
        "they're": "they are",
        "shouldn't": "should not",
        "isn't": "is not",
        "won't": "will not",
        "didn't": "did not",
        "doesn't": "does not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "wouldn't": "would not",
        "aren't": "are not",
        "couldn't": "could not",
        "mustn't": "must not",
    }

    # Replace contractions in the text
    for contraction, expanded_form in contraction_mapping.items():
        text = re.sub(contraction, expanded_form, text)

    return text



# A function to clean tweets by removing name tags, 
# punctuations, links, encoded characters, and underscores
def clean_tweet(tweet):
    """Removes name tags, punctuations, links, encoded characters, and underscores."""
        
     # Lowercase the tweet text
    tweet = tweet.lower()
    
    # Defining a list of regex patterns and corresponding replacements
    
    patterns = [
        (r'@\w+', ''),             # Removing name tags
        (r'[^\w\s]', ''),          # Removing punctuations
        (r'http\S+', ''),          # Removing links (http/https)
        (r'[^\x00-\x7F]+', ''),    # Removing encoded characters
        (r'_', ''),                # Removing underscores
        (r'\d', ''),               # Removing numbers
    ]
    
    # Applying regex substitutions sequentially
    for pattern, replacement in patterns:
        tweet = re.sub(pattern, replacement, tweet)
    
    # Decoding HTML entities
    tweet = html.unescape(tweet)
    
    # Remove extra spaces and rejoin sentence
    tweet = ' '.join(tweet.split())
        
    # Returning the cleaned tweet
    return tweet


def remove_emojis(text):
    # Define a regular expression pattern for emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-A
                               u"\U0001F900-\U0001F9FF"  # Supplemental Arrows-B
                               u"\U0001FA00-\U0001FA6F"  # Supplemental Symbols and Pictographs
                               u"\U0001FA70-\U0001FAFF"  # Emoji modifiers
                               u"\U0001F004-\U0001F0CF"  # Extended emoticons
                               u"\U0001F170-\U0001F251"  # Enclosed characters
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    
    # Using the sub() method to remove emojis from the text
    text_without_emojis = emoji_pattern.sub(r'', text)
    
    return text_without_emojis



# Defining a function to tokenize text using a regular expression pattern
def tokenize_with_regexp(text):
    
        # Define the regular expression pattern for tokenization
    pattern = r'\w+|\$[\d\.]+|\S+'
        
        # Create a RegexpTokenizer with the pattern
    tokenizer = RegexpTokenizer(pattern)
        
        # Tokenize the text and return the tokens
    tokens = tokenizer.tokenize(text)
        
    return tokens


stopwords = nltk.corpus.stopwords.words('english')

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text): 
    
    output= [i for i in text if i not in stopwords]
    
    return output


#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization and remove duplicates in text
def lemmatizer(text):
    
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    
        
    return lemm_text


# Loading the model 
model = joblib.load("best_logistic_regression_model.pkl")



# Define a Streamlit app
def predict_sentiment(user_input):
    
    # Tweet processing
    
    # Remove contractions
    user_input = remove_contractions(user_input) 
    # Removes name tags, punctuations, links, encoded characters, and underscores
    user_input = clean_tweet(user_input)
    # A regular expression pattern for emojis
    user_input = remove_emojis(user_input)
    # Tokenize text using a regular expression pattern
    user_input = tokenize_with_regexp(user_input)
    # Remove stopwords
    user_input = remove_stopwords(user_input)
    # Lemmatize the tweets
    user_input = lemmatizer(user_input)
    
    
    # Predict sentiment
    model_predict = model.predict([' '.join(user_input)])  # Join tokens into a sentence
  # Emotional feedback
    if model_predict == 0:
        response = '😞 This text expresses negative sentiment. 😞\n'
        response += 'We understand it may not be a great day, but remember, you can always fly with us for a brighter experience! ✈️'
    elif model_predict == 1:
        response = '😐 This text shows a neutral emotion toward the brand or product. 😐\n'
        response += 'We appreciate your feedback and look forward to serving you better in the future. If you have any suggestions, feel free to share! 🤝'
    elif model_predict == 2:
        response = '😃 Great news! This text is filled with positive sentiment. 😃\n'
        response += 'We are thrilled that you had a positive experience with us. Let us know if you would like to share your amazing story with others! 🌟'
    else:
        response = '🤔 I cannot confidently determine the sentiment of this text. 🤔\n'
        response += 'If you have any specific feedback or questions, please do not hesitate to reach out, and we will be happy to assist you! 📞✉️'

    return response

    
    # Define a Streamlit app
def main():
# Set page title and background
    st.set_page_config(page_title="Aviators X Sentiment Analysis Project", page_icon="✈️", layout="wide")

    # Header section
    st.title("Sentiment Analysis with Aviators X")
    st.markdown("🚀 **Welcome to the Aviators X Sentiment Analysis Project!** 🚀")

    # Objectives section
    st.header("Objectives")
    st.markdown("✨ The objectives of this project include: ✨")
    st.markdown("1. Analyzing sentiment in user-provided text.✨")
    st.markdown("2. Predicting whether text expresses negative, neutral, or positive sentiment.")
    st.markdown("3. Enhancing the user experience with a user-friendly interface.")

    # User Input section
    st.sidebar.header("User Input")
    user_input = st.sidebar.text_area("Enter a tweet:")

    if st.sidebar.button('Derive Output', key="predict_button"):
        test_result = predict_sentiment(user_input)
        st.success(f'✨ The model predicts: {test_result} ✨')
        

















    st.subheader(' ------------------------Created By :  Aviators Flatiron / Moringa School ---------------------- :sunglasses:')





















   
   
   
   
   
   
   
   
   
       
if __name__ == '__main__':
    main()