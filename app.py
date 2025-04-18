import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
# from keras.src.backend.common import name_scope_stack

# name_scope_stack = []  # force initialize the name_scope_stack before loading
# model = load_model("imdb_rnn_model.h5")
# import h5py

# f = h5py.File("imdb_rnn_model.h5", "r")
# print(f.attrs.keys())
# Load the model
# model = load_model('imdb_rnn_model.h5')
# print(imdb.get_word_index())
# print(imdb.get_word_index().items())
model = load_model("imdb_rnn_model.h5")

# model.summary()

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])

def decode_review(encoded_review):
    """Decode the review from integers to words."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def padding_review(review):
    """Return a padded review."""
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# def predict_sentiment(user_review):
#     """Predict the sentiment of the review."""
#     padded_review = padding_review(user_review)
#     prediction = model.predict(padded_review)
#     sentiment = 'positive' if (prediction[0][0] * 10) > 0.5 else 'negative'
#     return sentiment, prediction[0][0] * 10


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

user_review = st.text_area("Movie Review", "Type your review here...")

if st.button('Classify'):
    try:
        padded_review = padding_review(user_review)
        prediction = model.predict(padded_review)
        sentiment = 'positive' if (prediction[0][0]) > 0.5 else 'negative'
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Probability: {prediction[0][0] * 100:.2f} %")
    except Exception as e:
        st.write("Error in processing the review. Please check your input.")
        st.write(e)
    
    # st.write(f"Decoded Review: {decode_review(preprossed_review[0])}")

else:
    st.write("Please enter a review and click 'Classify' to see the sentiment prediction.")
