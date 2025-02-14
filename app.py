import streamlit as st
from transformers import pipeline
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained Hugging Face model
chatbot = pipeline("text-generation", model="distilgpt2")

# Define stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords and punctuation
def remove_stopwords(input_text):
    input_text = re.sub(r'[^a-zA-Z\s]', '', input_text)  # Remove punctuation
    word_tokens = word_tokenize(input_text.lower())
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

# Define chatbot response logic
def healthcare_chatbot(user_input):
    filtered_input = remove_stopwords(user_input)

    if "symptom" in filtered_input:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in filtered_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in filtered_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot(filtered_input, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit UI
st.title("Healthcare Assistant Chatbot")
st.write("How can I assist you today?")

user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        st.write("**User:**", user_input)
        response = healthcare_chatbot(user_input)
        st.write("**Healthcare Assistant:**", response)
    else:
        st.write("Please enter a query.")
