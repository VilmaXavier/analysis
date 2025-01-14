import numpy as np
import pickle
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import random
import json

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load data
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.keras")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
data_file = open("intent.json").read()
intents = json.loads(data_file)

# Clean and preprocess input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict class
def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get response
def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

# Test the chatbot
print("Chatbot is ready to talk! (type 'quit' to exit)")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    intents_list = predict_class(message, model)
    response = get_response(intents_list, intents)
    print(f"Chatbot: {response}")
from test import predict_class, get_response

def nltk_response(message, model, words, classes, intents):
    """
    Process the user input and generate a response using the chatbot model.

    Args:
        message (str): User's input message.
        model: The trained chatbot model.
        words (list): List of tokenized words used in training.
        classes (list): List of intent classes.
        intents (dict): The intents dataset.

    Returns:
        str: The chatbot's response.
    """
    try:
        # Predict the class of the input message
        intents_list = predict_class(message, model)
        # Generate response based on predicted intent
        response = get_response(intents_list, intents)
        return response
    except Exception as e:
        return f"Error processing message: {e}"
