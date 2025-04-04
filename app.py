import time
import matplotlib.pyplot as plt

from nltk_chatbot import nltk_response
from gpt_chatbot import gpt_response
from bert_chatbot import bert_response

from keras.models import load_model
import pickle
import json

# Load shared resources for NLTK chatbot
model = load_model("chatbot_model.keras")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
intents = json.loads(open("intent.json").read())

# Sample test inputs to evaluate across all models
test_inputs = [
    "Hello there!",
    "What is your name?",
    "Tell me a joke.",
    "How can I reset my password?",
    "Goodbye"
]

# Dictionaries to store metrics
timing = {"GPT": [], "NLTK": [], "BERT": []}
responses = {"GPT": [], "NLTK": [], "BERT": []}

print("Running chatbot comparison...\n")

# Loop through test messages
for message in test_inputs:
    print(f"User: {message}")
    
    # GPT
    start = time.time()
    gpt_ans = gpt_response(message)
    end = time.time()
    responses["GPT"].append(gpt_ans)
    timing["GPT"].append(end - start)
    print(f"GPT Response: {gpt_ans}")
    
    # NLTK
    start = time.time()
    nltk_ans = nltk_response(message, model, words, classes, intents)
    end = time.time()
    responses["NLTK"].append(nltk_ans)
    timing["NLTK"].append(end - start)
    print(f"NLTK Response: {nltk_ans}")
    
    # BERT
    start = time.time()
    bert_ans = bert_response(message)
    end = time.time()
    responses["BERT"].append(bert_ans)
    timing["BERT"].append(end - start)
    print(f"BERT Response: {bert_ans}\n")

# Plotting the comparison
models = ["GPT", "NLTK", "BERT"]
average_time = [sum(timing[m])/len(timing[m]) for m in models]

plt.figure(figsize=(10, 6))
plt.bar(models, average_time, color=["orange", "skyblue", "lightgreen"])
plt.title("Average Response Time Comparison")
plt.ylabel("Time (seconds)")
plt.xlabel("Chatbot Models")
plt.tight_layout()
plt.savefig("comparison_graph.png")
plt.show()
