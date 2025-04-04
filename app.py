import time
import matplotlib.pyplot as plt

from nltk_chatbot import nltk_response
from gpt_chatbot import gpt_response
from bert_chatbot import bert_response

import pickle
import json
from keras.models import load_model

# Load NLTK chatbot assets
try:
    model = load_model("chatbot_model.keras")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    intents = json.load(open("intent.json"))
except Exception as e:
    print("Error loading model or resources:", e)
    exit()

# Test Inputs
test_inputs = [
    "Hi there!",
    "What's your name?",
    "Tell me something funny",
    "Can you help with login?",
    "Bye!"
]

# Store results
timing = {"GPT": [], "NLTK": [], "BERT": []}
responses = {"GPT": [], "NLTK": [], "BERT": []}

print("\n📊 Starting Chatbot Benchmark...\n")

for msg in test_inputs:
    print(f"User: {msg}")
    
    # GPT
    start = time.time()
    gpt_resp = gpt_response(msg)
    timing["GPT"].append(time.time() - start)
    responses["GPT"].append(gpt_resp)
    print(f"GPT ➜ {gpt_resp}")
    
    # NLTK
    start = time.time()
    nltk_resp = nltk_response(msg, model, words, classes, intents)
    timing["NLTK"].append(time.time() - start)
    responses["NLTK"].append(nltk_resp)
    print(f"NLTK ➜ {nltk_resp}")
    
    # BERT
    start = time.time()
    bert_resp = bert_response(msg)
    timing["BERT"].append(time.time() - start)
    responses["BERT"].append(bert_resp)
    print(f"BERT ➜ {bert_resp}\n")

# Bar Chart
avg_times = [sum(timing[m])/len(timing[m]) for m in timing]
plt.bar(timing.keys(), avg_times, color=['#FF8C00', '#1E90FF', '#32CD32'])
plt.title("📊 Avg. Response Time per Model")
plt.ylabel("Time (s)")
plt.savefig("comparison_graph.png")
plt.show()
