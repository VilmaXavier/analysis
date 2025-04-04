# gpt_chatbot.py (simulated GPT-like chatbot)

import json
import random

# Load intents file
with open("data/intent.json") as file:
    intents = json.load(file)

def match_intent(user_input):
    user_input = user_input.lower()
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input:
                return intent
    return None

def gpt_response(message):
    intent = match_intent(message)
    if intent:
        return random.choice(intent["responses"])
    else:
        return "Hmm, I'm not sure how to respond to that."
