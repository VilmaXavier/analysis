# test_and_compare.py
from nltk_chatbot import nltk_response
from gpt_chatbot import gpt_response
from bert_chatbot import bert_response
from evaluate import evaluate_models

test_inputs = [
    "Hi",
    "How can I book a ticket?",
    "What is the refund policy?",
    "Bye",
    "Thanks a lot"
]

true_intents = [
    "greeting",
    "booking",
    "refund",
    "goodbye",
    "thanks"
]

nltk_preds = []
bert_preds = []
gpt_preds = []

print("Testing chatbot responses:\n")
for msg in test_inputs:
    print(f"> User: {msg}")
    n = nltk_response(msg)
    b = bert_response(msg)
    g = gpt_response(msg)
    print(f"  NLTK: {n}")
    print(f"  BERT: {b}")
    print(f"  GPT: {g}\n")
    nltk_preds.append(n)
    bert_preds.append(b)
    gpt_preds.append(g)

evaluate_models(true_intents, [nltk_preds, bert_preds, gpt_preds], ["NLTK", "BERT", "GPT"])
