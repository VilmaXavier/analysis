# bert_chatbot.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("models/bert_model.pt", num_labels=5)
model.eval()

intents = json.load(open("data/intent.json"))
labels = [intent['tag'] for intent in intents['intents']]

def bert_response(message):
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    if confidence < 0.3:
        return "I'm not sure I understand."
    tag = labels[pred_idx]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return intent['responses'][0]
