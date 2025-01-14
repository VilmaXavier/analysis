import nltk
nltk.download('punkt')

import streamlit as st
from keras.models import load_model
import pickle
import json
import matplotlib.pyplot as plt
from text_nltk import nltk_response
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the trained models and other resources
def load_resources():
    # Load the NLTK model and data
    nltk_model = load_model("chatbot_model.keras")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    with open("intent.json") as f:
        intents = json.load(f)

    # Load BERT model and tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(classes))  # Adjust num_labels accordingly

    # Load GPT model and tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

    return nltk_model, words, classes, intents, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model

# Function to show chatbot for different models
def show_chatbot(model_type, nltk_model, words, classes, intents, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model):
    user_input = st.text_input("You: ", "")
    
    if user_input:
        if model_type == "nltk":
            response = nltk_response(user_input, nltk_model, words, classes, intents)
        elif model_type == "bert":
            response = bert_response(user_input, bert_tokenizer, bert_model)  # Use actual BERT response function
        elif model_type == "gpt":
            response = gpt_response(user_input, gpt_tokenizer, gpt_model)  # Use actual GPT response function
        st.write(f"{model_type.capitalize()} Chatbot: {response}")

# Function for BERT response generation
def bert_response(message, tokenizer, model):
    inputs = tokenizer.encode(message, return_tensors="pt", truncation=True, padding=True, max_length=64)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Decode the label to return the intent
    return f"BERT Response: Predicted Intent Class {predicted_class}"  # Modify this as per your model's output

# Function for GPT response generation
def gpt_response(message, tokenizer, model, max_length=50):
    inputs = tokenizer.encode(message, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Sidebar options
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio("Choose Model", ["nltk", "bert", "gpt", "analysis"])

# Load all resources
nltk_model, words, classes, intents, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model = load_resources()

# Main content
if model_option in ["nltk", "bert", "gpt"]:
    st.title(f"{model_option.capitalize()} Chatbot")
    st.write(f"Ask questions to the {model_option} chatbot!")
    show_chatbot(model_option, nltk_model, words, classes, intents, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model)

elif model_option == "analysis":
    st.title("Model Comparison and Analysis")
    st.write("Here is the analysis of the three models:")

    # Example accuracy values for analysis (replace with actual model accuracies)
    model_names = ["NLTK", "BERT", "GPT"]
    accuracies = [0.85, 0.92, 0.90]

    # Plotting model comparison
    fig, ax = plt.subplots()
    ax.bar(model_names, accuracies, color=['blue', 'green', 'red'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')

    st.pyplot(fig)

    # Show additional analysis if needed
    st.write("Model Analysis based on accuracy and performance.")
    st.write("Further analysis can include metrics like F1-score, Precision, Recall, etc.")
