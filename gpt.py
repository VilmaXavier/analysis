from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Function to generate a response from the GPT model
def generate_response(prompt, max_length=50):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response from the model
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Inference function for GPT response
def gpt_response(message, model):
    prompt = f"Chatbot: {message}"  # Format the input message
    response = generate_response(prompt)  # Generate the response from GPT
    return f"GPT Response: {response}"
