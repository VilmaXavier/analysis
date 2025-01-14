import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intent.json').read()  # Load your intent JSON file
intents = json.loads(data_file)

# Tokenize words and process patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word from the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        
        # Add the intent class to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove unwanted words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Print details about data
print(f"{len(documents)} documents")
print(f"{len(classes)} classes", classes)
print(f"{len(words)} unique lemmatized words", words)

# Save the processed words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Create a bag of words for each pattern
    bag = []
    pattern_words = doc[0]
    # Lemmatize and tokenize the pattern words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Fill the bag of words: 1 if word exists in the pattern, 0 otherwise
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    
    # Create output row: 1 for the tag in the classes list, 0 for others
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Add to training data
    training.append([bag, output_row])

# Shuffle and convert to NumPy array for training
random.shuffle(training)
training = np.array(training, dtype=object)

# Split training data into X (input) and Y (output)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created")

# Build the model (3 layers: input layer, hidden layer, output layer)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model in the Keras format (.keras)
model.save('chatbot_model.keras')
print("Model trained and saved as 'chatbot_model.keras'")
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
