import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))  # Vocabulary (115 words)
classes = pickle.load(open('classes.pkl', 'rb'))  # Intent tags (11 classes)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='chatbot.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def create_bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)  # Creates vector of 115 zeros
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array([bag], dtype=np.float32)

def predict_intent(sentence):
    bow = create_bag_of_words(sentence)
    interpreter.set_tensor(input_details[0]['index'], bow)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    results = [[i, float(r)] for i, r in enumerate(output[0]) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chat interface
print("Bot: Hello! Type 'quit' to exit")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    intents_list = predict_intent(user_input)
    response = get_response(intents_list)
    print("Bot:", response)