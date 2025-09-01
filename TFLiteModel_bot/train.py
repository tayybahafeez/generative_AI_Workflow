import random
import json
import pickle
import numpy as np
# import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import nltk

# Download essential NLTK datasets
nltk.download('punkt')      # For tokenization
nltk.download('wordnet')    # For lemmatization
nltk.download('omw-1.4')    # Open Multilingual WordNet (required for wordnet)
nltk.download('punkt_tab')
# Initialize lemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open(r'C:\Users\hp\Documents\create_chatbot_using_python\intents.json').read())# Preprocess data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and split data
random.shuffle(training)
training = np.array(training)
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile and train
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("Training complete. Model saved as 'chatbot_model.h5'")