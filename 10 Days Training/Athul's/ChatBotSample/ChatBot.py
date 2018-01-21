import numpy as np 
import pickle
import random
from util import DatasetLoader

from tensorflow.contrib import keras
from keras.models import load_model

# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Get the responses
dataset_loader = DatasetLoader('intents.json')
responses = dataset_loader.get_responses()

# Restore the model
model = load_model('chatbot_model.h5')


def classify(sentence):
    sentence_encoded = dataset_loader.encode_sentence(sentence, words)
    prediction = model.predict(np.array([sentence_encoded]))
    intent = classes[np.argmax(prediction)]

    return intent

def response(sentence):
    intent = classify(sentence)

    return random.choice(responses[intent])

if __name__ == '__main__':
    print("Chat with Fantacode's chatbot.")
    while(True):
        #msg = input('You : ')
        msg = 'Hi there'
        if(msg == "exit"):
            break

        print('Bot :', response(msg))