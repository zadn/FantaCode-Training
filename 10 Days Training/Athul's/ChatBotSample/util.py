import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import json

class DatasetLoader():

    
    
    def __init__(self, intents_file):
        self.intents_file = intents_file
        
    def createDataset(self, verbose=False):
        
        with open(self.intents_file) as json_data: 
            intents = json.load(json_data)

            words = []
            classes =[]
            documents = []
            ignore_words = ['?']

            for intent in intents['intents']:
                for pattern in intent['patterns']:

                    self.w = nltk.word_tokenize(pattern)
                    words.extend(self.w)
                    documents.append((self.w, intent['tag']))

                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

            # stem and lower each word and remove duplicates
            words = [stemmer.stem(self.w.lower()) for self.w in words if self.w not in ignore_words]
            words = sorted(list(set(words)))

            # remove duplicates
            classes = sorted(list(set(classes)))

            if verbose:
                #print (len(documents), "documents", documents)
                print (len(classes), "classes", classes)
                print (len(words), "unique stemmed words", words)

            training_data = []

            output_empty = [0]*len(classes)

            for doc in documents:
                input_data_encoded = []
                input_data = doc[0]
                # Stem each word
                input_data = [stemmer.stem(word.lower()) for word in input_data]

                for self.w in words:
                    if self.w in input_data:
                        input_data_encoded.append(1)
                    else:
                        input_data_encoded.append(0)

                output_data_encoded = list(output_empty)
                output_data_encoded[classes.index(doc[1])] = 1

                training_data.append([input_data_encoded, output_data_encoded])

            # shuffle our features and turn into np.array
            random.shuffle(training_data)
            training_data = np.array(training_data)

            # create train and test lists
            train_x = list(training_data[:,0])
            train_y = list(training_data[:,1])

            # save all of our data structures
            import pickle
            pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

            return np.array(train_x), np.array(train_y), words

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def encode_sentence(self, sentence, words):
        sentence_words = self.clean_up_sentence(sentence)

        encoded_sentence = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    encoded_sentence[i] = 1

        return encoded_sentence

    def get_responses(self):

        with open(self.intents_file) as json_data: 
            intents = json.load(json_data)

            responses = dict()
            for intent in intents['intents']:
                responses[intent['tag']] = intent['responses']

        return responses
