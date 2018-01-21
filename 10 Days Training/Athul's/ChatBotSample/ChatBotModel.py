import numpy as np
from util import DatasetLoader

from tensorflow.contrib import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import RMSprop


# Parameters
num_epochs = 10000
batch_size = 8

# Load the Dataset
dataset_loader = DatasetLoader('intents.json')
x_train, y_train, vocab = dataset_loader.createDataset()

input_dim = x_train.shape[1]
num_classes = y_train.shape[1]
shape_of_y = y_train.shape[:]

# Create the model

model = Sequential()
# Add a Fully connected layer (Dense Layer) with input_dim input neurons and 8 output neurons
model.add(Dense(8, activation='relu', input_shape=(input_dim,)))
model.add(Dropout(0.2))

# This time we don't have to specify the input size
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_data=None)
model.save('chatbot_model.h5')



# sentence = 'How are you?'
# sentence_encoded = dataset_loader.encode_sentence(sentence, vocab)
# prediction = model.predict(np.array([sentence_encoded]))
# print(prediction)
