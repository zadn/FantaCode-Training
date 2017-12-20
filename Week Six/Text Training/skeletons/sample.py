import numpy as np
import os
os.environ['THEANO_FLAGS'] = "device=gpu"
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as prep
import time
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
test_split = 0.33
data = imdb.load_data(num_words=top_words)

X_train = data[0][:-1]
y_train = data[0][-1:]
X_test = data[1][:-1]
y_test = data[1][-1:]

# pad dataset to a maxumum review length in words
max_words = 500
X_train = prep.pad_sequences(X_train, maxlen=max_words)
X_test = prep.pad_sequences(X_test, maxlen=max_words)
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
start = time.time()
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=5, batch_size=128, verbose=2)
print("> Training is done in %.2f seconds." % (time.time() - start))
scores = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy: %.2f%%" % (scores[1] * 100))
# Accuracy: 86.87%
# gpu: 62s
# cpu: 58s