from __future__ import print_function
from functools import reduce
import re
import numpy as np

from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences








def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):

    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):

    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)

def vectorize_question(story, query, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    stories = tokenize(story)

    x = [word_idx[w] for w in stories]
    # let's not forget that index 0 is reserved
    xs.append(x)

    xq = [word_idx[w] for w in query.split()]
    xqs.append(xq)

    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen)



def categorize_stories(data, word_idx ):
    rev_word_idx = []
    for item in data:
        if item != 0:
            rev_word_idx.append(list(word_idx.keys())[list(word_idx.values()).index(item)])
    return rev_word_idx




def get_model(vocab_size, story_maxlen, query_maxlen):

    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    BATCH_SIZE = 16
    EPOCHS = 400

    sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(0.3)(encoded_question)
    encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
    encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

    merged = layers.add([encoded_sentence, encoded_question])
    merged = RNN(EMBED_HIDDEN_SIZE)(merged)
    merged = layers.Dropout(0.3)(merged)
    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model([sentence, question], preds)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    return model, BATCH_SIZE, EPOCHS

def train_model(model, x, xq, y, tx, txq, ty, BATCH_SIZE, EPOCHS ):
    model.fit([x, xq], y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.05)
    loss, acc = model.evaluate([tx, txq], ty, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    model.save_weights('data/model_weights.h5')
    print('Model Saved !!!')
    return


def get_data():
    train = get_stories(open('tasks/en/qa2_two-supporting-facts_train.txt'))
    test = get_stories(open('tasks/en/qa2_two-supporting-facts_test.txt'))

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    return x, xq, y, tx, txq, ty, vocab_size, word_idx, story_maxlen, query_maxlen





def load_model():
    x, xq, y, tx, txq, ty, vocab_size, word_idx, story_maxlen, query_maxlen = get_data()

    model, BATCH_SIZE, EPOCHS = get_model(vocab_size, story_maxlen, query_maxlen)
    model.load_weights('data/model_weights.h5')


    predicted = []
    stories = input("Enter the story: ")


    query = input("Enter your query : ")
    testx, testxq = vectorize_question(stories, query, word_idx, story_maxlen, query_maxlen)
    prediction = model.predict([testx, testxq])
    predicted.append(prediction.argmax())
    predicted = categorize_stories(predicted, word_idx)
    print('Answer : ', predicted[0])


train = input('Do you want to train the model? (yes/no)')
if train.strip() == 'yes':
    x, xq, y, tx, txq, ty, vocab_size, word_idx, story_maxlen, query_maxlen = get_data()
    model, BATCH_SIZE, EPOCHS = get_model(vocab_size, story_maxlen, query_maxlen)
    train_model(model, x, xq, y, tx, txq, ty, BATCH_SIZE, EPOCHS)
    load_model()
elif train.strip() == 'no':
    load_model()

#     Sample Input : Mary got the milk there. John moved to the bedroom. Sandra went back to the kitchen. Mary travelled to the hallway.
#      Sample Query : Where is the milk
#
#       Corresponding Out : 'hallway'.

# More samples can be found in tasks/en/qa2_two-supporting_facts_test.txt
else:
    print('Invalid Input')