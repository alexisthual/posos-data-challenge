
import os
import gensim
import numpy as np
import pandas as pd
import keras

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# GLOBAL PARAMETERS
NB_CATEGORIES = 52
EPOCHS = 10
PADDING = 150
PRE_TRAINED_DIM = 300

class CNN_model():
    def __init__(self, nbCategories=None, verbose=False, trainable = True):
        self.verbose = verbose

        self.nbCategories = nbCategories
        self.paddingLength = PADDING
        self.maxNumberWords = (1e4)
        self.trainable = trainable
        
        self.tokenizer = text.Tokenizer(num_words=self.maxNumberWords)
        
    def preprocess(self, x):
        # Inits tokenizer
        self.tokenizer.fit_on_texts(x)
        # Turns word sentences to word sequences
        sequences = self.tokenizer.texts_to_sequences(x)
        sequences = sequence.pad_sequences(sequences, self.paddingLength)
        return sequences

    def preprocessLabels(self, labels):
        return to_categorical(labels, num_classes=self.nbCategories)

    def buildModel(self, embedding):
        self.embedding = embedding
        drop_rate = 0.25
        nb_filters = 100
        filter_size = 3
        
        my_input = keras.Input(shape=(self.paddingLength,), name= 'input')
        
        embedding = (Embedding(input_dim = self.embedding.shape[0], output_dim = self.embedding.shape[1],
            weights = [self.embedding], input_length = self.paddingLength, trainable = self.trainable, name = 'embedding')) (my_input)
        
        embedding_dropped = Dropout(drop_rate, name = 'drop1')(embedding)
        conv = Conv1D(nb_filters, filter_size, activation= 'relu', name = 'conv1')(embedding_dropped)
        pooled_conv = GlobalMaxPooling1D(name = 'pool1')(conv)
        pooled_conv_dropped = Dropout(drop_rate, name = 'drop2')(pooled_conv)
    
        prob = Dense(self.nbCategories, activation= 'softmax', name = 'dense1') (pooled_conv_dropped)
        
        self.model = Model(my_input, prob)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    def train(self, x, y, epochs=5, batch_size = 64, validation_data = None):
        self.model.fit(x, y, shuffle='batch', epochs=epochs, batch_size = batch_size, validation_data = validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

if __name__ == '__main__':

    # LOAD DATA
    dataFolder = 'challenge_data'
    path2embedding = '../fr/'
    xPath = os.path.join(dataFolder, 'input_train.csv')
    yPath = os.path.join(dataFolder, 'challenge_output_data_training_file_predict_the_expected_answer.csv')
    model = CNN_model(nbCategories=NB_CATEGORIES)
    # Loading, parsing and spliting training and testing data
    x = pd.read_csv(xPath, delimiter=';', usecols=[1]).values.ravel()
    y = pd.read_csv(yPath, delimiter=';', usecols=[1]).values.ravel()
    y = model.preprocessLabels(y)

    # BUILD VOCABULARY
    # keras tokenizer gives all informations about our vocabulary
    model.preprocess(x)
    x_vocab  = list(model.tokenizer.word_index.keys())
    print('Size of the vocab', len(x_vocab))

    # Load Google's pre-trained french Word2Vec model (trained on wiki and of size 300).
    pre_trained_wv = gensim.models.Word2Vec.load(path2embedding + "fr.bin")
    # we use an embedding size of len(x_vocab) + 1 because the 0 is used for the padding
    embeddings = np.zeros((len(x_vocab) + 1 , PRE_TRAINED_DIM))
    not_in_pretrained = []
    for word, idx in model.tokenizer.word_index.items():
            if word not in pre_trained_wv.vocab:
                vec = np.zeros(PRE_TRAINED_DIM)
                not_in_pretrained.append(word)
            else:
                vec = pre_trained_wv[word]
            # word_to_index is 1-based! the 0-th row, used for padding, stays at zero
            embeddings[idx,] = vec
    print('----------embedding created----------')
    print('number of words in corpus that do not appear in pretrained word2vec: ', len(not_in_pretrained))

    # build the neural network of our model
    model.buildModel(embeddings)
    model.model.summary()

    print('total number of model parameters:',model.model.count_params())

    # process the data
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42)

    xTrain = model.preprocess(xTrain)
    xTest = model.preprocess(xTest)

    # Training model
    model.train(xTrain, yTrain, epochs=EPOCHS, validation_data = (xTest, yTest))


    # Testing model

    loss = model.evaluate(xTest, yTest)

    prediction = model.predict(xTest)
    predictionCategories = np.argmax(prediction, axis=1)
    yTestCategories = np.argmax(yTest, axis=1)
    accuracy = 100 * sum([predictionCategories[i] == yTestCategories[i] for i in range(len(yTestCategories))]) / len(yTestCategories)

    print('Accuracy: {:.2f} %\nLoss: {}'.format(accuracy, str(loss)))

