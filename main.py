import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


class testModel():
    def __init__(self, nbCategories=None, verbose=False):
        self.verbose = verbose

        self.nbCategories = nbCategories
        self.paddingLength = 150
        self.maxNumberWords = 1000
        self.tokenizer = text.Tokenizer(num_words=self.maxNumberWords)

        self.buildModel()

    def preprocess(self, x):
        # Inits tokenizer
        self.tokenizer.fit_on_texts(x)
        # Turns word sentences to word sequences
        sequences = self.tokenizer.texts_to_sequences(x)
        sequences = sequence.pad_sequences(sequences, self.paddingLength)

        return sequences

    def preprocessLabels(self, labels):
        return to_categorical(labels, num_classes=self.nbCategories)

    def buildModel(self):
        self.model = Sequential()
        self.model.add(Embedding(1000, 64,
            input_length=self.paddingLength, trainable=False))
        self.model.add(SimpleRNN(32))
        self.model.add(Dense(self.nbCategories))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

    def train(self, x, y, epochs=5):
        self.model.fit(x, y, shuffle='batch', epochs=epochs)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    # All variables
    verbose = True
    nbCategories = 52
    epochs = 5

    dataFolder = './challenge_data'
    xPath = os.path.join(dataFolder, 'input_train.csv')
    yPath = os.path.join(dataFolder, 'challenge_output_data_training_file_predict_the_expected_answer.csv')

    # Model creation
    model = testModel(nbCategories=nbCategories, verbose=verbose)

    # Loading, parsing and spliting training and testing data
    x = pd.read_csv(xPath, delimiter=';', usecols=[1]).values.ravel()
    y = pd.read_csv(yPath, delimiter=';', usecols=[1]).values.ravel()
    y = model.preprocessLabels(y)

    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42)

    xTrain = model.preprocess(xTrain)
    xTest = model.preprocess(xTest)

    # Training model
    model.train(xTrain, yTrain, epochs=epochs)

    # %% Testing model
    loss = model.evaluate(xTest, yTest)

    prediction = model.predict(xTest)
    predictionCategories = np.argmax(prediction, axis=1)
    yTestCategories = np.argmax(yTest, axis=1)
    accuracy = 100 * sum([predictionCategories[i] == yTestCategories[i] for i in range(len(yTestCategories))]) / len(yTestCategories)

    print('Accuracy: {:.2f} %\nLoss: {}'.format(accuracy, str(loss)))
