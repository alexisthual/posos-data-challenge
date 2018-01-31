import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split


class testModel():
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.epochs = 5
        self.paddingLength = 150
        self.maxNumberWords = 1000
        self.tokenizer = text.Tokenizer(num_words=self.maxNumberWords)

        self.buildModel()

    def preprocess(self, x):
        # Inits tokenizer
        self.tokenizer.fit_on_texts(x)
        # Turns word sentences to word sequences
        sequences = self.tokenizer.texts_to_sequences(x)

        return sequence.pad_sequences(sequences, self.paddingLength)

    def buildModel(self):
        self.model = Sequential()
        self.model.add(Embedding(1000, 64,
            input_length=self.paddingLength, trainable=False))
        self.model.add(SimpleRNN(32))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, x, y):
        self.model.fit(x, y, shuffle='batch', epochs=self.epochs)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    verbose = True
    dataFolder = './challenge_data'
    xPath = os.path.join(dataFolder, 'input_train.csv')
    yPath = os.path.join(dataFolder, 'challenge_output_data_training_file_predict_the_expected_answer.csv')

    x = pd.read_csv(xPath, delimiter=';', usecols=[1]).values.ravel()
    y = pd.read_csv(yPath, delimiter=';', usecols=[1]).values.ravel()

    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42)

    if verbose:
        # Checks that indices are in the same order
        # print(all([xTrain[n][0] == yTrain[n][0] for n in range(len(xTrain))]))

        print(xTrain.shape)
        print(xTrain[0])

    model = testModel(verbose)
    xTrain = model.preprocess(xTrain)
    xTest = model.preprocess(xTest)
    print(xTrain)
    print(xTrain.shape)

    model.train(xTrain, yTrain)

    # %% Just this
    loss = model.evaluate(xTest, yTest)
    prediction = model.predict(xTest).ravel()
    print(prediction)
    accuracy = 100 * sum([round(prediction[i]) == yTest[i] for i in range(len(yTest))]) / len(yTest)
    print('Accuracy: {:.2f} %\nLoss: {}'.format(accuracy, str(loss)))
