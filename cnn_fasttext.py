
# coding: utf-8

# In[55]:


import os
import gensim
import numpy as np
import pandas as pd
import keras
import csv

from tqdm import tqdm
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv1D, Dense, Dropout, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

import matplotlib.pyplot as plt


# In[2]:


'''# check if gpu is on (see in console)
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess'''


# In[69]:


# GLOBAL PARAMETERS
NB_CATEGORIES = 52
EPOCHS = 10
PADDING = 150
TRAINABLE = False
BATCHSIZE = 32


# In[65]:


class CNN_model():
    def __init__(self, nbCategories=None, verbose=False, trainable = True, medicaments = []):
        self.verbose = verbose

        self.nbCategories = nbCategories
        self.paddingLength = PADDING
        self.maxNumberWords = (1e5)
        self.trainable = trainable
        
        self.tokenizer = text.Tokenizer(num_words=self.maxNumberWords)
        
        self.medicaments = medicaments

    def preprocess(self, x):
        # Inits tokenizer
        self.tokenizer.fit_on_texts(x)
        # Turns word sentences to word sequences
        sequences = self.tokenizer.texts_to_sequences(x)
        sequences = sequence.pad_sequences(sequences, self.paddingLength)
        return sequences
    
    def spelling_correction(self, x, correct_dict ={}, verbose = False):
        corrected_x = []
        for w in x.split():
            if w in correct_dict.keys():
                w_corrected = corrected_dict[w]
                if verbose == True:
                    print('Correction of ' + w + ' in ' + w_corrected)
                w = w_corrected
            corrected_x.append(w)
        return ' '.join(corrected_x)
    
    def preprocessLabels(self, labels):
        return to_categorical(labels, num_classes=self.nbCategories)

    def buildModel(self, embedding):
        self.embedding = embedding
        drop_rate = 0.5
        nb_filters = 32
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

    def train(self, x, y, epochs=5, batch_size = BATCHSIZE, validation_data = None, callback = False):
        if callback == True:
        # checkpoint
            filepath="models_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            # Fit the model
            self.model.fit(x, y, shuffle='batch', epochs=epochs, batch_size = batch_size, validation_data = validation_data, callbacks = callbacks_list)
        else:
            self.model.fit(x, y, shuffle='batch', epochs=epochs, batch_size = batch_size, validation_data = validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)


# In[21]:


dataFolder = '../posos-data-challenge/challenge_data'
medicsPath = os.path.join(dataFolder, 'medicaments_france.xls')
correctionsPath = os.path.join(dataFolder, 'corrections.csv')

xPath = os.path.join(dataFolder, 'input_train.csv')
yPath = os.path.join(dataFolder, 'challenge_output_data_training_file_predict_the_expected_answer.csv')


# In[22]:


# adding the medicament list
MEDICAMENTS = []
medic_db = pd.read_excel(medicsPath)
for m in medic_db['Dénomination spécialité']:
    med = []
    for w in m.split():
        if w.lower()!=w:
            med.append(w)
    med = ' '.join(med)
    if len(med)!=0:
        med = med.lower()
        if med not in MEDICAMENTS:
            MEDICAMENTS.append(med.lower())
print(len(MEDICAMENTS))
for m in medic_db['Libellé ATC']:
    med = m.split()[0].lower()
    if med not in MEDICAMENTS:
        MEDICAMENTS.append(med)
print('Liste de médicaments regroupant les libéllés ATC et les dénominations de spécialité,  de taille: {}'.format(len(MEDICAMENTS)))
print('Sample of medicament names: \n', MEDICAMENTS[:10])


# In[66]:


model = CNN_model(nbCategories=NB_CATEGORIES, trainable = TRAINABLE, medicaments= MEDICAMENTS)
# Loading, parsing and spliting training and testing data
x = pd.read_csv(xPath, delimiter=';', usecols=[1]).values.ravel()
y = pd.read_csv(yPath, delimiter=';', usecols=[1]).values.ravel()

# spelling correction
corrected_dict = {}
for key, val in csv.reader(open(correctionsPath)):
    corrected_dict[key] = val
for i, s in enumerate(x):
    x[i] = model.spelling_correction(s, corrected_dict, verbose = False)


# In[50]:


# keras tokenizer gives all informations about our vocabulary
model.preprocess(x)
x_vocab  = list(model.tokenizer.word_index.keys())
print('Size of the vocab', len(x_vocab))


# In[25]:


# Load Google's pre-trained french Word2Vec model (trained on wiki and of size 300).
PRE_TRAINED_DIM = 300
# using fasttext
path2embedding = '../wiki.fr.vec'
pre_trained_wv = gensim.models.KeyedVectors.load_word2vec_format(path2embedding, binary= False)


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
print('number of words in corpus that do not appear in pretrained Fasttext: ', len(not_in_pretrained))


# In[67]:


model.buildModel(embeddings)
model.model.summary()

print('total number of model parameters:',model.model.count_params())


# In[70]:


yp = model.preprocessLabels(y)

xTrain, xTest, yTrain, yTest = train_test_split(
    x, yp, test_size=0.2, random_state=42)

xTrain = model.preprocess(xTrain)
xTest = model.preprocess(xTest)

# Training model
model.train(xTrain, yTrain, epochs=EPOCHS, validation_data = (xTest, yTest))


# In[28]:


# %% Testing model
loss = model.evaluate(xTest, yTest)

prediction = model.predict(xTest)
predictionCategories = np.argmax(prediction, axis=1)
yTestCategories = np.argmax(yTest, axis=1)
accuracy = 100 * sum([predictionCategories[i] == yTestCategories[i] for i in range(len(yTestCategories))]) / len(yTestCategories)

print('Accuracy: {:.2f} %\nLoss: {}'.format(accuracy, str(loss)))


# In[29]:


plt.figure(figsize=(15, 5))
plt.hist(y, bins = 52, label = 'train labels', density = True, alpha = 0.6)
plt.hist(predictionCategories, bins = 52, label = 'predicted labels', density= True, alpha = 0.6)
plt.legend()
plt.show()


# In[11]:


yTestCategories

