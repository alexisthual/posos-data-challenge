{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd\n",
    "import os\n",
    "import enchant\n",
    "from cnn_w2vec import CNN_model\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "NB_CATEGORIES = 52\n",
    "TRAINABLE = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size of the vocab 10806\n"
     ]
    }
   ],
   "source": [
    "dataFolder = '../posos-data-challenge/challenge_data'\n",
    "xPath = os.path.join(dataFolder, 'input_train.csv')\n",
    "yPath = os.path.join(dataFolder, 'challenge_output_data_training_file_predict_the_expected_answer.csv')\n",
    "model = CNN_model(nbCategories=NB_CATEGORIES, trainable = TRAINABLE)\n",
    "# Loading, parsing and spliting training and testing data\n",
    "x = pd.read_csv(xPath, delimiter=';', usecols=[1]).values.ravel()\n",
    "y = pd.read_csv(yPath, delimiter=';', usecols=[1]).values.ravel()\n",
    "\n",
    "# keras tokenizer gives all informations about our vocabulary\n",
    "model.preprocess(x)\n",
    "x_vocab  = list(model.tokenizer.word_index.keys())\n",
    "print('Size of the vocab', len(x_vocab))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d = enchant.Dict('fr_FR')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "not_in_enchant_fr = []\n",
    "for e in x_vocab:\n",
    "    if d.check(e) == False:\n",
    "        not_in_enchant_fr.append(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "nbre de mots erronés 4373\n"
     ]
    }
   ],
   "source": [
    "not_in_enchant_fr[:10]\n",
    "print('nbre de mots erronés', len(not_in_enchant_fr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "angiooedèmes :  ['angio-œdèmes']\n"
     ]
    }
   ],
   "source": [
    "ix = 20\n",
    "\n",
    "print(not_in_enchant_fr[ix],': ', d.suggest(not_in_enchant_fr[ix]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
