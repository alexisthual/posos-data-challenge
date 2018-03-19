# build a model based on similarity of keys words
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial


STOP_WORDS = ['alors', 'au', 'aucuns', 'aussi', 'autre', 'avant', 'avec', 'avoir', 'bon', 'car',
              'ce', 'cela', 'ces', 'ceux', 'chaque', 'ci', 'comme', 'comment', 'dans', 'des',
              'du', 'dedans', 'dehors', 'depuis', 'devrait', 'doit', 'donc', 'dos', 'début', 'elle',
              'elles', 'en', 'encore', 'essai', 'est', 'et', 'eu', 'fait', 'faites', 'fois',
              'font', 'hors', 'ici', 'il', 'ils', 'je', 'juste', 'la', 'le', 'les',
              'leur', 'là', 'ma', 'maintenant', 'mais', 'mes', 'mine', 'moins', 'mon', 'mot',
              'même', 'ni', 'nommés', 'notre', 'nous', 'ou', 'où', 'par', 'parce', 'pas',
              'peut', 'peu', 'plupart', 'pour', 'pourquoi', 'quand', 'que', 'quel', 'quelle', 'quelles',
              'quels', 'qui', 'sa', 'sans', 'ses', 'seulement', 'si', 'sien', 'son', 'sont',
              'sous', 'soyez', 'sujet', 'sur', 'ta', 'tandis', 'tellement', 'tels', 'tes', 'ton',
              'tous', 'tout', 'trop', 'très', 'tu', 'voient', 'vont', 'votre', 'vous', 'vu',
              'ça', 'étaient', 'état', 'étions', 'été', 'être', 'de', 'un', 'une', 'ai', 'ne', 'on']


def parseWord(wordCount, totalWordCount, density, item):
    if density:
        return [item[0], wordCount[0, item[1]] / totalWordCount]
    else:
        return [item[0], wordCount[0, item[1]]]


def vectorizeVocabulary(corpus, verbose=False, density=False):
    # Generate word tokens
    countVectorizer = CountVectorizer(input='content')
    countVector = countVectorizer.fit_transform(corpus)
    vocabulary = countVectorizer.vocabulary_
    wordCount = np.sum(countVector, axis=0)
    totalWordCount = np.sum(wordCount)

    vocabulary = list(
        map(partial(parseWord, wordCount, totalWordCount, density), vocabulary.items()))

    # Sort words by usage
    sortedVocabulary = sorted(vocabulary, key=lambda x: x[1], reverse=True)

    if verbose:
        print("countVector.shape: {}".format(str(countVector.shape)))
        print("wordCount.shape: {}".format(str(wordCount.shape)))
        print(sortedVocabulary[:5])

    return sortedVocabulary


def selectCat(x, y, cat_index):
    selected_questions = []
    for xx, yy in zip(x, y):
        if yy == cat_index:
            selected_questions.append(xx)
    return selected_questions


def extractTop_k(x, y, k, stop_words=STOP_WORDS):
    best_vocab = []
    nb_cat = len(set(y))
    for cat in np.arange(nb_cat):
        questions = selectCat(x, y, cat)
        cat_vocab = vectorizeVocabulary(corpus=questions)
        cat_best_vocab = []
        for w in cat_vocab:
            if len(cat_best_vocab) < k:
                if w[0] not in stop_words:
                    cat_best_vocab.append(w)
        best_vocab.append(cat_best_vocab)
    return best_vocab


'''Extract the keyword vector that contain the k most numerous word
 for each category and their weights with respect to their
 category (nb_occurences/category_size)

 if a dictionnary {cat: [keywords]} is wanted, set toVector = False
'''


def keyWordsExtract(x, y, k, stop_words=STOP_WORDS, toVector=False):
    best_vocab = extractTop_k(x, y, k, stop_words)
    nb_cat = len(set(y))
    countCategories = [0] * nb_cat
    for point in y:
        countCategories[point] += 1
    K = {}
    K_weights = {}
    for cat, best in enumerate(best_vocab):
        key_words = []
        key_weights = []
        for key_word in best:
            key_words.append(key_word[0])
            key_weights.append(key_word[1] / countCategories[cat])
        K[cat] = key_words
        K_weights[cat] = key_weights
    if toVector:
        K_vect = []
        K_vectw = []
        for c in np.arange(nb_cat):
            K_vect += K[c]
            K_vectw += K_weights[c]
        return K_vect, K_vectw
    else:
        return K, K_weights
