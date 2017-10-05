# -*- coding: utf-8 -*-
import codecs
import numpy as np
#np.random.seed(1337)
import gc
import re
import pandas as pd
import csv
import os

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
#from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical

#sys.setdefaultencoding('utf-8')
re_weight = True
BASE_DIR = './input/'
GLOVE_DIR = 'D:/Embeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
#MAX_SEQUENCE_LENGTH = 230
MAX_NB_WORDS = 220000
EMBEDDING_DIM = 300
#Some statistical analysis on the length of tokenized sequences
#mean = 11, std=6, 75%ile = 13, 95%ile = 23, 99%ile = 31, max = 245
MAX_SEQUENCE_LENGTH = 32
VALIDATION_SPLIT = 0.1


def text_to_wordlist(text, remove_stopwords=True, stem_words=False, lemmatize = False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them       
      
    #try: 
    #    text = text.lower()
    #except AttributeError: 
    #    pass
    
    # Optionally, remove stop words
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    if lemmatize:
        text = text.split()
        lemmer = WordNetLemmatizer()
        lemmed_words = [lemmer.lemmatize(word) for word in text]
        text = " ".join(lemmed_words)
        
    # Return a list of words
    return(text)
    
texts_1 = [] 
texts_2 = []
labels = []

print('Processing text dataset')
with codecs.open(TRAIN_DATA_FILE, encoding='utf_8',errors="ignore") as f:
#Original datafile needs to the decoded first, but excel csv does not
#with codecs.open(TRAIN_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        #test_texts_1.append(text_to_wordlist(values[1].decode('utf-8')))
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
    
    #df = pd.read_csv(f, delimiter=',', nrows = 1000)
    #df = pd.read_csv(f, delimiter=',')
    #texts_1 = df['question1'].apply(text_to_wordlist).tolist()
    #texts_2 = df['question2'].apply(text_to_wordlist).tolist()
    #labels = df['is_duplicate'].apply(int).tolist()  
    
print('Found %s texts.' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_labels = []  # list of label ids

with codecs.open(TEST_DATA_FILE, encoding='utf_8',errors="ignore") as f:
#with codecs.open(TEST_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_labels.append(values[0])
    #df = pd.read_csv(f, delimiter=',')
    #df = pd.read_csv(f, delimiter=',', nrows = 5000)
    
    #test_texts_1 = df['question1'].apply(text_to_wordlist).tolist()
    #test_texts_2 = df['question2'].apply(text_to_wordlist).tolist()
    #test_labels = df['test_id'].tolist()
        
print('Found %s texts.' % len(test_texts_1))
'''
texts_1 = [s.encode('utf_8') for s in texts_1]
texts_2 = [s.encode('utf_8') for s in texts_2]
test_texts_1 = [s.encode('utf_8') for s in test_texts_1]
test_texts_2 = [s.encode('utf_8') for s in test_texts_2]
test_labels = [s.encode('utf_8') for s in test_labels]
'''
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_labels = np.array(test_labels)

del test_sequences_1
del test_sequences_2
del sequences_1
del sequences_2
#del texts_1
#del texts_2
#del test_texts_1
#del test_texts_2

gc.collect()

nb_words = min(MAX_NB_WORDS, len(word_index))
print('Preparing embedding matrix.')

def build_embedding_matrix():
    
    file_name = "embedding_matrix.npy"

    # try to load from binary file
    try:
        arr = np.load(file_name)
        print("Found embedding matrix")
        return arr
    except:
        pass

    print('Indexing word vectors.')
    embeddings_index = {}
    f = codecs.open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding='utf_8')
    #f = codecs.open(os.path.join(LEX_DIR, 'lexvec.300d.W.pos.vectors'), encoding='utf_8')
#   f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf_8')

    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

    embeddings = np.random.normal(0, 1, (len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    word_to_embedding = embeddings_index # Load from text file

    for word, index in tokenizer.word_index.items():
        word_vector = word_to_embedding.get(word)

        if word_vector is not None:
            embeddings[index] = word_vector

    np.save(file_name, embeddings)

    return embeddings

embedding_matrix = build_embedding_matrix()

'''
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    word = word.decode('utf-8','ignore')
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
'''


print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print('Creating validation set.')
np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

del data_1
del data_2

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344
    