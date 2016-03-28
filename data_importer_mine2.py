import numpy as np
import re
import itertools
from collections import Counter
import csv
import pandas as pd
from string import punctuation
numbers = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

def split_master_data_into_seperate_files():
    """
    Takes the original datafile and uses Pandas to save it to seperate csvs for easier
    handling and clearning in later steps.
    """
    data = pd.read_csv('./data/agendas_data.csv')
    data.content_coding.to_csv('./data/content_coding.csv', index=False)
    data.id.to_csv('./data/id.csv', index=False)
    data.dataset.to_csv('./data/dataset.csv', index=False)
    data.tekst.to_csv('./data/tekst.csv', index=False)
    data.var1.to_csv('./data/var1.csv', index=False)
    data.var3.to_csv('./data/var3.csv', index=False)
    data.var4.to_csv('./data/var4.csv', index=False)
    data.var5.to_csv('./data/var5.csv', index=False)
    data.var6.to_csv('./data/var6.csv', index=False)
    data.var7.to_csv('./data/var7.csv', index=False)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    """
    Custom load routine :D
    """
    texts = list(open('./data/tekst.csv').readlines())
    texts = [clean_str(sentence) for sentence in texts]
    texts = [s.strip() for s in texts]
    x_text = texts
    codes = list(open('./data/var4.csv').readlines())
    codes = [s.strip() for s in codes]
    global real_codes
    real_codes = codes
    global dictionary_of_codes
    dictionary_of_codes_reverse = {}
    token_codes = [dictionary_of_codes_reverse[i] for i in codes if dictionary_of_codes_reverse.setdefault(i,len(dictionary_of_codes_reverse)+1)]   ##Create one_hot vector
    dictionary_of_codes = dict((v,k) for k,v in dictionary_of_codes_reverse.iteritems())
    token_codes_vector = np.eye(5)[token_codes]
    y = token_codes_vector
    return [x_text, y, real_codes, dictionary_of_codes]

def quick_load_data_and_labels():
    """
    Custom load routine :D
    Loads from already stemmed & padded sentences
    """
    texts = list(open('./data/preprocessed_text.csv').readlines())
    texts = [clean_str(sentence) for sentence in texts]
    texts = [s.strip() for s in texts]
    x_text = texts
    codes = list(open('./data/var4.csv').readlines())
    codes = [s.strip() for s in codes]
    global real_codes
    real_codes = codes
    global dictionary_of_codes
    dictionary_of_codes_reverse = {}
    token_codes = [dictionary_of_codes_reverse[i] for i in codes if dictionary_of_codes_reverse.setdefault(i,len(dictionary_of_codes_reverse)+1)]   ##Create one_hot vector
    dictionary_of_codes = dict((v,k) for k,v in dictionary_of_codes_reverse.iteritems())
    token_codes_vector = np.eye(5)[token_codes]
    y = token_codes_vector
    return [x_text, y, real_codes, dictionary_of_codes]

def text_cleaner_and_tokenizer(texts):
    """
    takes a list of sentences, removes punctuation, numbers, stopwords and stems.
    Then joins everything back together and returns the filtered texts as a list of unicode strings
    :param texts: list of unprocessed strings
    :return: list of unicode strings
    """
    i = 0
    stopword_list = set(stopwords.words('danish'))
    stemmer = SnowballStemmer("danish", ignore_stopwords=False)
    filtered_texts = []

    for sentence in texts:
        """
        for symbol in punctuation:
            sentence = sentence.replace(symbol,'')
        """
        for num in numbers:
            sentence = sentence.replace(str(num),'')
        sentence = sentence.decode('utf-8').lower()
        words_in_sentence = word_tokenize(sentence, language='danish')
        filtered_sentence = []
        for word in words_in_sentence:
            if word not in stopword_list:
                stem_word = stemmer.stem(word)
                filtered_sentence.append(stem_word)

        sentence = ' '.join(filtered_sentence)
        filtered_texts.append(sentence)

        i = i +1
        if i % 1000 == 0:
            print(i)
    print('Done :D!')
    return filtered_texts

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 16          ##max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence = re.sub("[^\w]", " ",  sentence).split()      ##ADAM: Proper sentence split
        sentence = sentence[:15]                                ##ADAM: Limit sentence
        num_padding = sequence_length - len(sentence)
        for j in range(num_padding):                            ##ADAM: Proper padding
            sentence.append(' <P> ')
        ##new_sentence = sentence + [padding_word] * num_padding
        sentence = ' '.join(sentence)
        padded_sentences.append(sentence)
        if i % 1000 == 0:                                       ##ADAM: See what is printed :D
            print('Padding sentence: ' + str(i))
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    print("Starting vocabulary")
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    print("Vocabulary done!")
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])      ##Might be useful if I ever need to revert back to sentences
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, real_codes, dictionary_of_codes = load_data_and_labels()                                      ##Actual loading of data
    sentences_stemmed = text_cleaner_and_tokenizer(sentences)
    sentences_padded = pad_sentences(sentences_stemmed)                                     ##Runs padding
    sentences_padded = [a.split() for a in sentences_padded]                        ## Transforms padded sentences back into list of list of words. Needed for efficient vocab building
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)                      ##Builds vocabulary
    x, y = build_input_data(sentences_padded, labels, vocabulary)                   ##Constructs the array
    id = list(open('./data/id.csv').readlines())

    with open('./data/preprocessed_text.csv', 'w') as file:
        sentences_padded = [' '.join(a) for a in sentences_padded]
        file.write('\n'.join(sentences_padded))
        print("Updated preprocessed texts!")
    return [id, x, y, vocabulary, vocabulary_inv, real_codes, dictionary_of_codes]

def quick_load_data():
    """
    Loads pre-padded and stemmed sentences to save time

    """
    sentences, labels, real_codes, dictionary_of_codes = quick_load_data_and_labels()                                      ##Actual loading of data
    ##sentences_stemmed = text_cleaner_and_tokenizer(sentences)
    ##sentences_padded = pad_sentences(sentences_stemmed)                                     ##Runs padding
    sentences = [a.split() for a in sentences]                        ## Transforms padded sentences back into list of list of words. Needed for efficient vocab building
    vocabulary, vocabulary_inv = build_vocab(sentences)                      ##Builds vocabulary
    x, y = build_input_data(sentences, labels, vocabulary)                   ##Constructs the array
    print("Successfully loaded preprocessed data")
    id = list(open('./data/id.csv').readlines())
    return [id, x, y, vocabulary, vocabulary_inv, real_codes, dictionary_of_codes]

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            one_batch = shuffled_data[start_index:end_index]
            yield shuffled_data[start_index:end_index]            ##Fixed af classes
            ##print('Batch done!')

##id, x, y, vocab, inv_vocab, real_codes, dic_of_codes = load_data()

print("YAY! and stop")