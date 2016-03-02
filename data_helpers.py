import numpy as np
import re
import itertools
from collections import Counter
import csv

def write_predictions(output):
    with open("predictions.csv", 'w') as f:
        writer = csv.writer(f)                  ##Somehow remove linebreaks?! Split array up?
        writer.writerows(output[:1000])

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
    Costum load routine :D
    """
    texts = list(open('tekst_out2.csv').readlines())
    texts = [s.strip() for s in texts]
    x_text = texts
    codes = list(open('code_out2.csv').readlines())
    codes = [s.strip() for s in codes]
    global real_codes
    real_codes = codes
    d = {}
    token_codes = [d[i] for i in codes if d.setdefault(i,len(d)+1)]             ##Create one_hot vector
    token_codes_vector = np.eye(196)[token_codes]
    y = token_codes_vector
    return [x_text, y, real_codes]



def THEIR_load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


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
    sentences, labels, real_codes = load_data_and_labels()                                      ##Actual loading of data
    sentences_padded = pad_sentences(sentences)                                     ##Runs padding
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)                      ##Builds vocabulary
    x, y = build_input_data(sentences_padded, labels, vocabulary)                   ##Constructs the array
    return [x, y, vocabulary, vocabulary_inv, real_codes]


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
