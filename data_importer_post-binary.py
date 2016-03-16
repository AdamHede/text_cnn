from string import punctuation
numbers = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("danish", ignore_stopwords=False)


texts = list(open('./data/tekst.csv').readlines())
codes = list(open('./data/var1.csv').readlines())
id = list(open('./data/id.csv').readlines())

def text_reader(texts):
    clean_sentences = []
    for sentence in texts:
        clean_sentence = sentence.rstrip('\n')
        clean_sentences.append(clean_sentence)
    return clean_sentences

texts = text_reader(texts)

def text_cleaner_and_tokenizer(texts):
    i = 0
    stopword_list = set(stopwords.words('danish'))
    stemmer = SnowballStemmer("danish", ignore_stopwords=False)
    filtered_texts = []
    for sentence in texts:
        global_sentence_limit = 16      ##Sets all sentences to 16 words.
        num_padding = global_sentence_limit - len(sentence.split())     ##Detects number of paddings needed
        for symbol in punctuation:                      ##Punctuation removal
            sentence = sentence.replace(symbol,'')
        for num in numbers:                             ##Number removal
            sentence = sentence.replace(str(num),'')
        sentence = sentence.decode('utf-8').lower()
        words_in_sentence = word_tokenize(sentence, language='danish')
        filtered_sentence = []
        j = 0
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

texts = text_cleaner_and_tokenizer(texts)

def sentence_padder(texts):
    padded_sentences = []
    global_sentence_length = 16
    for i in range(len(texts)):
        sentence = texts[i]
        num_padding = global_sentence_length - len(sentence.split())
        new_sentence = sentence + ' <P>' * num_padding
        padded_sentences.append(new_sentence)
        if i % 1000 == 0:
            print('Currently padded:' + str(i))
    return padded_sentences

texts = sentence_padder(texts)

##def code_selector(codes):



def data_builder(id, texts, codes):
    rows = zip(id, texts, codes)
    return rows


processed_texts = text_cleaner_and_tokenizer(texts)

data = data_builder(id, processed_texts, codes)





print('yay')