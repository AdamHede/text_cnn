from string import punctuation
numbers = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("danish", ignore_stopwords=False)


texts = list(open('./data/tekst.csv').readlines())


def text_cleaner_and_tokenizer(texts):
    i = 0
    stopword_list = set(stopwords.words('danish'))
    stemmer = SnowballStemmer("danish", ignore_stopwords=False)
    filtered_texts = []
    for sentence in texts:
        for symbol in punctuation:
            sentence = sentence.replace(symbol,'')
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

text_cleaner_and_tokenizer(texts)


print('yay')