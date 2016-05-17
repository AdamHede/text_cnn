##from __future__ import unicode_literals
import csv
from string import punctuation
import re
numbers = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

stemmer = SnowballStemmer("danish")


def write_predictions(output):
    with open("predictions.csv", 'w') as f:
        writer = csv.writer(f)
        ids = [x[0] for x in output]                      ##Open up sentences
        ids = [a.replace('\n', '') for a in ids]    ##Strips out linebreak
        real_codes = [b[1] for b in output]                     ##Open up real_codes in similar format
        pred_codes = [c[2] for c in output]
        results = zip(ids, real_codes, pred_codes)                         ##Zip it up! Ready to write
        writer.writerows(results)

with open('master_data.csv', 'r') as r, open('tekst_out2.csv', 'w') as text_out, open('code_out2.csv', 'w') as code_out:
    reader = csv.reader(r)
    text_writer = csv.writer(text_out, delimiter=b',')
    code_writer = csv.writer(code_out, delimiter=b',')
    i = 0

    for row in reader:
        current_text = str(row[1])
        current_code = int(row[0])
        current_text = current_text.decode('utf-8').lower()     ##Notice: decodes unicode
        for p in punctuation:
            current_text = current_text.replace(p,'')           ## Clean punctuation
        for num in numbers:
            current_text = current_text.replace(str(num),'')    ## Clean numbers

        stop_words_list = set(stopwords.words('danish'))

        word_tokens_list = word_tokenize(current_text, 'danish')

        filtered_text = []

        for word in word_tokens_list:
            if word.encode('utf-8') not in stop_words_list:
                stem_word = stemmer.stem(word)
                filtered_text.append(stem_word)                 ## Removed: .encode('utf-8')

        current_text = ' '.join(filtered_text).encode('utf-8')
        ##print(current_text)

        try:
            text_out.write(current_text.encode('utf-8') + '\n')        ##Join the string back together to make it work.  ##
        except AttributeError:
            text_out.write('')
        code_out.write('{}'.format(current_code) + '\n')
        i = i+1
        if i % 1000 == 0:
            print('Line ' + str(i) + ' Done! On to the next :D')

print('Output file creation done!')

with open('tekst_out2.csv', 'r') as input_file, open('text_out.csv', 'w') as output_file:
    sentences = input_file.readlines()
    text_writer = csv.writer(output_file)

    padded_sentences = []

    sequence_length = max(len(x.split()) for x in sentences)


    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence.split())
        new_sentence = sentence + '<PAD/> ' * num_padding
        split_new_sentences = new_sentence.split()
        flat_new_words = [word for word in split_new_sentences if word is not '\n']
        flat_new_sentence = ' '.join(flat_new_words)
        padded_sentences.append(flat_new_sentence)
        print('Current line: ' + str(i))
    output_file.write('%s\n' % padded_sentences)


##Merge text and codes back together
with open('text_out.csv', 'r') as text, open('code_out2.csv', 'r') as code, open('master_out.csv', 'w') as master_out:
    sentences = text.readlines()
    code_reader = code.readlines()
    master_writer = csv.writer(master_out, delimiter=',')

    texts = []
    for row in sentences:
        try:
            current_text = str(row[0])
            current_text = current_text.decode('utf-8').lower()
            texts.append(current_text.encode('utf-8'))
        except IndexError:
            texts.append('')

    codes = []
    for row in code_reader:
        current_code = int(row[0])
        codes.append(current_code)


    rows = zip(codes, texts)
    for row in rows:
        master_writer.writerow(row)

print('SUCCESSFULLY CREATED MASTER FILE!')