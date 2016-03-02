import csv
import numpy as np
import os
import re

with open('tekst_out2.csv', 'r') as input_file, open('text_out.csv', 'w') as output_file:
    sentences = input_file.readlines()
    text_writer = csv.writer(output_file)

    padded_sentences = []

    sequence_length = 16     ##max(len(x.split()) for x in sentences)


    for i in range(len(sentences)):
        sentence = sentences[i]
        words_in_sentence = re.sub("[^\w]", " ",  sentence).split()
        length_of_sentence = len(words_in_sentence)
        num_padding = sequence_length - length_of_sentence
        new_sentence = words_in_sentence[:15]                   ##Force length to 16
        for j in range(num_padding):
            new_sentence.append(' <P> ')
        ##split_new_sentences = new_sentence.split()
        flat_new_words = [word for word in new_sentence if word is not '\n']
        flat_new_sentence = ' '.join(flat_new_words)
        padded_sentences.append(flat_new_sentence)
        if i % 1000 == 0:
            print('Current line: ' + str(i))

    for sentence in padded_sentences:
        output_file.write(sentence + '\n')

##Merge text and codes back together
with open('text_out.csv', 'r') as text, open('code_out2.csv', 'r') as code, open('master_out.csv', 'w') as master_out:
    sentences = text.read().splitlines()
    code_reader = code.read().splitlines()
    master_writer = csv.writer(master_out, delimiter=',')

    texts = []
    for row in sentences:
        try:
            current_text = str(row)
            current_text = current_text.decode('utf-8').lower()
            texts.append(current_text.encode('utf-8'))
        except IndexError:
            texts.append('')

    codes = []
    for row in code_reader:
        current_code = int(row)
        codes.append(current_code)


    rows = zip(codes, texts)
    for row in rows:
        master_writer.writerow(row)

print('SUCCESSFULLY CREATED MASTER FILE!')
