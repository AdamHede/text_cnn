import csv

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
    output_file.writelines(padded_sentences)

print('breaker!')