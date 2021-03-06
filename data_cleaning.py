import json
import re
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
# split data up into questions and answers
# format of data is: test_data[conversation][line]['object type']
# maybe should change so that q is strictly user and a is strictly operator
def split_q_a_2(test_data):
    questions = []
    answers = []

    for i in range(len(test_data)):
        conv = test_data[i]
        max_length = len(conv)
        j=0

        while j < max_length-1:
            if conv[j]['sender'] == 'user' and conv[j+1]['sender'] == 'operator':
                questions.append(conv[j]['utterance'])
                answers.append(conv[j+1]['utterance'])
                j +=2

    return questions, answers

def split_q_a(test_data):
    """Splits questions and answers from raw dialogue"""
    questions = []
    answers = []

    for i in range(len(test_data)):
        conv = test_data[i]
        for j in range(len(conv) - 1):
            questions.append(conv[j]['utterance'])
            answers.append(conv[j + 1]['utterance'])

    return questions, answers

def clean_text(text):
    """Removes unwanted characters like @, |||, etc"""
    # lowercase text
    text = text.lower()

    # substitute contractions
    # text = re.sub(r"i'm", "i am", text)
    # text = re.sub(r"he's", "he is", text)
    # text = re.sub(r"she's", "she is", text)
    # text = re.sub(r"it's", "it is", text)
    # text = re.sub(r"that's", "that is", text)
    # text = re.sub(r"what's", "what is", text)
    # text = re.sub(r"where's", "where is", text)
    # text = re.sub(r"how's", "how is", text)
    # text = re.sub(r"\'ll", " will", text)
    # text = re.sub(r"\'ve", " have", text)
    # text = re.sub(r"\'re", " are", text)
    # text = re.sub(r"\'d", " would", text)
    # text = re.sub(r"\'re", " are", text)
    # text = re.sub(r"won't", "will not", text)
    # text = re.sub(r"can't", "cannot", text)
    # text = re.sub(r"n't", " not", text)
    # text = re.sub(r"n'", "ng", text)
    # text = re.sub(r"'bout", "about", text)
    # text = re.sub(r"'til", "until", text)

    # get rid of username handles
    text = re.sub(r"(?:@)(\S+|$)|", "", text)

    # get rid of special characters
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

def apply_cleaning(ls):
    """Applies the clean_text function to a list of sentences"""
    clean_ls = []

    for t in ls:
        clean_ls.append(clean_text(t))
    return clean_ls

def filter_data(questions, answers, min_length, max_length):
    """Only keeps Q/A pairs that stay within the min and max sentence lengths"""
    short_questions_temp = []
    short_answers_temp = []

    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]

        if len(q.split())>=min_length and len(q.split())<=max_length \
        and len(a.split())>=min_length and len(a.split())<=max_length:
            short_questions_temp.append(q)
            short_answers_temp.append(a)

    return short_questions_temp, short_answers_temp

def create_word_freq(ls, vocab):
    """Counts how many times a given word appears"""
    for t in ls:
        for w in t.split():
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1
    return vocab

def word_2_int_dict(vocab, threshold):
    """Creates a dictionary from given vocab. Threshold determines if the word goes into the dictionary or not"""
    word2int = {}
    word_num=0
    for word, count in vocab.items():
        if count>=threshold:
            word2int[word] = word_num
            word_num+=1
    return word2int

def convert_words2int(text, mapping):
    """Converts words to their integer representation as defined in the dictionary"""

    t_vec = []
    for t in text:
        w_vec = []
        for w in t.split():
            if w not in mapping:
                w_vec.append(mapping['<UNK>'])
            else:
                w_vec.append(mapping[w])
        t_vec.append(w_vec)

    return t_vec

def sort_by_length(q,a, max_length):
    """Sorts Q/A pairs by their sentence length for ease of training"""

    sorted_q = []
    sorted_a = []

    for l in range(1, max_length+1):
        for i in enumerate(q):
            if len(i[1]) == l:
                sorted_q.append(q[i[0]])
                sorted_a.append(a[i[0]])

    return sorted_q, sorted_a

# Visualizing functions
def count_length(ls):
    """Counts how long sentences are"""
    count_ls = []
    for i in ls:
        count_ls.append(len(i.split()))
    return count_ls


# parameters for data
max_sentence_length = 50
min_sentence_length = 2
freq_threshold = 100

# save path
path = r'/Users/luchen/Documents/TrueAI/document_data/'

# load data
train_data = json.load(open('/Users/luchen/Downloads/sample_dataset/train/dialogues_task.json'))
validate_data = json.load(open('/Users/luchen/Downloads/sample_dataset/valid/dialogues_task.json'))
test_data = json.load(open('/Users/luchen/Downloads/sample_dataset/test/dialogues_task.json'))

# split questions and answers
questions, answers = split_q_a(train_data)
q_valid, a_valid = split_q_a(validate_data)
q_test, a_test = split_q_a(test_data)

# clean questions and answers
clean_qs = apply_cleaning(questions)
clean_as = apply_cleaning(answers)

clean_qv = apply_cleaning(q_valid)
clean_av = apply_cleaning(a_valid)

clean_qt = apply_cleaning(q_test)
clean_at = apply_cleaning(a_test)

# create histograms to see sentence lengths
count_qs = count_length(questions)
count_as = count_length(answers)

final_count = [count_qs, count_as]
np.save('final_count', np.array(final_count))

# remove questions/answers that are too long/too short
[final_qs, final_as] = filter_data(clean_qs, clean_as, min_sentence_length, max_sentence_length)
[final_qvs, final_avs] = filter_data(clean_qv, clean_av, min_sentence_length, max_sentence_length)
[final_qts, final_ats] = filter_data(clean_qt, clean_at, min_sentence_length, max_sentence_length)

# combine all questions/answers together to form the dictionary needed
q_s = final_qs + final_qvs + final_qts
a_s = final_as + final_avs + final_ats

# count how many times each word appears in the corpus
vocab = {}
vocab = create_word_freq(q_s, vocab)
vocab = create_word_freq(a_s, vocab)
np.save('vocab', vocab)

# map words to integers
q_word2int = word_2_int_dict(vocab, freq_threshold)
a_word2int = word_2_int_dict(vocab, freq_threshold)

# add unique codes
codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes:
    q_word2int[code] = len(q_word2int)
    a_word2int[code] = len(a_word2int)

# integers to words
q_int2word = {v_i: v for v, v_i in q_word2int.items()}
a_int2word = {v_i: v for v, v_i in a_word2int.items()}

# convert text to numbers
q_vec = convert_words2int(final_qs, q_word2int)
a_vec = convert_words2int(final_as, a_word2int)

q_vec_v = convert_words2int(final_qvs, q_word2int)
a_vec_v = convert_words2int(final_avs, a_word2int)

q_vec_t = convert_words2int(final_qts, q_word2int)
a_vec_t = convert_words2int(final_ats, a_word2int)

q_vec, a_vec = sort_by_length(q_vec, a_vec, max_sentence_length)
q_vec_v, a_vec_v = sort_by_length(q_vec_v, a_vec_v, max_sentence_length)
q_vec_t, a_vec_t = sort_by_length(q_vec_t, a_vec_t, max_sentence_length)

# cache data
with open(path + 'train_q_final.json','w') as f:
    json.dump(q_vec, f, ensure_ascii=False)

with open(path + 'train_a_final.json','w') as f:
    json.dump(a_vec, f, ensure_ascii=False)

with open(path + 'valid_q_final.json','w') as f:
    json.dump(q_vec_v, f, ensure_ascii=False)

with open(path + 'valid_a_final.json','w') as f:
    json.dump(a_vec_v, f, ensure_ascii=False)

with open(path + 'test_q_final.json','w') as f:
    json.dump(q_vec_t, f, ensure_ascii=False)

with open(path + 'test_a_final.json','w') as f:
    json.dump(a_vec_t, f, ensure_ascii=False)

with open(path + 'q_dict.json','w') as f:
    json.dump(q_word2int, f, ensure_ascii=False)

with open(path + 'a_dict.json','w') as f:
    json.dump(a_word2int, f, ensure_ascii=False)
