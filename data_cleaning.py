import json
import re

# split data up into questions and answers
# format of data is: test_data[conversation][line]['object type']
# maybe should change so that q is strictly user and a is strictly operator
def split_q_a(test_data):
    questions = []
    answers = []

    for i in range(len(test_data)):
        conv = test_data[i]
        for j in range(len(conv) - 1):
            questions.append(conv[j]['utterance'])
            answers.append(conv[j + 1]['utterance'])

    return questions, answers

# might need to think of another way to get rid of the whitespace trailing the thing
# clean text, so make it all lowercase and change contractions to actual words
def clean_text(text):
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
    clean_ls = []

    for t in ls:
        clean_ls.append(clean_text(t))
    return clean_ls

def filter_data(questions, answers, min_length, max_length):
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for q in questions:
        if len(q.split()) >= min_length and len(q.split())<=max_length:
            short_questions_temp.append(q)
            short_answers_temp.append(answers[i])
        i+=1

    short_questions = []
    short_answers = []

    i = 0
    for a in short_answers_temp:
        if len(a.split()) >= min_length and len(a.split()) <= max_length:
            short_answers.append(a)
            short_questions.append(short_questions_temp[i])
        i+=1

    return short_questions, short_answers

def create_word_freq(ls, vocab):
    for t in ls:
        for w in t.split():
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1
    return vocab

def word_2_int_dict(vocab, threshold):
    word2int = {}
    word_num=0
    for word, count in vocab.items():
        if count>=threshold:
            word2int[word] = word_num
            word_num+=1
    return word2int

def convert_words2int(text, mapping):

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
    sorted_q = []
    sorted_a = []

    for l in range(1, max_length+1):
        for i in enumerate(q):
            if len(i[1]) == l:
                sorted_q.append(q[i[0]])
                sorted_a.append(a[i[0]])

    return sorted_q, sorted_a

# load data
train_data = json.load(open('/Users/luchen/Downloads/sample_dataset/train/dialogues_task.json'))
# validate_data = json.load(open('/Users/luchen/Downloads/sample_dataset/valid/dialogues_task.json'))
# test_data = json.load(open('/Users/luchen/Downloads/sample_dataset/test/dialogues_task.json'))

# split questions and answers
questions, answers = split_q_a(train_data)

# clean questions and answers
clean_qs = apply_cleaning(questions)
clean_as = apply_cleaning(answers)

# remove questions/answers that are too long/too short
[final_qs, final_as] = filter_data(clean_qs, clean_as, 1, 25)

# count how many times each word appears in the corpus
vocab = {}
vocab = create_word_freq(final_qs, vocab)
vocab = create_word_freq(final_as, vocab)

# map words to integers (sketchily)
# consider using word2vec to map words
q_word2int = word_2_int_dict(vocab, 5)
a_word2int = word_2_int_dict(vocab, 5)

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
a_vec = convert_words2int(final_as, q_word2int)

q_final, a_final = sort_by_length(q_vec, a_vec, 40)

# cache data
with open('/Users/luchen/Documents/TrueAI/train_q_final_2.json','w') as f:
    json.dump(q_final, f, ensure_ascii=False)

with open('/Users/luchen/Documents/TrueAI/train_a_final_2.json','w') as f:
    json.dump(a_final, f, ensure_ascii=False)

with open('/Users/luchen/Documents/TrueAI/train_q_word2int_2.json','w') as f:
    json.dump(q_word2int, f, ensure_ascii=False)

with open('/Users/luchen/Documents/TrueAI/train_a_word2int_2.json','w') as f:
    json.dump(a_word2int, f, ensure_ascii=False)
