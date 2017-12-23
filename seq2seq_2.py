import tensorflow as tf
import json
import time
import re
import numpy as np

# ------------------------
# DATA CLEANING FNS
# ------------------------
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


# initialize model inputs as placeholders
def model_inputs():
    """Create placeholders for inputs to the model"""
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob


def model_inputs_2(batch_size, max_length):
    input_data = tf.placeholder(tf.int32, [batch_size, max_length], name='input')
    targets = tf.placeholder(tf.int32, [batch_size, max_length], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

# Build the encoding RNN model, and return the final states of the RNN
def encoding_layer(rnn_inputs, num_rnns, num_layers, keep_prob, seq_length):

    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_rnns)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob)

    encode_cell = tf.nn.rnn_cell.MultiRNNCell([drop]* num_layers)

    enc_output, enc_state = tf.nn.dynamic_rnn(
        encode_cell,
        rnn_inputs,
        sequence_length=seq_length,
        dtype=tf.float32)

    return enc_output, enc_state

# append code <GO> so that it knows when to encode/decode
def process_encoding_input(target_data, word2int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    dec_input = tf.concat([tf.fill([batch_size, 1],
                                   word2int['<GO>']), ending], 1)

    return dec_input


# Final Model
def seq2seq_model(input_data, target_data, keep_prob, batch_size, seq_length,
                  a_vocab_size, q_vocab_size, encoding_embedding_size, decoding_embedding_size,
                  rnn_size, num_layers, q_word2int):

    # Map a sentence onto a lower dimension factor space
    encoded_embed_input = tf.contrib.layers.embed_sequence(
        input_data,
        a_vocab_size,
        encoding_embedding_size,
        initializer=tf.random_uniform_initializer(-1,1))

    # Get final state from encoding
    enc_output, encoded_state = encoding_layer(encoded_embed_input,
                                   rnn_size,
                                   num_layers,
                                   keep_prob,
                                   seq_length)

    # Decode
    decoded_input = process_encoding_input(target_data, q_word2int, batch_size)
    decoded_embeddings = tf.Variable(tf.random_uniform([q_vocab_size+1, decoding_embedding_size],-1,1))

    decoded_embed_input = tf.nn.embedding_lookup(decoded_embeddings, decoded_input)


    # decoding layer needs target sentences, target embedded sentences, encoded states, total word number,
    # sequence length, rnn size, depth, translating word to number, drop rate, batch size
    train_logits, infer_logits = decoding_layer(
        decoded_embed_input,
        decoded_embeddings,
        encoded_state,
        enc_output,
        seq_length,
        rnn_size,
        num_layers,
        q_word2int,
        keep_prob,
        batch_size)

    return train_logits, infer_logits


def decoding_layer(dec_embed_input, dec_embeddings, enc_state,
                   enc_output, seq_length, rnn_size, num_layers,
                   q_word2int, keep_prob, batch_size):

    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob)

        decode_cell = tf.nn.rnn_cell.MultiRNNCell([drop]*num_layers)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                                   enc_output,
                                                                   memory_sequence_length=seq_length)

        decode_cell = tf.contrib.seq2seq.AttentionWrapper(decode_cell,
                                                          attention_mechanism,
                                                          attention_layer_size=rnn_size)

        # for training
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, seq_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, helper, enc_state)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        train_logits = outputs.rnn_output

        # for inference
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,tf.fill([batch_size],'<GO>') ,q_word2int('<EOS>'))
        decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, helper, enc_state)
        outputs,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
        infer_logits = outputs.predicted_ids

    return train_logits, infer_logits


def pad_sentence_batch(sentence_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [word2int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, batch_size):
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, q_word2int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, a_word2int))
        yield pad_questions_batch, pad_answers_batch


# -------------------------------------
# Initialize Parameters
# -------------------------------------

# parameters for tensorflow
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

# parameters for pre-processing
min_length = 2
max_length = 40
freq_threshold = 5


# Load data
q_train = json.load(open('/Users/luchen/Documents/TrueAI/train_q_final.json'))
a_train = json.load(open('/Users/luchen/Documents/TrueAI/train_a_final.json'))
q_word2int = json.load(open('/Users/luchen/Documents/TrueAI/train_q_word2int.json'))
a_word2int = json.load(open('/Users/luchen/Documents/TrueAI/train_a_word2int.json'))

# validation data
validate_data = json.load(open('/Users/luchen/Downloads/sample_dataset/valid/dialogues_task.json'))

q_valid, a_valid = split_q_a(validate_data)
q_valid = apply_cleaning(q_valid)
a_valid = apply_cleaning(a_valid)
[q_valid, a_valid] = filter_data(q_valid, a_valid, min_length, max_length)

for i in range(len(a_valid)):
    a_valid[i] += ' <EOS>'

q_valid = convert_words2int(q_valid, q_word2int)
a_valid = convert_words2int(a_valid, a_word2int)

with open('/Users/luchen/Documents/TrueAI/valid_q_final.json','w') as f:
    json.dump(q_valid, f, ensure_ascii=False)

with open('/Users/luchen/Documents/TrueAI/validn_a_final.json','w') as f:
    json.dump(a_valid, f, ensure_ascii=False)

# -----------------------------------------------
# Set up Tensorflow
# -----------------------------------------------

# Load session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Set up inputs
# input_data, targets, lr, keep_prob = model_inputs_2(batch_size, max_length)
input_data, targets, lr, keep_prob = model_inputs()

# Sequence length is max sentence length for each batch
sequence_length = tf.placeholder_with_default(
    max_length,
    None,
    name='sequence_length')
# For sequence loss
input_shape = tf.shape(input_data)

# Create training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]),
    targets,
    keep_prob,
    batch_size,
    sequence_length,
    len(a_word2int),
    len(q_word2int),
    encoding_embedding_size,
    decoding_embedding_size,
    rnn_size,
    num_layers,
    q_word2int)

with tf.name_scope("optimization"):
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for
                        grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# Run Tensorflow

display_step = 100
stop_early = 0
stop = 5
validation_check = ((len(q_train))//batch_size//2)-1
total_train_loss = 0
summary_valid_loss = []

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(q_train, a_train, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs,
                          batch_i,
                          len(q_train) // batch_size,
                          total_train_loss / display_step,
                          batch_time * display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in \
                    enumerate(batch_data(q_valid, a_valid, batch_size)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: learning_rate,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(q_valid) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!')
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break

    if stop_early == stop:
        print("Stopping Training.")
        break


