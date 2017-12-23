import tensorflow as tf
import time
import numpy as np
import json

def model_inputs():
    """Create placeholders for inputs to the model"""
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, [1,1], name='keep_prob')

    return input_data, targets, lr, keep_prob

def seq2seq_model(input_data,
                  target_data,
                  input_dict,
                  target_dict,
                  enc_emb_size,
                  dec_emb_size,
                  batch_size,
                  num_units,
                  seq_length):

    enc_emb_inp = tf.contrib.layers.embed_sequence(input_data,
                                                   len(input_dict),
                                                   enc_emb_size,
                                                   initializer=tf.random_uniform_initializer(-1,1))

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1],
                                   target_dict['<GO>']), ending], 1)

    dec_emb = tf.Variable(tf.random_uniform([len(input_dict)+1, dec_emb_size], -1,1))
    dec_emb_inp = tf.nn.embedding_lookup(dec_emb, dec_input)


    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, enc_emb_inp,
        sequence_length=seq_length, time_major=True, dtype=tf.float32)

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(
        dec_emb_inp, seq_length, time_major=True)
    # Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state)
    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = outputs.rnn_output

    return logits


def pad_sentence_batch(sentence_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [word2int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, q_dict, a_dict, batch_size):
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, q_dict))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, a_dict))
        yield pad_questions_batch, pad_answers_batch


###############
max_length=40
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
##############

q_train = json.load(open('/Users/luchen/Documents/TrueAI/train_q_final.json'))
a_train = json.load(open('/Users/luchen/Documents/TrueAI/train_a_final.json'))
q_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_q_final.json'))
a_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_a_final.json'))

input_dict = json.load(open('/Users/luchen/Documents/TrueAI/train_q_word2int.json'))
target_dict = json.load(open('/Users/luchen/Documents/TrueAI/train_a_word2int.json'))




tf.reset_default_graph()
sess = tf.InteractiveSession()

input_data, target_data, lr, keep_prob = model_inputs()
sequence_length = tf.placeholder(tf.int32, [None,])
input_shape = tf.shape(input_data)

train_logits = seq2seq_model(input_data,
                             target_data,
                             input_dict,
                             target_dict,
                             encoding_embedding_size,
                             decoding_embedding_size,
                             batch_size,
                             rnn_size,
                             sequence_length)

with tf.name_scope("optimization"):
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        target_data,
        tf.ones([batch_size, input_shape[1]]))
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
            batch_data(q_train, a_train, input_dict, target_dict, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            feed_dict={input_data: questions_batch,
             target_data: answers_batch,
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
                           target_data: answers_batch,
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
