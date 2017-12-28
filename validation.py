# Validation!!!!!

# Running an inference graph
q_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_q_final.json'))
a_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_a_final.json'))

a_valid_input = append_SOS(a_valid, a_dict)
a_valid_output = append_EOS(a_valid, a_dict)

[pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length] = test_batch_data(q_valid, a_valid_input, a_valid_output, q_dict, a_dict, batch_size)

with test_graph.as_default():
    test_source_inputs, test_target_inputs, test_target_outputs, test_source_seq_length, test_target_seq_length, \
    test_lr, test_kp = init_placeholders()

    test_logits, _ = build_full_model(test_source_inputs,
                                    test_source_seq_length,
                                    source_vocab_size,
                                    target_vocab_size,
                                    batch_size,
                                    enc_emb_size,
                                    dec_emb_size,
                                    num_units,
                                    False,
                                    sos_id, eos_id,
                                    1,
                                    target_inputs=None,
                                    target_seq_length=None,
                                    num_topics=num_topics)


    with tf.name_scope("optimization"):
        t_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=test_target_outputs, logits=test_logits)
        test_loss = (tf.reduce_sum(t_cross_entropy) / batch_size)
        t_params = tf.trainable_variables()
        t_gradients = tf.gradients(test_loss, t_params)
        t_clipped_gradients, _ = tf.clip_by_global_norm(t_gradients, max_gradient_norm)

        t_optimizer = tf.train.AdamOptimizer(test_lr)
        test_op = t_optimizer.apply_gradients(zip(t_clipped_gradients, t_params))

    test_saver = tf.train.Saver()
    test_initializer = tf.global_variables_initializer()

test_session = tf.Session(graph=test_graph)

test_session.run(test_initializer)
test_saver.restore(test_session, checkpoint)

_,t_loss = test_session.run([test_op, test_loss],
                 feed_dict={test_source_inputs: pad_q_batch,
                            test_target_inputs: pad_a_in_batch,
                            test_target_outputs: pad_a_out_batch,
                            test_source_seq_length: q_length,
                            test_target_seq_length: a_length,
                            test_lr: learning_rate,
                            test_kp: keep_prob})

print(t_loss)
