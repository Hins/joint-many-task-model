import tensorflow as tf
import numpy as np
from my_lstm import MyLSTM
from helper import *
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
LSTMStateTuple = tf.compat.v1.nn.rnn_cell.LSTMStateTuple


class JMT:

    def __init__(self, dim, reg_lambda, lr=0.01):
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.lr = lr

    def load_data(self):
        data = np.load('data/data.npz', allow_pickle=True)['data'].item()
        self.sent = data['word_level']['sent'].values
        self.pos = data['word_level']['pos']
        self.i2p = data['word_level']['i2p']
        self.i2c = data['word_level']['i2c']
        self.chun = data['word_level']['chunk']
        self.sent1 = data['sent_level']['sent1']
        self.sent2 = data['sent_level']['sent2']
        self.i2e = data['sent_level']['i2e']
        self.rel = data['sent_level']['rel']
        self.ent = data['sent_level']['entailment']
        self.w2i = data['w2i']

        self.vec = np.array(data['vec'] + [[0] * 300])
        self.max_length = max([len(i) for i in self.sent])
        print('***Data loaded***')

    def build_model(self):
        ''' Builds the whole computational graph '''
        def sentence_op(inputs, t_pos, t_chunk):
            with tf.compat.v1.variable_scope('pos', reuse=tf.compat.v1.AUTO_REUSE):
                embeddings = tf.compat.v1.constant(self.vec, dtype=tf.compat.v1.float32)
                embeds = tf.compat.v1.nn.embedding_lookup(embeddings, inputs)
                fw_lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                bw_lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm, inputs=embeds,
                                                             sequence_length=length(embeds), dtype=tf.compat.v1.float32)
                concat_outputs = tf.compat.v1.concat([outputs[0] , outputs[1]],2)
                y_pos = activate(concat_outputs, [
                    self.dim * 2, len(self.i2p)], [len(self.i2p)])
                t_pos_sparse = tf.compat.v1.one_hot(
                    indices=t_pos, depth=len(self.i2p), axis=-1)
                loss = cost(y_pos, t_pos_sparse)
                loss += tf.compat.v1.reduce_sum([self.reg_lambda * tf.compat.v1.nn.l2_loss(x)
                                       for x in tf.compat.v1.trainable_variables()])
                optimize_op = tf.compat.v1.train.AdagradOptimizer(
                    self.lr).minimize(loss)

            with tf.compat.v1.variable_scope('chunk', reuse=tf.compat.v1.AUTO_REUSE):
                inputs1 = tf.compat.v1.concat([embeds, concat_outputs, y_pos], 2)
                fw_lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                bw_lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                outputs1, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm, inputs=inputs1,
                                                              sequence_length=length(embeds), dtype=tf.compat.v1.float32)
                concat_outputs1 = tf.compat.v1.concat(outputs1, 2)
                y_chunk = activate(concat_outputs1, [
                                   self.dim * 2, len(self.i2c)], [len(self.i2c)])
                t_chunk_sparse = tf.compat.v1.one_hot(
                    indices=t_chunk, depth=len(self.i2c), axis=-1)
                loss1 = cost(y_chunk, t_chunk_sparse)
                loss1 += tf.compat.v1.reduce_sum([self.reg_lambda * tf.compat.v1.nn.l2_loss(x)
                                        for x in tf.compat.v1.trainable_variables()])
                optimize_op1 = tf.compat.v1.train.AdagradOptimizer(
                    self.lr).minimize(loss1)

            with tf.compat.v1.variable_scope('relatedness', reuse=tf.compat.v1.AUTO_REUSE):
                with tf.compat.v1.variable_scope('layer_1'):
                    inputs2 = tf.compat.v1.concat([embeds, concat_outputs1, y_pos, y_chunk], 2)
                    fw_lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                    bw_lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                    outputs2, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm, inputs=inputs2,
                                                                  sequence_length=length(embeds), dtype=tf.compat.v1.float32)
                    concat_outputs2 = tf.compat.v1.concat(outputs2, 2)
                with tf.compat.v1.variable_scope('layer_2'):
                    inputs3 = tf.compat.v1.concat([embeds, concat_outputs2, y_pos, y_chunk], 2)
                    fw_lstm1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                    bw_lstm1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                    outputs3, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm1, cell_bw=bw_lstm1, inputs=inputs3,
                                                                  sequence_length=length(embeds), dtype=tf.compat.v1.float32)
                    concat_outputs3 = tf.compat.v1.concat(outputs3, 2)
                    s = tf.compat.v1.reduce_max(concat_outputs3, reduction_indices=1)

                with tf.compat.v1.variable_scope('layer_3'):
                    inputs4 = tf.compat.v1.concat([embeds, concat_outputs3, y_pos, y_chunk], 2)
                    fw_lstm2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                    bw_lstm2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.dim, state_is_tuple=True)
                    outputs4, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm2, cell_bw=bw_lstm2, inputs=inputs4,
                                                                  sequence_length=length(embeds), dtype=tf.compat.v1.float32)
                    concat_outputs4 = tf.compat.v1.concat(outputs3, 2)
                    s1 = tf.compat.v1.reduce_max(concat_outputs4, reduction_indices=1)

            return s, s1, optimize_op, optimize_op1, loss, loss1, y_pos, y_chunk

        with tf.compat.v1.variable_scope('sentence', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            self.inp = tf.compat.v1.placeholder(
                shape=[None, self.max_length], dtype=tf.compat.v1.int32, name='input')
            self.t_p = tf.compat.v1.placeholder(
                shape=[None, self.max_length], dtype=tf.compat.v1.int32, name='t_pos')
            self.t_c = tf.compat.v1.placeholder(
                shape=[None, self.max_length], dtype=tf.compat.v1.int32, name='t_chunk')
            s11, s12, self.optimize_op, self.optimize_op1, self.loss, self.loss1, self.y_pos, self.y_chunk = sentence_op(
                self.inp, self.t_p, self.t_c)
            scope.reuse_variables()
            self.inp1 = tf.compat.v1.placeholder(
                shape=[None, self.max_length], dtype=tf.compat.v1.int32, name='input1')
            s21, s22 = sentence_op(self.inp1, self.t_p, self.t_c)[:2]

            d = tf.compat.v1.concat([tf.compat.v1.abs(tf.compat.v1.subtract(s11, s21)), tf.compat.v1.multiply(s11,
                                                                                                              s21)], 1)
            d1 = tf.compat.v1.concat([tf.compat.v1.subtract(s12, s22), tf.compat.v1.multiply(s12, s22)], 1)
        with tf.compat.v1.variable_scope('relation', reuse=tf.compat.v1.AUTO_REUSE):
            self.y_rel = tf.compat.v1.squeeze(
                activate(d, [self.dim * 4, 1], [1], activation=tf.compat.v1.nn.relu))
            self.t_rel = tf.compat.v1.placeholder(shape=[None], dtype=tf.compat.v1.float32)
            self.loss2 = rmse_loss(self.y_rel, self.t_rel)
            self.loss2 += tf.compat.v1.reduce_sum([self.reg_lambda * tf.compat.v1.nn.l2_loss(x)
                                         for x in tf.compat.v1.trainable_variables()])
            self.optimize_op2 = tf.compat.v1.train.AdagradOptimizer(
                self.lr).minimize(self.loss2)

        with tf.compat.v1.variable_scope('entailment', reuse=tf.compat.v1.AUTO_REUSE):
            self.t_ent = tf.compat.v1.placeholder(shape=[None], dtype=tf.compat.v1.int32)
            t_ent_sparse = tf.compat.v1.one_hot(indices=self.t_ent, depth=3, axis=-1)
            self.y_ent = activate(d1, [self.dim * 4, 3], [3])
            self.loss3 = - tf.compat.v1.reduce_mean(t_ent_sparse * tf.compat.v1.log(self.y_ent))
            self.loss3 += tf.compat.v1.reduce_sum([self.reg_lambda * tf.compat.v1.nn.l2_loss(x)
                                         for x in tf.compat.v1.trainable_variables()])
            self.optimize_op3 = tf.compat.v1.train.AdagradOptimizer(
                self.lr).minimize(self.loss3)
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        print('***Model built***')

    def get_predictions(self, sess, graph, task_desc):
        resp = []
        if 'chunk' in task_desc:
            for sub_task in task_desc["chunk"]:
                inp = sub_task.split()
                inputs = [[self.w2i[i] if i in self.w2i else 0 for i in inp] +
                          [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
                preds = sess.run(self.y_chunk, {self.inp: inputs})[0]
                preds = np.argmax(preds, axis=-1)[:len(inp)]
                preds = [self.i2c[i] for i in preds]
                resp.append(preds)
        return resp

    def train_model(self, graph, train_desc, resume=False):
        saver = tf.compat.v1.train.Saver()
        batch_size = train_desc['batch_size']
        with tf.compat.v1.Session(graph=graph) as sess:
            if resume:
                saver = tf.compat.v1.train.import_meta_graph('saves/model.ckpt.meta')
                saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./saves'))
                print('training resumed')
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
            if 'pos' in train_desc:
                print('***Training POS layer***')
                for i in range(train_desc['pos']):
                    a, b, c = get_batch_pos(self, batch_size)
                    _, l = sess.run([self.optimize_op, self.loss],
                                    {self.inp: a, self.t_p: b})
                    if i % 50 == 0:
                        print(l)
                        saver.save(sess, 'saves/model.ckpt')
            if 'chunk' in train_desc:
                print('***Training chunk layer***')
                for i in range(train_desc['chunk']):
                    a, b, c = get_batch_pos(self, batch_size)
                    _, l1 = sess.run([self.optimize_op1, self.loss1], {
                        self.inp: a, self.t_p: b, self.t_c: c})
                    if i % 50 == 0:
                        print(l1)
                        saver.save(sess, 'saves/model.ckpt')
            if 'relatedness' in train_desc:
                print('***Training semantic relatedness***')
                for i in range(train_desc['relatedness']):
                    a, b, c, _ = get_batch_sent(self, batch_size)
                    _, l2 = sess.run([self.optimize_op2, self.loss2], {self.inp: a,
                                                                       self.inp1: b, self.t_rel: c})
                    if i % 50 == 0:
                        print(l2)
                        saver.save(sess, 'saves/model.ckpt')
            if 'entailment' in train_desc:
                print('***Training semantic entailment***')
                for i in range(train_desc['entailment']):
                    a, b, _, c = get_batch_sent(self, batch_size)
                    _, l3 = sess.run([self.optimize_op3, self.loss3], {self.inp: a,
                                                                       self.inp1: b, self.t_ent: c})
                    if i % 50 == 0:
                        print(l3)
                        saver.save(sess, 'saves/model.ckpt')
