import tensorflow as tf
import numpy as np
from tensorflow import keras


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, dim, drop_rate):
        super(MultiHead, self).__init__()
        self.h_dim = dim // n_head
        self.n_head = n_head
        self.wk = keras.layers.Dense(n_head * self.h_dim)
        self.wq = keras.layers.Dense(n_head * self.h_dim)
        self.wv = keras.layers.Dense(n_head * self.h_dim)
        self.o_dense = keras.layers.Dense(dim)
        self.drop = keras.layers.Dropout(rate=drop_rate)
        self.attention = None

    def call(self, q, k, v, mask, training):
        _q = self.wk(q)
        _k, _v = self.wk(k), self.wv(v)
        _q = self.split_heads(_q)
        _k = self.split_heads(_k)
        _v = self.split_heads(_v)
        result = self.scale_dot(_q, _k, _v, mask)
        o = self.o_dense(result)
        o = self.drop(o, training=training)
        return o

    def split_heads(self, x):
        # n,step,h*dim
        _x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.h_dim))
        _x = tf.transpose(_x, perm=[0, 2, 1, 3])  # n h step h_dim
        return _x

    def scale_dot(self, q, k, v, mask=None):
        # q,k,v shape:( n h step h_dim)
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)
        o = tf.matmul(self.attention, v)
        o = tf.transpose(o, perm=[0, 2, 1, 3])  # n step h h_dim
        o = tf.reshape(o, (o.shape[0], o.shape[1], -1))
        return o


class PositionwiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super(PositionwiseFFN, self).__init__()
        num = model_dim * 4
        self.l1 = keras.layers.Dense(num)
        self.l2 = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.l1(x, activation=keras.activations.relu)
        o = self.l2(o)
        return o


class EncoderLayer(keras.layers.Layer):
    def __init__(self, n_head, dim, drop_rate):
        super(EncoderLayer, self).__init__()
        self.mul = MultiHead(n_head, dim, drop_rate)
        self.ffn = PositionwiseFFN(dim)
        self.drop = keras.layers.Dropout(rate=drop_rate)
        self.norm = [keras.layers.LayerNormalization(axis=-1) for _ in range(2)]

    def call(self, x, mask, training):  # n step dim
        context = self.mul.call(x, x, x, mask, training)
        o1 = self.norm[0](context + x)
        o2 = self.drop(self.ffn.call(o1))
        o = self.norm[1](o2 + o1)
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, dim, drop_rate, n_layer):
        super(Encoder, self).__init__()
        self.layer = [EncoderLayer(n_head, dim, drop_rate) for _ in range(n_layer)]

    def call(self, x, mask, training):
        for l in self.layer:
            x = l.call(x, mask, training)
        return x


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, dim, drop_rate):
        super(DecoderLayer, self).__init__()
        self.mul = [MultiHead(n_head, dim, drop_rate) for _ in range(2)]
        self.ffn = PositionwiseFFN(dim)
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(3)]
        self.drop = keras.layers.Dropout(rate=drop_rate)

    def call(self, y, x, y_lookahead_pad, x_padding, training):
        o1 = self.mul[0].call(y, y, y, y_lookahead_pad, training)
        o1 = self.ln[0](o1 + y)
        o2 = self.num[1].call(y, x, x, x_padding, training)
        o2 = self.ln[1](o1 + o2)
        o3 = self.ffn.call(o2)
        o3 = self.drop(self.ln[2](o2 + o3))
        return o3


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, dim, drop_rate, n_layer):
        super(Decoder, self).__init__()
        self.layer = [DecoderLayer(n_head, dim, drop_rate) for _ in range(n_layer)]

    def call(self, y, x, y_lookahead_pad, x_padding, training):
        for l in self.layer:
            y = l.call(y, x, y_lookahead_pad, x_padding, training)
        return y


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, step, dim, n_vocab):
        super(PositionEmbedding, self).__init__()
        pos = np.arange(step)[:, None]
        pe = pos / (np.power(1000, 2 * np.arange(dim))[None, :] / dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 0::1] = np.cos(pe[:, 0::1])
        pe = pe[None, :, :]
        self.pe = tf.constant(pe, dtype=tf.float32)
        self.embed = keras.layers.Embedding(input_dim=n_vocab, output_dim=dim,
                                            embeddings_initializer=tf.initializers.random_normal(0, 0.1))

    def call(self, x):
        return self.embed(x) + self.pe


class Transformer(keras.layers.Layer):
    def __init__(self, max_len, n_head, model_dim, vocab_num, drop_rate, n_layer, pad_idx=0):
        super(Transformer, self).__init__()
        self.max_len=max_len
        self.pad_idx = pad_idx
        self.embed = PositionEmbedding(max_len, model_dim, vocab_num)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        self.o = keras.layers.Dense(vocab_num)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.adam = keras.optimizers.Adam(0.01)

    def call(self, x, y, training=None):
        emb_x, emb_y = self.embed(x), self.embed(y)
        pad_mask = self._pad_mask(x)
        encode_z = self.encoder.call(emb_x, pad_mask, training)
        y_out = self.decoder.call(emb_y, encode_z, self._look_ahead_mask(y), pad_mask, training)
        return self.o(y_out)

    def step(self, x, y, training=True):
        with tf.GradientTape as tape:
            logits=self.call(x, y[:,:-1], training)
            bool=tf.math.not_equal(y[:,1:], self.pad_idx)
            loss=tf.reduce_mean(tf.boolean_mask(self.loss(y[:,1:],logits),bool))
        grads=tape.gradient(loss,self.trainable_variables)
        self.adam.apply_gradients(zip(grads,self.trainable_variables))
        return loss,logits



    def _pad_bool(self, seq, pad_idx):
        return tf.math.not_equal(seq, pad_idx)

    def _pad_mask(self, seq):
        # seq: n,step        n,h,step,step
        pad_mask = tf.cast(self._pad_bool(seq, self.pad_idx), tf.float32)
        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
        return pad_mask

    def _look_ahead_mask(self, seq):
        mask = 1 - tf.linalg.band_part(tf.ones(len(seq), len(seq)), -1, 0)
        bool = self._pad_bool(seq, self.pad_idx)
        mask = tf.where(bool[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask

    def translate(self,src,v2i,i2v):
        src_pad=pad_zero(src,self.max_len)
        src_emd=self.embed.call(src_pad)
        encoder_z=self.encoder.call(src_emd,mask=self._pad_mask(src_pad),training=False)
        tgt=pad_zero([[v2i['<GO>']] for _ in range(len(src))],self.max_len+1)
        tgti=0
        while True:
            tgt_emb=self.embed(tgt[:,:-1])
            decoder_z=self.decoder.call(tgt_emb,encoder_z,y_lookahead_pad=self._look_ahead_mask(tgt[:,:-1]),
                                     x_padding=self._pad_mask(src_pad),training=False)

            logits=self.o(decoder_z)[:,tgti,:].numpy()
            pre=np.argmax(logits,axis=1)
            tgt[:,tgti]=pre
            tgti+=1
            if tgti>=self.max_len:
                break
        return ["".join([i2v[i] for i in tgt[j,1:]]) for j in range(len(src))]


def pad_zero(seqs, max_len):
    matrix = np.zeros((len(seqs), max_len))
    for i, seq in enumerate(seqs):
        matrix[i, :len(seq)] = seq
    return matrix
