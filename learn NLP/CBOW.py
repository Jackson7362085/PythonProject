from tensorflow import keras
import tensorflow as tf
import numpy as np
import itertools
corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",
    "1 2 3 4 5 6 7 8 9",
    "1 9 4 6 5 9 3 6 3 2",
    "1 3 2 6 1 8 9 9 0 1 1",
    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
    "a b c d e f g h i g k l m n o p q r s t u v w x y z"
]
class Dataset():
    def __init__(self,x,y,i2v,v2i):
        self.x=x
        self.y=y
        self.i2v=i2v
        self.v2i=v2i
        self.vocab=v2i.keys()
    def sample(self,n):
        idx=np.random.randint(0,len(self.x),n)
        x=self.x[idx]
        y=self.y[idx]
        return x,y
    def num(self):
        return len(self.v2i)

def process_corpus(corpus,skip_window=2,method='cbow'):
    word=[d.split(" ")for d in corpus]
    word=np.array(list(itertools.chain(*word)))
    vocab,count=np.unique(word,return_counts=True)
    vocab=vocab[np.argsort(count)[::-1]]
    v2i={v:i for i,v in enumerate(vocab)}
    i2v={i:v for v,i in v2i.items()}
    js=[i for i in range(-skip_window,skip_window+1) if i!=0]
    data=[]
    if method.lower()=='cbow':
        for c in corpus:
            words=c.split(" ")
            idx=[v2i[w] for w in words]
            for i in range(skip_window,len(words)-skip_window):
                context=[]
                for j in js:
                    context.append(idx[i+j])
                context.append(idx[i])
                data.append(context)

    data=np.array(data)
    x,y=data[:,:-1],data[:,-1]
    return Dataset(x,y,i2v,v2i)
class CBOW(keras.Model):
    def __init__(self,v_dim,emb_dim):
        super().__init__()
        self.v_dim=v_dim
        self.emd_dim=emb_dim
        self.embeding=keras.layers.Embedding(v_dim,emb_dim,embeddings_initializer=keras.initializers.RandomNormal(0.,0,1))
        self.nec_w=self.add_weight(name='nec_w',shape=[v_dim,emb_dim],initializer=keras.initializers.TruncatedNormal(0.,0.1))
        self.nec_b=self.add_weight(name='nec_b', shape=(v_dim,),initializer=keras.initializers.constant(0.1))
        self.opt=keras.optimizers.Adam(0.01)
    def call(self,x):
        o=self.embeding(x)
        o=tf.reduce_mean(o,axis=1)
        return o
    def loss(self,x,y):
        o=self.call(x)
        return tf.reduce_mean(tf.nn.nce_loss(weights=self.nec_w,biases=self.nec_b,
                                             labels=tf.expand_dims(y,axis=1),inputs=o,num_sampled=5,
                                             num_classes=self.v_dim))

    def step(self,x,y):
        with tf.GradientTape() as tape:
            loss=self.loss(x,y)
            grads=tape.gradient(loss,self.trainable_variables)
            self.opt.apply_gradients(zip(grads,self.trainable_variables))
        return loss.numpy()

def train(model,data):
    for i in range(2000):
        x,y=data.sample(8)
        loss=model.step(x,y)
        if i%200==0:
            print("step:{} loss:{}".format(i,loss))

if __name__=="__main__":

    data=process_corpus(corpus,2,'cbow')
    model=CBOW(data.num(),2)
    train(model,data)
    print(model.embeding.get_weights())
    print(model.embeding.get_weights()[0],'*****')


