from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM, Input, RepeatVector
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import random
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pandas as pd

### sequence reversing with LSTM with attention ###
###################################################

### sequence generator parameters ###
NUMBER_OF_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 20

### MODEL PARAMETERS ###
LSTM_SIZE_ENCODER = 128
LSTM_SIZE_DECODER = 128

## LEARNING PARAMETERS ###
BATCH_SIZE = 64

#pretentious words
words = "cosmos existence time space infinity is evolution virtual dimension"

#input
sentences = []
#"labels" - reversed sentences
reverse_sentences = []

#generating sentences
for i in range(NUMBER_OF_SENTENCES):
    length = random.randint(1,MAX_SENTENCE_LENGTH)
    sentence = [np.random.choice(np.array(words.split())) for i in range(length)]
    reverse_sentence = sentence[::-1]
    sentences.append(" ".join(sentence))
    reverse_sentences.append(" ".join(reverse_sentence))

#tokenization turning to integers    
docs = sentences
t = Tokenizer()
t.fit_on_texts(docs)
sequences = t.texts_to_sequences(docs)
reverse_sequences = t.texts_to_sequences(reverse_sentences)
#adding padding
#sequences padded from left, reversed sequences padded from right
pad_sequences = sequence.pad_sequences(sequences)
pad_reverse_sequences = sequence.pad_sequences(reverse_sequences,padding = "post")
#one hot encoding reversed sentences
pad_onehot_reverse_sequences = to_categorical(pad_reverse_sequences)
data_x = pad_sequences
data_y = pad_onehot_reverse_sequences

#tes-validation split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( data_x, data_y, test_size=0.2, random_state=42)
#dictionary_size AKA number of classes
dictonary_size = len(t.word_index) + 1

# Parameters FOR LSTM:
# Input gate: input, previous output, and bias.
ix = tf.Variable(tf.truncated_normal([LSTM_SIZE_ENCODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
im = tf.Variable(tf.truncated_normal([LSTM_SIZE_DECODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
ib = tf.Variable(tf.zeros([1, LSTM_SIZE_DECODER]),dtype=tf.float32)
# Forget gate: input, previous output, and bias.
fx = tf.Variable(tf.truncated_normal([LSTM_SIZE_ENCODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
fm = tf.Variable(tf.truncated_normal([LSTM_SIZE_DECODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
fb = tf.Variable(tf.zeros([1, LSTM_SIZE_DECODER]), dtype=tf.float32)
# Memory cell: input, state and bias.                             
cx = tf.Variable(tf.truncated_normal([LSTM_SIZE_ENCODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
cm = tf.Variable(tf.truncated_normal([LSTM_SIZE_DECODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
cb = tf.Variable(tf.zeros([1, LSTM_SIZE_DECODER]),dtype=tf.float32)
# Output gate: input, previous output, and bias.
ox = tf.Variable(tf.truncated_normal([LSTM_SIZE_ENCODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
om = tf.Variable(tf.truncated_normal([LSTM_SIZE_DECODER, LSTM_SIZE_DECODER], -0.1, 0.1),dtype=tf.float32)
ob = tf.Variable(tf.zeros([1, LSTM_SIZE_DECODER]),dtype=tf.float32)

#initial state for scan function
init_state = tf.placeholder(shape=[2, None, LSTM_SIZE_DECODER], dtype=tf.float32)

#Attention weights parameters
Wcm = tf.Variable(tf.truncated_normal([LSTM_SIZE_DECODER,MAX_SENTENCE_LENGTH], -0.1, 0.1),dtype=tf.float32)

#Wym = tf.Variable(tf.truncated_normal([LSTM_SIZE_ENCODER,1], -0.1, 0.1),dtype=tf.float32)

# def calculate_attention(i):
# return tf.matmul(i,Wym)

# Definition of the cell computation.
def lstm_step(prev, i):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    o, state = tf.unstack(prev)
    #import pdb; pdb.set_trace() 
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
     
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return tf.stack([output_gate * tf.tanh(state), state])


x_ = Input(shape = (MAX_SENTENCE_LENGTH, ))
y_ = tf.placeholder(tf.float32, shape = (None,MAX_SENTENCE_LENGTH,dictonary_size))
embed = Embedding(dictonary_size,20)(x_)
lstm_encoder_hidden = LSTM(LSTM_SIZE_ENCODER)(embed)
hidden_repeated  = RepeatVector(MAX_SENTENCE_LENGTH)(lstm_encoder_hidden)

#lets use our own cell

states = tf.scan(lstm_step, tf.transpose(hidden_repeated, [1,0,2]), initializer=init_state)
lstm_decoder = tf.transpose(states, [1,2,0,3])[0]

#lstm_decoder = LSTM(128,return_sequences=True)(hidden_repeated)

#tensorflow CODE can we?

dense = Dense(dictonary_size)(lstm_decoder)
#softmax = Activation('softmax')(dense)
cross_entropy_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits = dense,
labels = y_)
cross_entropy = tf.reduce_mean(cross_entropy_)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

#metrics
correct_prediction = tf.equal(tf.argmax(y_,2), tf.argmax(dense, 2))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

#connect keras with session
sess = tf.Session()
set_session(sess)
#initialize variables
sess.run(tf.global_variables_initializer())

#computations
losses = []
accuraccies = []
for i in range(30000):
    offset = (i * BATCH_SIZE) % (X_train.shape[0] - BATCH_SIZE)
    _train_x = X_train[offset:(offset + BATCH_SIZE)]
    _train_y = y_train[offset:(offset + BATCH_SIZE)]
    sess.run([train_step],feed_dict={x_: _train_x,
                                  y_: _train_y,
                                  init_state : np.zeros([2, BATCH_SIZE, LSTM_SIZE_DECODER])})
    if (i % 100 == 0):
        loss,acc = sess.run([cross_entropy,accuracy],feed_dict={x_: _train_x,
                                  y_: _train_y,
                                  init_state : np.zeros([2, BATCH_SIZE, LSTM_SIZE_DECODER])})
        print ("Iter " + str(i) + ", Minibatch Loss= " + \
        "{:.6f}".format(loss) + ", Training Accuracy= " + \
        "{:.5f}".format(acc))
        
        loss,acc = sess.run([cross_entropy,accuracy],feed_dict={x_: X_test,
                                  y_: y_test,
                                  init_state : np.zeros([2, X_test.shape[0], LSTM_SIZE_DECODER])})
        print ("Iter " + str(i) + ", Test Loss= " + \
        "{:.6f}".format(loss) + ", Test Accuracy= " + \
        "{:.5f}".format(acc))
        losses.append(loss)
        accuraccies.append(acc)

results = pd.DataFrame({"accuracy" : accuraccies, "loss" : losses})
results.to_pickle("results_lstm.pkl")