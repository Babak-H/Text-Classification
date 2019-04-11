import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# we need to fit model with sequence of tokens with specific length
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
# normal LSTM/GRU and the Version with Cuda
from keras.layers import Dense, Embedding, GRU, LSTM, CuDNNLSTM, CuDNNGRU, Dropout, Bidirectional
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, rmsprop

# keras wrapper for k-fold cross-validation
from keras.wrappers.scikit_learn import KerasClassifier
# normsl cross validation
from sklearn.model_selection import cross_val_score, train_test_split
# cross validation for hyperparameter tuning
from sklearn.model_selection import GridSearchCV


def plot_model(result):
    acc = result.history['acc']
    val_acc = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    x = range(1, len(acc)+1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label= 'Validation acc')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='validation loss')
    plt.legend()
    

# nltk.download('punkt')
# nltk.download('wordnet')

x_raw = []
y_raw = []

with open("SMSSpamCollection", 'r') as f:
    for line in f:
        y_raw.append(line.split()[0])
        x_raw.append(' '.join(i for i in line.split()[1:]))
        
y = [1 if i=='ham' else 0 for i in y_raw]

print(max(len(s) for s in x_raw))
print(min(len(s)for s in x_raw))
sorted_X = sorted(len(s) for s in x_raw)
print(sorted_X[len(sorted_X) // 2])


# cleaning the text without the tokenizer method :
# import nltk
# from string import punctuation
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
# from nltk.stem import WordNetLemmatizer
# 
# remove_terms = punctuation + '0123456789'
# 
# def preprocessing(text):
#     words = word_tokenize(text)
#     tokens = [w for w in words if w.lower() not in remove_terms]
# #     tokens = [word for word in tokens if word.isalpha()]
#     lemma = WordNetLemmatizer()
#     tokens = [lemma.lemmatize(word) for word in tokens]
#     pre_processed_text = ' '.join(tokens)
#     return pre_processed_text
# 
# x_raw = [preprocessing(sentence) for sentence in x_raw]



tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_raw)
sequences = tokenizer.texts_to_sequences(x_raw)

vocab_size = len(tokenizer.word_index)+1
print(vocab_size)

# divide sum of length of all sequences by number of all sequences to find averge length of each sequence
sum([len(x) for x in sequences]) // len(sequences)

pad = 'post' 
max_len = 25
embedding_size = 100
batch_size = 20
n_epochs = 50
sequences = pad_sequences(sequences, maxlen=max_len, padding=pad, truncating=pad)
sequences.shape

X_train, X_test, y_train, y_test = train_test_split(sequences, y, test_size = 0.2, random_state= 0)

# =============================================================================
##### LSTM MODEL ####
# =============================================================================

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model.add(Dropout(0.8))
model.add(CuDNNLSTM(140, return_sequences=False))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid', name='Classification'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
save_best = ModelCheckpoint('SpamDetection.hdf', save_best_only=True, monitor='val_acc', mode='max')
# callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Uses Automatic Verification Datasets (fastest option)
# model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[callback_early_stopping])
results = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[save_best])

# model.load_weights(filepath='SpamDetection.hdf')
eval_ = model.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(results)


# Uses K-fold cross validation Datasets
# Hyperparameter tuning:
def build_classifier(drop, layers):
     model = Sequential()
     model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
     model.add(Dropout(drop))
     model.add(CuDNNLSTM(layers, return_sequences=False))
     model.add(Dropout(drop))
     model.add(Dense(1, activation='sigmoid', name='Classification'))
     model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
     return model

model = KerasClassifier(build_fn=build_classifier)
# parameters than we want to tune and the values to try
parameters = {'batch_size' : [25, 32, 64], 'epochs' : [40],'drop' : [0.1, 0.2, 0.5], 'layers': [100, 150]}

# create the cross validation object and train it with 10 different folds
grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=2)
grid_search = grid_search.fit(X_train, y)
 
# find best results
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
 
print(best_parameters)
print(best_accuracy)


# =============================================================================
#### GRU MODEL ####
# =============================================================================

model1 = Sequential()
model1.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model1.add(Dropout(0.8))
model1.add(CuDNNGRU(140, return_sequences=False))
model1.add(Dropout(0.86))
model1.add(Dense(1, activation='sigmoid', name='Classification'))
model1.summary()

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
results1 = model1.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2)

eval_ = model1.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(results1)

# =============================================================================
#### Bidirectional LSTM MODEL ####
# =============================================================================

model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model2.add(Dropout(0.8))
model2.add(Bidirectional(CuDNNLSTM(140, return_sequences=False)))
model2.add(Dropout(0.8))
model2.add(Dense(1, activation='sigmoid', name='Classification'))
model2.summary()

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

results2 = model2.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2)

eval_ = model2.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(results2)

# =============================================================================
#### CNN MODEL ####
# =============================================================================

from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

n_epochs = 10
model3 = Sequential()
model3.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model3.add(Conv1D(128, 3, activation='relu'))
model3.add(MaxPool1D(3))
model3.add(Dropout(0.2))
model3.add(Conv1D(128, 3, activation='relu'))
model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))
model3.add(Dense(64, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(32, activation='relu'))
model3.add(Dropout(0.2))
model3.summary()
model3.add(Dense(1, activation='sigmoid'))


model3.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
result3 = model3.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, validation_split=0.2, verbose=1)
eval_ = model3.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy

plot_model(result3)


# =============================================================================
#### CNN MODEL with GLOVE ####
# =============================================================================


# =============================================================================
# corpus : 9000 words

# Results
# 
# Model                Accuracy
# 
# LSTM  :              98.4%
# Bidirectional LSTM : 98.3%
# GRU :                98.1%
# CNN :                98.11%
# =============================================================================








