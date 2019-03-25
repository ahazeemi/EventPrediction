
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Dense
import csv
from nltk.corpus import stopwords
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import os

'''
Hierarchical LSTM Model for Event Prediction through w2e dataset
The model is able to learn two level dependencies
1) Within an event
2) Across Events
'''



def get_event_details(file_name):
    stopwords_set = set(stopwords.words('english'))
    events = []
    total_words = 0
    max_length = 0
    event_chains_lengths = []
    chain_length = 0
    with open(file_name, encoding='utf-8') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            if len(row) == 3:   # i.e. the line does not mark end of an event chain
                event = row[2].strip().split()
                filtered_words = []
                for word in event:
                    word = word.lower()
                    if word not in stopwords_set:
                        filtered_words.append(word)
                total_words += len(filtered_words)
                if len(filtered_words) > max_length:
                    max_length = len(filtered_words)
                filtered_words = ' '.join(filtered_words)
                events.append(filtered_words)
                chain_length += 1
            else:
                if chain_length != 0:
                    event_chains_lengths.append(chain_length)
                chain_length = 0

    average_length = total_words/len(events)
    return events, average_length, max_length, event_chains_lengths


def tokenizer(docs, max_length):
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1

    encoded_docs = t.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded_docs, vocab_size, t


def create_embedding_matrix(glove_file, vocab_size, t):
    print("loading glove model")
    embeddings_index = dict()
    f = open(glove_file,encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def encoded_data_to_onehot(arr, vocab_size):
    one_hot = [0]*vocab_size
    for i in range(0,len(arr)):
        if(arr[i]==0):
            break
        one_hot[arr[i]-1]=1


    return one_hot

def onehot_to_words(arr, reverse_word_map):
    words = []
    for x, val in enumerate(arr):
        #print(val)
        if val >=0.001 :
            #print(val)
            words.append(reverse_word_map[x+1])

    return words

def get_lstm_input_data(padded_docs, event_chains_lengths, sliding_window_length, vocab_size):

    sliding_window_length+=1
    train_x = []
    train_y = []
    index = 0
    for i in range(0, len(event_chains_lengths)):
        temp = []
        for j in range(0, event_chains_lengths[i]):
            temp.append(padded_docs[index])
            index += 1

        chain = []
        label = []
        for j in range(0, len(temp)):
            if j + sliding_window_length > len(temp):
                break
            chain.append(temp[j: j + sliding_window_length - 1])
            label.append(list(map(lambda x: encoded_data_to_onehot(x,vocab_size), temp[j + 1: j + sliding_window_length])))

        train_x.extend(chain)
        train_y.extend(label)

    return np.array(train_x), np.array(train_y)


def main():

    train = True
    test = False

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    event_details, average_length, max_length, event_chains_lengths = get_event_details('topicGroupsChains2.csv')
    padded_events, vocab_size, doc_tokenizer = tokenizer(event_details, max_length)
    x_train, y_train = get_lstm_input_data(padded_events, event_chains_lengths, sliding_window_length=5,vocab_size=vocab_size)

    # text = []
    # reverse_word_map = dict(map(reversed, doc_tokenizer.word_index.items()))
    # with open('prediction.txt', 'w',encoding='utf-8') as f:
    #     for i in range(0, len(y_train)):
    #         temp = []
    #         for j in range(0, len(y_train[i])):
    #             words = onehot_to_words(y_train[i][j], reverse_word_map)
    #             f.write("%s\n" % ' '.join(words))
    #             temp.append(words)
    #
    #         text.append(temp)

    print(x_train.shape)
    print(y_train.shape)



    #
    #
    # max_num_sentences = 3
    # max_sentence_length = max_length
    # emb_dim = 300
    #
    # # Encode each timestep
    # in_sentence = Input(shape=(max_sentence_length,), dtype='int64')
    # embedded_sentence = Embedding(vocab_size, emb_dim, weights=[embedding_matrix],
    #                               trainable=False)(in_sentence)
    # lstm_sentence = LSTM(lstm1_units)(embedded_sentence)
    # encoded_model = Model(in_sentence, lstm_sentence)
    #
    # sequence_input = Input(shape=(max_num_sentences, max_sentence_length), dtype='int64')
    # seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    # seq_encoded = Dropout(0.2)(seq_encoded)
    #
    # # Encode entire sentence
    # seq_encoded = LSTM(lstm2_units)(seq_encoded)7

    if train:
        path = r'D:\glove.6B.100d.txt'
        embedding_matrix = create_embedding_matrix(path, vocab_size, doc_tokenizer)

        lstm1_units = 512
        lstm2_units = 1024

        max_num_sentences = 5
        max_num_words = max_length
        emb_dim = 100

        # embedding_matrix = []
        model = Sequential()
        e = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=max_num_words, trainable=False)
        model.add(TimeDistributed(e, input_shape=(max_num_sentences, max_num_words)))
        model.add(TimeDistributed(LSTM(lstm1_units), input_shape=(max_num_sentences, max_num_words, emb_dim)))
        model.add(LSTM(lstm2_units, return_sequences=True))
        #model.add(Dropout(0.2))
        model.add(Dense(vocab_size, activation='softmax'))
        #plot_model(model, to_file='lstm.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()

        print("starting training")
        model.fit(x_train,y_train, epochs=1, verbose=1)
        model.save('lstm.h5')

    if test:
        model = load_model('lstm.h5')
        predicted = model.predict(x_train)
        reverse_word_map = dict(map(reversed, doc_tokenizer.word_index.items()))
        arr = np.array(predicted)
        print(arr.shape)

        text = []
        with open('prediction.txt', 'w') as f:
            for i in range(0,len(predicted)):
                temp=[]
                for j in range(0,len(predicted[i])):
                    words = onehot_to_words(predicted[i][j],reverse_word_map)
                    f.write("%s\n" % ' '.join(words))
                    temp.append(words)

                text.append(temp)
                print(i)

    #=plot_model(model, to_file='lstm.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    main()