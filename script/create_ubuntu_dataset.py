from __future__ import print_function
from nltk.stem import *
from nltk.corpus import stopwords
import pandas as pd
import re
import collections
import cPickle
import random
import numpy as np
import tensorflow as tf
import matplotlib
import argparse

folder = '../data/'


# Read the original file where the data was saved
def read_file(filename):
    data = pd.read_csv(folder + filename)
    return data


def get_name_file(name_file,
                  folder=folder,
                  dataset="",
                  version=""):
    return folder + name_file + ("" if dataset == "" else "_") + dataset + (
        "" if version == "" else "_") + version + ".p"


# Tokenize a text
def tokenizer(text, stemmer):
    # Remove english stop words
    stop = stopwords.words('english')
    # Keep track of emoticones
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    # '\W' retire tous les non mots + passe tout en miniscule + ajoute les emoticones sans le nez - (:-) -> :) )
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    # Split word based on space
    tokenized = [w for w in text.split() if w not in stop]
    #  Transform each word into into its lexical root

    tokenized = [stemmer.stem(w) for w in tokenized]
    tokenized = tokenized[1:]
    return tokenized


def tokenizing_file(name, nb_lignes):
    data = read_file(name)  # read the file
    # tokenize all entries
    print('Tokenize sentences')
    stemmer = PorterStemmer()
    for index, _ in data.iterrows():
        if index < nb_lignes:
            for column in data:
                if column != 'Label':
                    data.set_value(index, column, tokenizer(str(data[column][index]), stemmer))

    # save the new tokens
    data = data.drop(data.index[range(nb_lignes, data.shape[0])])
    data.to_pickle(get_name_file('tokenized_sentences', dataset="u_corpus"))
    print('saved tokens')


def get_columns(name):
    if name == "man_page":
        list_column = ['description%d' % i for i in range(1, 5)]
        list_column.append('name')
        list_column.append('function')
        return list_column
    elif name == "u_corpus":
        return ['Context', 'Utterance']


# Create a dictionnary
def create_dictionnary(data,
                       name,
                       nb_lignes,
                       size_most_common_words,
                       man_file=True
                       ):
    print('Count number of words')
    all_words = []
    columns_to_consider = get_columns(name)

    def collect_words(columns, _data):
        for column in columns:
            for index, _ in _data.iterrows():
                words = _data[column][index]
                for word in words:
                    all_words.append(word)
                if index > nb_lignes:
                    break

    collect_words(columns_to_consider, data)
    print("Number of words in %s: %s" % (name, len(all_words)))

    if man_file:
        data_man = pd.read_csv(folder + 'man_tokenized_sentences.csv')
        columns_to_consider = get_columns('man_page')
        collect_words(columns_to_consider, data_man)

    print("Number of words in %s: %s" % (name, len(all_words)))

    count = collections.Counter(all_words).most_common(size_most_common_words)
    print('Creating dictionnaries')
    # Create index to word dictionnaries
    rdic = []
    # For words that are not the most seen
    rdic.append('UNK')
    # For appending at the end of each file
    for i in count:
        rdic.append(i[0])
    cPickle.dump(rdic, open(get_name_file('rdic', dataset=name), 'wb'))
    # Create word to index dictionnaries
    dic = {}
    for i in range(len(rdic)):
        dic[rdic[i]] = i
    cPickle.dump(dic, open(get_name_file('dic', dataset=name), 'wb'))

    def transform_tokens_into_index(_data, _name):
        windows_size = 2
        columns = get_columns(_name)
        for column in columns:
            for index, _ in _data.iterrows():
                words = _data[column][index]
                transform_data = []
                for word in words:
                    if word in dic:
                        index1 = dic[word]
                    else:
                        index1 = 0
                    transform_data.append(index1)
                _data.set_value(index, column, transform_data)
                if index > nb_lignes:
                    break
        _data = _data.drop(_data.index[range(nb_lignes, _data.shape[0])])
        # save as pickle
        _data.to_pickle(get_name_file('word_number', dataset=_name))

    transform_tokens_into_index(data, name)
    if man_file != None:
        transform_tokens_into_index(data_man, "man_page")

    print('Dictionnaries created')


def generate_batch_w2v(data,
                       size,
                       window_size=2):
    skip_gram_pair = []
    x_data = []
    y_data = []
    for i in range(size):
        cbow_pairs = []
        words = data['Utterance' if random.randint(0, 2) != 0 else 'Context'][random.randint(0, data.shape[0] - 1)]
        # print("size of words", len(words))
        if len(words) - window_size - 1 < window_size:
            i -= 1
            continue

        index = random.randint(window_size, len(words) - window_size - 1)
        cbow_pairs.append([[words[index + k] for k in range(-window_size, window_size + 1) if k != 0], words[index]])
        for pair in cbow_pairs:
            for w in pair[0]:
                skip_gram_pair.append([pair[1], w])
    r = np.random.choice(range(len(skip_gram_pair)), size, replace=False)
    for i in r:
        x_data.append(skip_gram_pair[i][0])
        y_data.append([skip_gram_pair[i][1]])
    return x_data, y_data


def build_model(data,
                embedding_size=24,  # dimensions of the vector representing an embeding
                num_sampled=64,  # Number of negative examples to sample.
                number_of_step=30000,  # Number of step
                window_size=2,  # where to look
                batch_size=128,  # number of sample per batch
                plot_only=2000,  # number of words to plot
                size_most_common_words=10000,  # size of the dictionnary
                voc_size=10002
                ):
    print('Building model')
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_label = tf.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Xavier init for weights
    fan_in = voc_size
    fan_out = embedding_size
    low = -4 * np.sqrt(6.0 / (fan_in + fan_out))  # 4 for sigmoid
    high = 4 * np.sqrt(6.0 / (fan_in + fan_out))
    nce_weight = tf.Variable(tf.random_uniform([voc_size, embedding_size], minval=low, maxval=high, dtype=tf.float32))
    # init bias
    nce_biases = tf.Variable(tf.zeros([voc_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weight, nce_biases, embed, train_label, num_sampled, voc_size))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # saved the mean nce_loss
    lossV = {}
    print('Starting session')
    with tf.Session() as sess:
        min_lost = 10000000.0
        tf.initialize_all_variables().run()

        for step in range(number_of_step):
            batch_inputs, batch_labels = generate_batch_w2v(data, batch_size, window_size)
            sess.run([train_op], feed_dict={train_inputs: batch_inputs, train_label: batch_labels})
            trained_embedding = embeddings.eval()
            if step % 500 == 0:
                loss_val = sess.run([loss], feed_dict={train_inputs: batch_inputs,
                                                       train_label: batch_labels})
                lossV[step] = loss_val[0]
                print("Iter %d: loss of %.5f" % (step, loss_val[0]))
                if min_lost > loss_val[0]:
                    print('Saving current state')
                    min_lost = loss_val
                    # Save the best model
                    # np.save('word2vec', trained_embedding)
                    # Save the best embeddings corresponding
                    final_embeddings = normalized_embeddings.eval()
                    # Save in file the best embedding
                    cPickle.dump(final_embeddings, open(get_name_file('final_embeddings'), 'wb'))


def sentences_to_embedding(embeddings=None, reverse_dictionnary=None, name='u_corpus'):
    if embeddings is None:
        embeddings = cPickle.load(open(get_name_file('final_embeddings'), 'rb'))
    if reverse_dictionnary is None:
        reverse_dictionnary = cPickle.load(open(get_name_file('rdic', dataset=name), 'rb'))

    labels = {i: embeddings[i] for i in range(len(reverse_dictionnary))}
    cPickle.dump(labels, open(get_name_file('index_to_embedding', dataset=name), 'wb'))

    labels = {reverse_dictionnary[i]: embeddings[i] for i in range(len(reverse_dictionnary))}
    cPickle.dump(labels, open(get_name_file('word_to_embedding', dataset=name), 'wb'))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--ubuntu_file', type=str, default='output.csv', help='file containing the ubuntu corpus')
    parser.add_argument('--nb_words', type=int, default=10000, help='Size of the vocabulary')
    parser.add_argument('--nb_examples', type=int, default=1000, help='Nb of rows to consider in the ubuntu corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of a batch of word2vec')
    parser.add_argument('--embedding_size', type=int, default=28, help='dimensions of the embeddings')
    parser.add_argument('--window_size', type=int, default=2, help='size of the windows')
    parser.add_argument('--folder_data', type=str, default='../data/', help='Location to load and save data')

    args = parser.parse_args()

    folder = args.folder_data

    # tokenizing_file(args.ubuntu_file, nb_lignes=args.nb_examples)

    # Read the tokenize pickle
    data = pd.read_pickle(get_name_file('tokenized_sentences', dataset="u_corpus"))
    # Create dictionnaries and replace all sentences by their index in the dictionnary
    create_dictionnary(data, 'u_corpus', size_most_common_words=args.nb_words, nb_lignes=args.nb_examples)
    data = pd.read_pickle(get_name_file('word_number', dataset='u_corpus'))
    build_model(data,
                size_most_common_words=args.nb_words,
                batch_size=args.batch_size,
                embedding_size=args.embedding_size,
                voc_size=args.nb_words + 2)
    sentences_to_embedding()


if __name__ == '__main__':
    main()
