import cPickle
import numpy as np
import theano
from hash_tables import KnowledgeDataBase
import argparse


def get_name_file(name_file,
                  folder='../data/',
                  dataset="",
                  version=""):
    return folder + name_file + ("" if dataset == "" else "_") + dataset + (
        "" if version == "" else "_") + version + ".p"


def create_batch(data,
                 index_to_embedding,
                 rdic,
                 dico,
                 number_of_examples,
                 start,
                 end,
                 size_embeddings,
                 max_length,
                 database):
    # Create dictionnary
    dic = {}
    dic['y'] = np.zeros((number_of_examples,)).astype(theano.config.floatX)
    for index in ['c', 'r', 'm']:
        dic[index] = np.zeros((number_of_examples, max_length, size_embeddings)).astype(theano.config.floatX)
    for index in ['c_mask', 'r_mask', 'm_mask']:
        dic[index] = np.ones((number_of_examples, max_length)).astype(theano.config.floatX)
    """
    1. Generate pair of context and its correct response
    """
    for index in range(start, end):
        el = index - start
        for column in data:
            if column == 'Label':
                dic['y'][el] = float(data['Label'][index])
            elif column == 'Utterance':
                embeddings = [index_to_embedding[nb_word] for nb_word in data['Utterance'][index]]
                nb_words = min(len(embeddings), max_length)
                for i in range(nb_words):
                    dic['r'][el][i] = embeddings[i]
                dic['r_mask'][el][nb_words:] = 0
            elif column == 'Context':
                embeddings = [index_to_embedding[nb_word] for nb_word in data['Context'][index]]
                nb_words = min(len(embeddings), max_length)
                for i in range(nb_words):
                    dic['c'][el][i] = embeddings[i]
                dic['c_mask'][el][nb_words:] = 0
            # Return words (needed to transform to embeddings)
            document = database.find_best_document(data['Context'][index])
            document_embedded = []
            for word in document:
                if word in dico:
                    document_embedded.append(index_to_embedding[dico[word]])
                else:
                    document_embedded.append(index_to_embedding[dico['UNK']])
            nb_words = min(len(document_embedded), max_length)
            for i in range(nb_words):
                dic['m'][el][i] = document_embedded[i]
            dic['m_mask'][el][nb_words:] = 0

    # Split between training and testing
    return dic


def create_train_test_sets(train_percentage,
                           embedding_size,
                           max_length_sequence,
                           folder='../data/'):
    print('Creating dataset for the neural network')
    database = KnowledgeDataBase()
    data = cPickle.load(open(get_name_file('word_number', dataset='u_corpus'), 'rb'))
    index_to_embedding = cPickle.load(open(get_name_file('index_to_embedding', dataset='u_corpus'), 'rb'))
    rdic = cPickle.load(open(get_name_file('rdic', dataset='u_corpus'), 'rb'))
    dic = cPickle.load(open(get_name_file('dic', dataset='u_corpus'), 'rb'))
    nb_el = data.shape[0]
    middle = int(train_percentage * nb_el)
    datasets = {
        'test': create_batch(data=data,
                             index_to_embedding=index_to_embedding,
                             rdic=rdic,
                             dico=dic,
                             number_of_examples=int(nb_el * (1 - train_percentage)),
                             start=1 + middle,
                             end=nb_el,
                             size_embeddings=embedding_size,
                             max_length=max_length_sequence,
                             database=database
                             ),
        'train': create_batch(data=data,
                              index_to_embedding=index_to_embedding,
                              rdic=rdic,
                              dico=dic,
                              number_of_examples=middle,
                              start=0,
                              end=middle,
                              size_embeddings=embedding_size,
                              max_length=max_length_sequence,
                              database=database
                              )}
    cPickle.dump(datasets, open('../data/dataset.p', 'wb'))
    return datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_percentage', type=float, default=0.8, help='Percentage of element for training set')
    parser.add_argument('--embedding_size', type=int, default=28, help='dimensions of the embeddings')
    parser.add_argument('--max_length_sequence', type=int, default=160,
                        help='Max number of words consider for context/response')
    args = parser.parse_args()


    create_train_test_sets(train_percentage=args.train_percentage, embedding_size=args.embedding_size,
                           max_length_sequence=args.max_length_sequence)
