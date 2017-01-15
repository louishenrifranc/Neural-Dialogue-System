import cPickle
import numpy as np
import lasagne.nonlinearities
import json
import os
from Layer import Layer
from theano import shared

"""
Util functions when building neural network using Lasagne librairy
Functions
--------
* get the type of the weights
* get the weights
* get the optimizer function
* get the non linearity
* get a Theano shared variable
--------
"""


def get_dtype():
    return 'float16'


def get_weights(name, n_in, n_out=None):
    """
    Create a 2D np array using specific initialization
    Parameters
    ----------
    name: string
        Initialization name
    n_in: int
        Number of rows
    n_out: int (default None for biais only)
        Number of columns
    :return: a np.array
    """
    if name == "randU":
        return np.random.uniform(-0.01, 0.01, size=(n_in, n_out))
    elif name == "randN":
        return np.random.normal(1, 0.1, size=(n_in, n_out))
    elif name == "glorot_uniform":
        return np.random.uniform(low=-4. / np.sqrt(6.0 / (n_in + n_out)),
                                 high=4. / np.sqrt(6.0 / (n_in + n_out)), size=(n_in, n_out))
    elif name == "constant_zero":
        if n_out is None:
            # For biais
            return np.zeros((n_in))
        else:
            return np.zeros((n_in, n_out))
    else:
        raise 'Unsupported weigh init %s' % name


def get_optimizer(name, loss, params, lr, decay_lr):
    '''
    Get a lasagne optimizer object
    Parameters
    ----------
    name: string
        Name of the optimizer
    loss:
        Loss function to minimize
    params:
        Params to updates
    :param lr: scalar or theano.tensor.scalar
        Learning rate
    :return: a lasagne.updates
    '''
    if name == "adadelta":
        return lasagne.updates.adadelta(loss, params)
    elif name == "adagrad":
        return lasagne.updates.adagrad(loss, params, learning_rate=lr)
    elif name == "rmsprop":
        return lasagne.updates.rmsprop(loss, params, learning_rate=lr, rho=decay_lr)


def get_non_linearity(name):
    '''
    Get a lasagne nonlinearities object
    Parameters
    ----------
    name: string (default : tanh)
        Nonlinearity function
    :return: a lasagne.nonlinearities
    '''
    if name == "relu":
        return lasagne.nonlinearities.rectify
    elif name == "tanh":
        return lasagne.nonlinearities.tanh
    elif name == "sigmoid":
        return lasagne.nonlinearities.sigmoid
    elif name == "linear":
        return lasagne.nonlinearities.linear
    else:
        raise 'Unsupported activation function %s' % name


def get_shared(name, n_in, n_out, borrow=True):
    """
    Create a theano.shared 2D array
    Parameters
    ----------
    name: string
        Initialization of the element in the tensor
    n_in: int
        Number of rows
    n_out: int
        Number of columns
    borrow: boolean
        Shared the tensor
    :return: a shared tensor
    """
    return shared(get_weights(name, n_in, n_out).astype(get_dtype()), borrow=borrow)


"""
Util functions to load data set.
Functions:
----------
* load the data set
* get next batch
----------
"""


def get_relative_filename(filename, data_folder='data/'):
    filename = data_folder + filename
    return os.path.join(os.path.dirname(__file__), '..', filename)


def load_dataset(name, datafolder):
    X, y = cPickle.load(open(get_relative_filename(name, data_folder=datafolder), 'rb'))
    nb_example = len(X)

    s1 = int(0.6 * nb_example)
    s2 = int(0.8 * nb_example)
    X_train, y_train = X[:s1, ], y[:s1]
    X_val, y_val = X[s1:s2, ], y[s1:s2]
    X_test, y_test = X[s2:, ], y[s2:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, outputs, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], outputs[excerpt]


"""
Util functions for spearmint
Functions:
----------
----------
"""


def get_layers(params):
    # Open file containing the predefined layers parameters
    json_file = json.load(
        open(get_relative_filename('predefined_values.json'), 'rb'))
    priors = json_file['layers']

    # Global hyper parameter definition. Used as default value, if not mention anywhere else
    default_values = json.load(
        open(get_relative_filename('default_values.json'), 'rb'))

    max_depth = 0
    # Max depth is the maximum between the number of layers defined in the config file, and the json file
    max_depth = max(max(len(params[param]) for param in params), len(priors))
    layers = []
    indexes = {}
    # For every layer
    for depth in xrange(max_depth):
        # Set default values
        # Overwritten if defined in the .pb or .json file
        layer_info = {name: value for name, value in default_values.iteritems()}

        # Set values from the config file
        # Overwritten if defined in json file
        for param in params:
            if len(params[param]) > 0:
                if not param in indexes:
                    indexes[param] = 0
                if indexes[param] < len(params[param]) and indexes[param] != -1:
                    layer_info[param] = params[param][indexes[param]]
                    indexes[param] += 1

        # Set values from the json file
        # Never overwritten
        for layer in xrange(len(priors)):
            # Search for the good layer
            if priors[layer]['layer_nb'] == depth:
                # Iterate over all parameters set
                for param in priors[layer]['properties']:
                    layer_info[param] = priors[layer]['properties'][param]
                    # The parameter that has been read was not for this layer (a coming one)
                    if param in indexes:
                        indexes[param] -= 1

                    if param in params and indexes[param] == len(params[param]):
                        indexes[param] = -1
        # Set the name
        layer_info['name'] = 'l' + str(depth)
        layers.append(Layer(layer_info))

    # For debug purpose
    for layer in layers:
        print(str(layer))

    return layers


def get_nn_parameters(filename):
    """
    Return neural network attributes in a dic
    Parameters
    ----------
    filename: string (default: data/global_nn_parameters.json)
        file containing the attributes of the neural network

    :return: a dict containing the optimizer,
                             the batch size,
                             the number of input,
                             the learning rate
    """
    return json.load(open(filename, 'rb'))
