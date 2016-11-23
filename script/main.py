from network_repr import get_network_str
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer, LSTMLayer, ConcatLayer, get_output, get_all_params
import theano
import numpy as np
import argparse
import random
import time
import cPickle


def parse_input_args():
    '''
    Parse the argument for the model
    :return: the args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--nonlinearities', type=str, default='relu', help='Non linearities')
    args = parser.parse_args()
    return args


def get_dtype():
    return theano.config.floatX


def get_weights(name, n_in, n_out):
    """
    Create a 2D np array using specific initialization
    Parameters
    ----------
    name: string
        Initialization name
    n_in: int
        Number of rows
    n_out: int
        Number of columns
    :return: a np.array
    """
    if name == "randU":
        return np.random.uniform(-0.01, 0.01, size=(n_in, n_out))
    elif name == "randN":
        return np.random.normal(1, 0.1, size=(n_in, n_out))
    elif name == "glorotU":
        return np.random.uniform(low=-4. / np.sqrt(6.0 / (n_in + n_out)),
                                 high=4. / np.sqrt(6.0 / (n_in + n_out)), size=(n_in, n_out))
    elif name == "zeros":
        return np.zeros((n_in, n_out))
    else:
        raise 'Unsupported weigth init %s' % name


def get_name_layer(name, id, depth=-1):
    """
    Facility function  to get the name of the layer
    while building the model
    """
    return name + "%d-%s" % (id, str(depth) if depth != -1 else '')


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
    :return: a theano.shared
    """
    return theano.shared(get_weights(name, n_in, n_out).astype(get_dtype()), borrow=borrow)


def get_optimizer(name, loss, params, lr):
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
    :param lr:
        Learning rate
    :return: a lasagne.updates
    '''
    if name == "adadelta":
        return lasagne.updates.adadelta(loss, params, learning_rate=lr)
    elif name == "adagrad":
        return lasagne.updates.adagrad(loss, params, learning_rate=lr)


def get_nonlinearity(name='tanh'):
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
    else:
        raise 'Unsupported activation function %s' % name


class NeuralDialogueSystem(object):
    """ LSTM Neural Network with Knowledge Encoder
    References
    ----------
    [Lowe], Pow, Charlin, Pineau, Serban
    "Incorporating Unstructured Textual
    Knowledge Sources into Neural Dialogue Systems"
    http://media.wix.com/ugd/b6d786_137894b7b3a341a09ed0c0b45b46dbb6.pdf
    Parameters
    ----------
    batch_size: int (default 100)
        Divide training data into k minibatches
        for efficiency
    n_recurent_layers: int (default 1)
        Number of LSTM layers for the context-
        response encoder
    hidden_size: int (default 100)
        Size of the hidden unit of a LSTM
    max_length_sequence: int (default 160)
        Max size for the context ~
         Maximum number of words inputing in the LSTM layer
    embedding_size: int (default: 30)
        Size of the embedding vectors
    optimizer: string (choice between adadelta and adagrad)
        Optimizer used for gradient descent
    isbidirectionnal: boolean (default: false)
        If true, every stack of LSTM layer consist
        of a backward and a forward pass over its input
    lr: float
        Learning rate
    lr_decay: float
        Learning rate decay
    l2_reg: float (default: 0.0)
        Lambda value for L2-regularization
        No regularization if L2_reg=0.0
    l1_reg: float (default: 0.0)
        Lambda value for L1-regularization
        No regularization if L1_reg=0.0
    untied: boolean (default: false)
        If true, the Knowledge Encoder is a different
        neural network than the context/response encoder
    pre_train_DE:
        If true, pretrain only KE-untied as the DE model
        (ie N = 0)
    grad_clipping: float (default: 10.0)
        Clip the gradient in the backward pass
    training_random: bool (default False)
        When set to true, the model is "trained" on random data
        Allow check model correctness
    data:
    """

    def __init__(self,
                 data,
                 batch_size=20,
                 n_recurent_layers=1,
                 hidden_size=20,
                 max_length_sequence=160,
                 embedding_size=28,
                 optimizer='adadelta',
                 isbidirectionnal=False,
                 lr=0.01,
                 lr_decay=0.96,
                 l2_reg=0.0,
                 l1_reg=0.0,
                 untied=False,
                 pre_train_DE=False,
                 grad_clipping=10,
                 n_epoch=100,
                 training_random=False,
                 train_only_KE=False
                 ):

        if pre_train_DE == True and untied == True:
            raise ValueError(
                "When pretraining neural network (pre_train_DE==True), weights should be tied (untied=False)")
            return
        if train_only_KE and not untied:
            raise StandardError('Can\'t train only knowledge encoder if weights are untied, or if pretraining phase')
            return
        #
        # 1. Copy parameters and create placeholders
        #
        self.BATCH_SIZE = batch_size
        self.HIDDEN_SIZE = hidden_size
        self.N_RECCURENT_LAYERS = n_recurent_layers
        self.MAX_LENGTH = max_length_sequence
        self.EMBEDDING_SIZE = embedding_size
        self.optimizer = optimizer
        self.N_EPOCH = n_epoch
        self.lr = lr
        self.data = data

        self.c = T.ftensor3('c')
        self.c_mask = T.fmatrix('c_mask')

        self.r = T.ftensor3('r')
        self.r_mask = T.fmatrix('r_mask')

        self.m = T.ftensor3('m')
        self.m_mask = T.fmatrix('m_mask')

        self.target = T.fvector('y')

        self.M = get_shared("glorotU", self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        self.N = get_shared("glorotU", self.HIDDEN_SIZE, self.HIDDEN_SIZE)

        self.shared_variable = {}
        for var in ['c', 'r', 'm']:
            self.shared_variable[var] = theano.shared(
                np.zeros((self.BATCH_SIZE, self.MAX_LENGTH, self.EMBEDDING_SIZE)).astype(get_dtype()))
        for var in ['c_mask', 'r_mask', 'm_mask']:
            self.shared_variable[var] = get_shared('zeros', self.BATCH_SIZE, self.MAX_LENGTH)
        self.shared_variable['y'] = theano.shared(np.zeros((self.BATCH_SIZE,)).astype(get_dtype()))

        #
        # 2. Build the model
        #
        print('Building model')
        model = {}
        # Input Layer is of size (N_BATCH, MAX_LENGTH, EMBEDDING_SIZE)
        model['l_in'] = InputLayer(
            shape=(self.BATCH_SIZE, self.MAX_LENGTH, self.EMBEDDING_SIZE))
        # The network also needs a way to provide a mask for each sequence.  I'll
        # use a separate input layer for that. Since the mask only determines
        # which indices are part of the sequence for each batch entry, they are
        # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
        model['l_mask'] = InputLayer(shape=(self.BATCH_SIZE, self.MAX_LENGTH))
        prev, prev_b = model['l_in'], model['l_in']
        # Iterate over the depth of the neural network (only building the DE part)
        # If KE weights are tied, then the same neural network will be used for
        # the context, the response, and the knowledge encoder
        for num_layer in xrange(n_recurent_layers):
            # If the LSTM is bidirectionnal, then two passes are made above the input for every depth
            if isbidirectionnal:
                model['l_backward' + str(num_layer)] = LSTMLayer(prev_b,
                                                                 self.HIDDEN_SIZE,
                                                                 # Only keep the mask for the InputLayer - LSTMLayer connection
                                                                 mask_input=model[
                                                                     'l_mask'] if num_layer == 0 else None,
                                                                 only_return_final=True if num_layer == n_recurent_layers - 1 else False,
                                                                 grad_clipping=grad_clipping,  # clip the gradients
                                                                 backwards=True,
                                                                 learn_init=True,  # initial hidden values are learned.
                                                                 peepholes=True)  # peephole connection
                prev_b = model['l_backward' + str(num_layer)]
            model['l_forward' + str(num_layer)] = LSTMLayer(prev,
                                                            self.HIDDEN_SIZE,
                                                            mask_input=model[
                                                                'l_mask'] if num_layer == 0 else None,
                                                            only_return_final=True if num_layer == n_recurent_layers - 1 else False,
                                                            grad_clipping=grad_clipping,
                                                            learn_init=True,
                                                            peepholes=True)
            prev = model['l_forward' + str(num_layer)]
        # Concat bidirectionnal layers
        if isbidirectionnal:
            model['l_concat'] = ConcatLayer([prev, prev_b])  # concat backward and forward neural network
            model['l_out'] = DenseLayer(model['l_concat'], num_units=self.HIDDEN_SIZE,
                                        nonlinearity=get_nonlinearity('tanh'))  # dense layer
        else:
            model['l_out'] = DenseLayer(prev, num_units=self.HIDDEN_SIZE,
                                        nonlinearity=get_nonlinearity('tanh'))

        # KE encoder if weights are untied:
        # Build a separate neural network of depth with one LSTM layer and one DenseLayer at the end
        if untied:
            model['l_forward_untied'] = LSTMLayer(model['l_in'],
                                                  self.HIDDEN_SIZE,
                                                  mask_input=model['l_mask'],
                                                  only_return_final=True,
                                                  grad_clipping=grad_clipping,
                                                  learn_init=True,
                                                  peepholes=True)
            model['l_out_untied'] = DenseLayer(model['l_forward_untied'], num_units=self.HIDDEN_SIZE,
                                               nonlinearity=get_nonlinearity('tanh'))
            # Print topology of the KE network
            # print(get_network_str(model['l_out_untied']))

        # Print topology of the DE network
        # print(get_network_str(model['l_out']))

        # Get the output given that the context was inputed
        c_output = lasagne.layers.get_output(model['l_out'], inputs=
        {model['l_in']: self.c, model['l_mask']: self.c_mask})
        # Get the output given that the response was inputed
        r_output = lasagne.layers.get_output(model['l_out'], inputs=
        {model['l_in']: self.r, model['l_mask']: self.r_mask})
        # Get the output given that the message was inputed
        m_output = lasagne.layers.get_output(model['l_out'] if not untied else model['l_out_untied'], inputs=
        {model['l_in']: self.m, model['l_mask']: self.m_mask})

        c_M_r = T.batched_dot(c_output, T.dot(r_output, self.M.T))
        m_N_r = T.batched_dot(m_output, T.dot(r_output, self.N.T))

        # Formula : out = sigmoid(context * M * response + knowledge * N * response)
        # When pre_train_DE is True, knowledge encoder output is not used (N = 0)
        results = c_M_r
        if not pre_train_DE:
            results += m_N_r
        self.out = T.nnet.sigmoid(results)

        #
        # 3. Define loss function as the mean over the batch element of the cross entropy between
        #    the output and the true target
        # (no regularization)
        self.cost = T.mean(lasagne.objectives.binary_crossentropy(self.out, self.target), axis=0)
        self.model = model

        #
        # 4. Create weights update list
        #
        # Three lists:
        # a. List of all parameters for a tied encoder
        # b. List of KE weights when DE and KE are separated
        # c. List of all weights when DE and KE are separated
        tied_params = get_all_params(self.model['l_out'], trainable=True) + [self.M]
        if not pre_train_DE:
            tied_params += [self.N]
        if untied:
            untied_params = get_all_params(self.model['l_out_untied'], trainable=True)
            all_params = tied_params + untied_params
            untied_params += [self.N]

        #
        # 5. Select an optimizer.
        #
        if untied:
            updates_untied = get_optimizer(optimizer, self.cost, untied_params, self.lr)
            updates_all = get_optimizer(optimizer, self.cost, all_params, self.lr)
        updates_tied = get_optimizer(optimizer, self.cost, tied_params, self.lr)

        #
        # 6. Create theano function for training and testing the neural network
        #    One training function is created depending on three parameters
        #     * pre training DE
        #     * training only KE
        #     * weights are tied
        #
        givens = {
            self.c: self.shared_variable['c'],
            self.r: self.shared_variable['r'],
            self.target: self.shared_variable['y'],
            self.c_mask: self.shared_variable['c_mask'],
            self.r_mask: self.shared_variable['r_mask']
        }
        # If we don't want to pretrain
        if not pre_train_DE:
            givens[self.m] = self.shared_variable['m']
            givens[self.m_mask] = self.shared_variable['m_mask']
        # If we don't want to train only the KE
        if not train_only_KE:
            if not untied:  # if weights are tied, pick only weights of the DE as KE = DE
                self.train_fn = theano.function([], self.cost, updates=updates_tied,
                                                givens=givens,
                                                on_unused_input='warn')
            else:  # if weights are untied, pick weights for the DE, and the KE as KE != DE
                self.train_fn = theano.function([], self.cost, updates=updates_all, givens=givens,
                                                on_unused_input='warn')
        # If we want to train the KE only
        else:
            self.train_fn = theano.function([], self.cost, updates=updates_untied, givens=givens,
                                            on_unused_input='warn')
        # Test function is similar for all cases
        self.test_fn = theano.function([], self.cost, givens=givens, on_unused_input='warn')

        # Select the function for getting the next batch (random data, or true labelled data
        self.get_next_batch = self.gen_random_data if training_random else self.gen_data
        print('Model built')

    def gen_random_data(self,
                        train_or_test):
        """
        Generate random data
        Set the shared variable for inputs, and return the current iteration index
        Parameters
        ----------
        train_or_test: str in ['test', 'train']
        :return: iteration index
        """
        nb_train_iter = 10  # Random value (number of elements will be nb_train_iter * batch_size)
        nb_test_iter = 2  # Random value (number of elements  will be nb_test_iter * batch_size)
        for iter in range(nb_train_iter if train_or_test == "train" else nb_test_iter):
            words_c = np.random.normal(1, 0.1, (self.BATCH_SIZE, self.MAX_LENGTH, self.EMBEDDING_SIZE)).astype(
                get_dtype())
            mask_c = np.ones((self.BATCH_SIZE, self.MAX_LENGTH)).astype(get_dtype())

            words_r = np.random.normal(1, 0.1, (self.BATCH_SIZE, self.MAX_LENGTH, self.EMBEDDING_SIZE)).astype(
                get_dtype())
            mask_r = np.ones((self.BATCH_SIZE, self.MAX_LENGTH)).astype(get_dtype())

            words_m = np.random.normal(1, 0.1, (self.BATCH_SIZE, self.MAX_LENGTH, self.EMBEDDING_SIZE)).astype(
                get_dtype())
            mask_m = np.ones((self.BATCH_SIZE, self.MAX_LENGTH)).astype(get_dtype())

            y = np.random.randint(0, 2, size=(self.BATCH_SIZE,)).astype(get_dtype())
            for i in range(self.BATCH_SIZE):
                nb_words_c = random.randint(1, self.MAX_LENGTH)
            words_c[i][nb_words_c:][:] = 0
            mask_c[i][nb_words_c:] = 0

            nb_words_r = random.randint(1, self.MAX_LENGTH)
            words_r[i][nb_words_r:][:] = 0
            mask_r[i][nb_words_r:] = 0

            nb_words_m = random.randint(1, self.MAX_LENGTH)
            words_m[i][nb_words_m:][:] = 0
            mask_m[i][nb_words_m:] = 0
            self.shared_variable['c'].set_value(words_c)
            self.shared_variable['r'].set_value(words_r)
            self.shared_variable['r_mask'].set_value(mask_r)
            self.shared_variable['c_mask'].set_value(mask_c)
            self.shared_variable['y'].set_value(y)
            self.shared_variable['m'].set_value(words_m)
            self.shared_variable['m_mask'].set_value(mask_m)
            yield iter

    def gen_data(self,
                 train_or_test,
                 shuffle_index=True
                 ):
        """
        Split training/testing data into batch of size self.BATCH_SIZE
        Set the shared variable and return the current iteration index
        Parameters
        ----------
        train_or_test: str in ['test', 'train']
        shuffle_index: bool (default=True)
            Shuffle indices in the data
        :return: iteration index
        """
        batch = self.data[train_or_test]
        number_of_elements = batch['y'].shape[0]
        indices = range(number_of_elements)
        if shuffle_index:
            np.random.shuffle(indices)
        number_iterations = number_of_elements / self.BATCH_SIZE
        iter = -1
        while iter < (number_iterations - 1):
            iter += 1
            indice = indices[self.BATCH_SIZE * iter: min(self.BATCH_SIZE * (iter + 1), number_of_elements)]
            self.shared_variable['c'].set_value(batch['c'][indice])
            self.shared_variable['r'].set_value(batch['r'][indice])
            self.shared_variable['r_mask'].set_value(batch['r_mask'][indice])
            self.shared_variable['c_mask'].set_value(batch['c_mask'][indice])
            self.shared_variable['y'].set_value(batch['y'][indice])
            self.shared_variable['m'].set_value(batch['m'][indice])
            self.shared_variable['m_mask'].set_value(batch['m_mask'][indice])
            yield iter

    def train(self):
        """
        Train the neural network
        (very simple training: no patience, no saving of the best model)
        At each epoch:
            * Iterate over the training set and calculate the mean error
            * Iterate over the test set and calculate the mean error
        :return:
        """
        for epoch in range(self.N_EPOCH):
            nb_iters = 0
            total_cost_epoch = 0
            start_time = time.time()
            for iter in self.get_next_batch('train'):
                cost_epoch = self.train_fn()
                total_cost_epoch += cost_epoch
                nb_iters += 1
                # print "Iter %d: %f" % (iter, cost_epoch)
            print "(Epoch %d) Train set: \t\t Cost: %f in %d secs" % (
                epoch, total_cost_epoch / nb_iters, float(time.time() - start_time))

            nb_iters = 0
            total_cost_epoch = 0
            for iter in self.get_next_batch('test'):
                cost_epoch = self.test_fn()
                total_cost_epoch += cost_epoch
                nb_iters += 1
            print "(Epoch %d) Test set: \t\t Cost: %f" % (epoch, total_cost_epoch / nb_iters)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--batch_size', type=int, default=32, help='Size of a batch')
    parser.add_argument('--nrecurrent_layers', type=int, default=1, help='Depth of the dual encoder network')
    parser.add_argument('--hiddensize', type=int, default=28, help='Size of the hidden layers of the LSTM cell')
    parser.add_argument('--optimizer', type=str, default='adadelta', help='Adaptative gradient method')
    parser.add_argument('--isbidirectionnal', type=bool, default=False, help='LSTM DE bidirectionnal')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.96, help='Learning rate decay')
    parser.add_argument('--untied', type=bool, default=False, help='')
    parser.add_argument('--pre_train_DE', type=bool, default=True, help='Train only the dual encoder (N=0)')
    parser.add_argument('--train_only_KE', type=bool, default=False, help='Train only the knowledge encoder')
    parser.add_argument('--grad_clipping', type=int, default=10, help='Clip gradient value')
    parser.add_argument('--embedding_size', type=int, default=28, help='Size of the embeddings')
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--training_random', type=bool, default=True, help='Train on random data')
    parser.add_argument('--max_length_sequence', type=int, default=160,
                        help='Windows size for context/response')
    parser.add_argument('--data', type=str, default=None,
                        help='File containing the data (train/test sets)')
    args = parser.parse_args()

    if args.data is None:
        if args.training_random == False:
            raise StandardError('no data provided. Make sure to create train/test set with gen_data.py before')
            return 1
        data = None
    else:
        # Load the data
        data = cPickle.load(open(args.data, 'rb'))
    print('All arguments : ', args)
    # Create the model
    rnn = NeuralDialogueSystem(data=data,
                               batch_size=args.batch_size,
                               n_recurent_layers=args.nrecurrent_layers,
                               hidden_size=args.hiddensize,
                               max_length_sequence=args.max_length_sequence,
                               optimizer=args.optimizer,
                               isbidirectionnal=args.isbidirectionnal,
                               lr=args.lr,
                               lr_decay=args.lr_decay,
                               untied=args.untied,
                               pre_train_DE=args.pre_train_DE,
                               embedding_size=args.embedding_size,
                               grad_clipping=args.grad_clipping,
                               n_epoch=args.n_epoch,
                               training_random=args.training_random,
                               train_only_KE=args.train_only_KE)
    # Train the model
    rnn.train()


if __name__ == '__main__':
    main()
