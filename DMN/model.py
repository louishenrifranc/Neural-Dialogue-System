import theano
import theano.tensor as T
import lasagne
from lasagne import layers
import utils


class model():
    def __init__(self):
        self.batch_size = 32
        self.embedding_size = 50
        self.nb_max_sentences = 10
        self.length_max_sentences = 30
        self.vocab_size = 10000
        self.nb_hidden = 32
        self.nb_hops = 5
        # Dimension of the input context is (batch_size, number of sentences, max size of sentences)
        self.context = T.itensor3('context')
        self.mask_context = T.imatrix('context_mask')
        # Dimension of the question input is (batch_size, max size of sentences)
        self.question = T.itensor3('question')
        self.mask_question = T.imatrix('question_mask')

        """
        Building the Input context module

        """
        mask_context = layers.InputLayer((self.batch_size * self.nb_max_sentences, self.length_max_sentences),
                                         input_var=self.mask_context)
        # (batch_size, nb_sentences, length_max_sentences)
        input_module = layers.InputLayer((self.batch_size, self.nb_max_sentences, self.length_max_sentences),
                                         input_var=self.context)
        # (batch_size, nb_sentences * length_max_sentences)
        input_module = layers.ReshapeLayer(input_module, (self.batch_size, -1))

        # (batch_size, nb_sentences * length_max_sequences, embedding_size)
        input_module = layers.EmbeddingLayer(input_module, self.vocab_size, self.embedding_size)

        # (batch_size, nb_sentences, length_max_sequences, embedding_size)
        input_module = layers.ReshapeLayer(input_module, (
            self.batch_size, self.nb_max_sentences, self.length_max_sentences, self.embedding_size))

        # (batch_size * nb_sentences, length_sentences, embedding_size)
        input_module = layers.ReshapeLayer(input_module, (
            self.batch_size * self.nb_max_sentences, self.length_max_sentences, self.embedding_size))

        # (batch_size * nb_sentences, nb_hidden)
        input_module = layers.GRULayer(input_module, self.nb_hidden, mask_input=mask_context,
                                       only_return_final=True)
        context = layers.get_output(input_module)
        # input_module = layers.ReshapeLayer(input_module, (self.batch_size, self.nb_max_sentences, self.nb_hidden))

        """
        Building the Input context module

        """
        # (bach_size, length_sentences)
        mask_question = layers.InputLayer((self.batch_size, self.length_max_sentences), input_var=self.mask_question)
        # (batch_size, length_sentences)
        question_module = layers.InputLayer((self.batch_size, self.length_max_sentences))

        # (batch_size, length_sentences, embedding_size)
        question_module = layers.EmbeddingLayer(question_module, self.vocab_size, self.embedding_size)

        # (batch_size, nb_hidden)
        question_module = layers.GRULayer(question_module, self.nb_hidden, mask_input=mask_question,
                                          only_return_final=True)
        question = layers.get_output(question_module)
        """
        Building the Memory module

        """
        memory = question
        self._M = utils.get_shared('glorot_uniform', self.nb_hidden, self.nb_hidden)

        for step in xrange(self.nb_hops):
            z_score_vector = T.concatenate([context,
                                            question,
                                            memory,
                                            context * question,
                                            context * memory,
                                            T.abs_(context - question),
                                            T.abs_(context - memory),
                                            T.dot(T.dot(context, self._M), question),
                                            T.dot(T.dot(context, self._M), memory)])

            self._M1 = utils.get_shared('glorot_uniform', self.nb_hidden * 9, self.nb_hidden)
            self._B1 = utils.get_shared('constant_zero', self.nb_hidden, None)
            z1 = T.tanh(T.dot(self._M1, z_score_vector) + self._B1)

            self._M2 = utils.get_shared('glorot_uniform', self.nb_hidden, 1)
            self._B2 = utils.get_shared('constant_zero', self.nb_hidden, None)
            z2 = T.nnet.sigmoid(T.dot(self._M2, z1) + self._B2)


if __name__ == '__main__':
    dmn = model()
