import cPickle
import random


class tfidf:
    """
    An extremely simple Python library to perform TF-IDF document comparison.
    [https://github.com/hrs/python-tf-idf]
    """
    W

    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def addDocument(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.documents.append([doc_name, doc_dict])

    def clearDocuments(self):
        self.documents = []

    def similarities(self, list_of_words):
        """Returns a list of all the [docname, similarity_score] pairs relative to a list of words."""

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # computing the list of similarities
        sims = []
        for doc in self.documents:
            score = 0.0
            doc_dict = doc[1]
            for k in query_dict:
                if k in doc_dict:
                    score += (query_dict[k] / self.corpus_dict[k]) + (doc_dict[k] / self.corpus_dict[k])
            sims.append([doc[0], score])
        return sims


class KnowledgeDataBase(object):
    """
    Object representing the knowledge dataset.
    A call on this object with some context embeddings words return the best matching document in the database.
    Scores are calculating with tf-IDF as described in the paper
    [Incorporating Unstructured Textual Knowledge Sources into Neural Dialogue Systems]
    Parameters
    ----------
    entity_hashtable: pickle object (default: None)
        See description in the paper
    relation_hashtable: pickle object (default: None)
        See description in the paper
    embedding_to_word: object (default: 'embedding_to_word.p')
        A pickle object containing translation from embedding to word
    """

    def __init__(self,
                 entity_hashtable=None,
                 relation_hashtable=None,
                 embedding_to_word='embedding_to_word.p'):
        if entity_hashtable is None:
            self.entity_hashtable = self.create_entity_table()
        else:
            self.entity_hashtable = entity_hashtable

        if relation_hashtable is None:
            self.relation_hashtable = self.create_relation_hashtable()
        else:
            self.relation_hashtable = relation_hashtable

        self.embedding_to_word = embedding_to_word

    def find_best_document(self, context, max_size=20):
        # list of best documents
        documents = []

        """
        1. Iterate over all words,
           If there are more than 20 context words that are in the entity hashtable, keep the longest words
           Then add the first 20 documents corresponding to this words.
        """
        words = []
        for word in context:
            if word in self.entity_hashtable:
                words.append(word)
        words.sort(key=len)

        for word in words:
            for document in self.entity_hashtable[word]:
                documents.append(document)
                if len(documents) == max_size:
                    break
            else:
                continue
            break

        """
        2. Iterate over all words in the relation hash table to extend the research
        The condition to go in this loop is too not have already 20 documents
        """
        words = []
        if len(documents) <= max_size:
            for word in context:
                if word in self.relation_hashtable:
                    words.append(word)
            words.sort(key=len)

            for word in words:
                key = self.relation_hashtable[word]
                for document in self.entity_hashtable[key]:
                    documents.append(document)
                    if len(documents) > max_size:
                        break
                    else:
                        continue
                break

        """
        3. Create a list of all words in our context, and calculate the tf-idf score on every documents
         in the list of documents just retrieve
        """
        words = context
        max_score = -1
        best_document = []
        for document in documents:
            tf_IDF = tfidf()
            tf_IDF.addDocument(document)
            score = tf_IDF.similarities(words)
            if score > max_score:
                max_score = score
                best_document = document

        """
        3.bis If no document has been found, return random knowledge (don't specify in the paper)
        """
        if len(documents) == 0:
            best_document_key = random.choice(self.entity_hashtable.keys())
            best_document = random.choice(self.entity_hashtable[best_document_key])

        """
        4. Return the best document given the context using the algorithm describe in the paper
        """
        return best_document

    def create_relation_hashtable(self,
                                  name="../data/tokenized_sentences_man.p",
                                  save=False):
        """
        Create the relation hash table
        The relation hash table maps each top word in the name to the name of the function it refers to.
        :param name: Name of the file which contained the man pages tokenized
        :return:
        """
        data = cPickle.load(open(name, 'rb'))
        relation_hashtable = {}
        for index, _ in data.iterrows():
            function = data["function"][index]
            names = data["name"][index]
            for name in names:
                relation_hashtable.setdefault(name, []).append(function)
        # cPickle.dump(relation_hashtable, open('relation_hashtable' + version + '.p', 'wb'))
        return relation_hashtable

    def create_entity_table(self,
                            name="../data/tokenized_sentences_man.p",
                            max_length=160, save=False):
        """
        Create the entity hash table
        The Entity Table maps each function to a corpus of document which are paragraph in the man pages of the function
        :param name: Name of the file which contained the man pages tokenized
        :param max_length: Max Size of the Paragraph. The paragraph can't exceed the size of the LSTM neural network unrolled
        :return:
        """
        data = cPickle.load(open(name, 'rb'))
        entity_hashtable = {}
        # iterate over the man functions
        for index, _ in data.iterrows():
            function = data['function'][index][0]
            # iterate over all descriptions until we haven't fill the description with max_length words
            for i in range(1, 5, 1):
                words = data["description" + str(i)][index]

                document = words[: min(len(words), max_length)]
                entity_hashtable.setdefault(function, []).append(document)
        # cPickle.dump(entity_hashtable, open('entity_hashtable' + version + '.p', 'wb'))
        return entity_hashtable
