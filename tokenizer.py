"""
TOKENIZERS FOR PRE-PROCESSING AND TOKENIZING INPUT TEXT

"""

import numpy as np
import re


class WordTokenizer:
    """
    Class for building word-based tokenizers either from pre-trained word embeddings or based on an input text.
    The latter case means this class can be used to tokenize classes, NER tag sequences etc.

    Typical use:
    tok = WordTokenizer(specify necessary flags)
    tok.initialize(file paths for text/pre-trained word embeddings)
    tokens = tok.method(input text)

    Note: PAD tokens are included in the associated vocabulary. OOV tokens are included as well, unless oov_tok is set
    to None, which can be useful with using this tokenizer to encode class labels.

    At the time of initialization, you can set the flags to decide in what 'mode' the tokenizer to operate in, based
    either on a pre-trained vocabulary and word vectors, or when to fit it to a piece of text to generate vocabulary.
    """

    def __init__(self, from_pre=False, pp=False, oov_tok='[OOV]'):
        """
        :param from_pre: Flag to determine if tokenizer is to be build from pre-trained embeddings
        :param pp: Flag to turn 'pre-processing' of text input on or off. If 'from_pre' is True, pp is automatically set
        to True, since pre-processing text is necessary when using pre-trained word vocabularies.
        :param oov_tok: Token to be used for out-of-vocabulary, ie. unknown words. Can also be set to None, in which
        case the oov_token isn't added to vocabulary. This needs to be the case when using this tokenizer to encode
        class labels.
        """
        self.pad_tok, self.pad_index = '[PAD]', 0
        self.oov_tok, self.oov_index = oov_tok, None

        self.word2in = dict()
        self.in2word = dict()
        self.emb_matrix = None
        self.vocab_size = None
        self.from_pre = from_pre
        if self.from_pre:
            self.pp = from_pre
        else:
            self.pp = pp

        self.word2in[self.pad_tok], self.in2word[self.pad_index] = self.pad_index, self.pad_tok
        self.fitted = False

    def text_to_indices(self, sents):
        """
        Converts a list of list of tokens into a list of list of integers, based on the vocabulary.
        And token out of the vocabulary will be replaced by the oov-index.
        :param sents: Input tokens
        :return: Indices of input tokens
        """
        if not self.fitted:
            raise RuntimeError("WordTokenizer not initialized")
        return [[self.word2in.get(w, self.oov_index) for w in sent] for sent in sents]

    def indices_to_text(self, indices):
        """
        Inverse of the text_to_indices method. Converts indices to the tokens associated with them.
        :param indices: Indices
        :return: Tokens
        """
        return [[self.in2word.get(i, self.oov_tok) for i in row] for row in indices]

    def from_glove(self, glove_file):
        """
        Read in pre-trained word vectors to assemble the vocabulary and the embedding matrix associated.
        :param glove_file: File path for GloVE/ pre-trained vectors
        """
        # Read in the glove file
        print("Reading the GloVe data from {}...".format(glove_file))
        with open(glove_file, 'r') as f:
            words = set()
            word_to_vec_map = dict()
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)

        # Initialise the Embedding matrix
        vocab_len = len(word_to_vec_map) + 2 # For PAD and OOV
        emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of GloVe word vectors (= 50)
        self.emb_matrix = np.zeros((vocab_len, emb_dim), np.float32)
        self.vocab_size = vocab_len
        self.fitted = True

        print("Creating Embedding matrix...")
        # Fill in word2in, in2word and the embedding matrix
        i = 1
        for w in sorted(words):
            self.word2in[w] = i
            self.in2word[i] = w
            self.emb_matrix[i] = word_to_vec_map[w]
            i += 1
        self.oov_index = i
        self.word2in[self.oov_tok], self.in2word[self.oov_index] = self.oov_index, self.oov_tok
        self.emb_matrix[self.oov_index] = np.random.uniform(-0.5, 0.5, emb_dim)

    def check_fit(self, texts):
        """
        Checks the fit of the pre-trained vocabulary on the data to give the number of unknown tokens.
        :param texts: Input text as list[list[str]]
        """
        words = {word for text in texts for word in text}
        common = len(words.intersection(self.word2in))
        print("Total number of words in the text are {}, of which {} are covered by the pre-trained"
              " vocabulary.".format(len(words), common))
        print("Percentage intersection is {}".format(common/len(words)))

    def fit_on_text(self, texts):
        """
        Fits the tokenizer to a given text, when the 'from_pre' flag is set to False. All words in the text are indexed
        and added to the vocabulary.
        :param texts: Input texts as a list[list[str]]
        """
        i = 1
        print("Creating vocabulary...")
        for text in texts:
            for word in text:
                if word not in self.word2in:
                    self.word2in[word] = i
                    self.in2word[i] = word
                    i += 1

        if self.oov_tok:
            self.oov_index = i
            self.word2in[self.oov_tok], self.in2word[self.oov_index] = self.oov_index, self.oov_tok

        self.vocab_size = len(self.word2in)
        self.fitted = True

    def _process_word(self, word):
        """
        Basic word level pre-processing to convert to lower case and remove all special characters.
        :param word: Input string
        :return: Processed string
        """
        word = word.lower()
        word = re.sub('[\W_]+', '', word)
        return word

    def pre_process(self, texts):
        """
        Takes text in a raw format, either as a list of strings, or a list of list of strings, and processes the data
        to convert all words to lowercase and remove any special characters.
        :param texts: Input text as list[list[str]] or list[str]
        :return: Processed text as list[list[str]]
        """
        processed = []
        for text in texts:
            if isinstance(text, str):
                text = text.split()
            processed.append([self._process_word(w) for w in text])
        return processed

    def initialize(self, texts, glove_file=None):
        """
        Based on the flags set at init, this method either fits the tokenizer object to a piece of text, or reads in
        a pre-trained word vocabulary and embeddings. In the second case, it also calculates the goodness of fit of the
        vocabulary on the input text. In both cases the processed text is cleaned up and returned.
        :param texts: Raw input text used to fit the tokenizer, or to be processed
        :param glove_file: File path for pre-trained embeddings
        :return: Processed text as a list of list of strings
        """
        if self.from_pre:
            self.from_glove(glove_file)
            processed = self.pre_process(texts)
            self.check_fit(processed)
        else:
            if self.pp:
                processed = self.pre_process(texts)
            else:
                processed = texts
            self.fit_on_text(processed)
        return processed


class CaseTokenizer:
    """
    This class creates a special tokenizer used to capture some character level information regarding words. Has 5
    pre-defined categories for words and generates index accordingly.
    Note that index 0 is used for the pad token to be consistent with other models.
    """
    def __init__(self):

        self.case2in = {
            'allLower': 1,
            'allUpper': 5,
            'firstCap': 2,
            'numeric': 3,
            'mixed': 4,
            'pad': 0
        }
        self.vocab_size = len(self.case2in)
        self.emb_matrix = np.zeros(self.vocab_size-1)
        self.emb_matrix = np.vstack((self.emb_matrix, np.identity(self.vocab_size-1, dtype='float32')))

    def case_def(self, word):
        """
        Given a string (word), identifies the case based index.
        :param: word(str)
        :return: index
        """
        nums = sum([int(c.isdigit()) for c in word])
        if nums>0:
            return self.case2in['numeric']
        if word.islower():
            return self.case2in['allLower']  # lowercase
        elif word.isupper():
            return self.case2in['allUpper']
        elif word[0].isupper():
            return self.case2in['firstCap']
        else:
            return self.case2in['mixed']

    def text_to_indices(self, sents):
        indices = []
        for sent in sents:
            if isinstance(sent, str):
                sent = sent.split()
            indices.append([self.case_def(w) for w in sent])
        return indices
