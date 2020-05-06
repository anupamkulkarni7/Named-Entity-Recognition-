"""
NAMED ENTITY RECOGNITION
"""
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout

from utils import *
from tokenizer import *


class SolverConfig:
    """
    This class can be used to specify model training related parameters.
    """
    def __init__(self, epochs=15, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size


class NERTagger:
    """
    Word level entity recognition system which takes in pre-build tokenizers. A deep sequence BiLSTM model is trained
    and used internally as a part of the model class. Note that in this reference, 'model' refers to the underlying
    sequence model, while 'tagger' refers to the actual NER system.

    TYPICAL USE:
    While training:
    First define tok_x, tok_c and tok_y object instances
    ner_tagger = NERTagger(tok_x, tok_c, tok_y)
    ner_tagger.build()
    ner_tagger.train(train data, validation data)
    ner_tagger.save() (Optional)
    preds = ner_tagger.predict(test_data)

    While testing:
    ner_tagger = NERTagger()
    ner_tagger.load( optional file paths )
    preds = ner_tagger.predict(test_data)
    """
    def __init__(self, tok_x=None, tok_c=None, tok_y=None, n_hidden=75, mask_zero=True, max_len=65):
        """
        :param tok_x: WordTokenizer object for input sentence or tokens
        :param tok_c: CaseTokenizer object for input sentence or tokens
        :param tok_y: WordTokenizer object for NER tags
        :param n_hidden: Dimension of the hidden layer in the LSTM of the sequence model
        :param mask_zero: Flag to control masking for 0 indexed sequences for padding
        :param max_len: Maximum sequence length, used for padding sequences and in model definition

        The params directory encapsulates all the underlying sequence model related attributes.
        """
        self.params = {
            'mask_zero': mask_zero,
            'w_embed_dim': 50,
            'cs_embed_dim': 5,
            'n_hidden': n_hidden,
            'max_len': max_len,
            'train_emb': False
        }
        self.model = None
        self.tok_x = tok_x
        self.tok_y = tok_y
        self.tok_c = tok_c

    def save(self, aux_file="tokenizers.pkl", model_file="model.h5"):
        """
        Save the system parameters such as the model and tokenizers.
        :param aux_file: File path for auxiliary files, ie. tokenizers
        :param model_file: File path for the sequence model file
        """
        print("Saving model weights to files...\nTokenizers: {}\nModel weights: {}".
              format(aux_file, model_file))

        pkl_dump((self.tok_x, self.tok_y, self.tok_c), aux_file)
        self.model.save(model_file)

    def load(self, aux_file="tokenizers.pkl", model_file="model.h5"):
        """
        Loads the saved tokenizer and sequence model from specified file paths
        :param aux_file: File path for auxiliary files, ie. tokenizers
        :param model_file: File path for the sequence model file
        """
        print("Loading model weights from files...\nTokenizers: {}\nModel weights: {}".
              format(aux_file, model_file))

        self.tok_x, self.tok_y, self.tok_c = pkl_load(aux_file)
        self.model = load_model(model_file)

    def _build_embed(self):
        """
        Builds the embedding layers for the sequence model. Two embedding layers are built: one for the word emeddings
        and the second for the case embeddings.
        In case that an embedding matrix is associated with the tokenizer ie. in case of pre-trained embeddings, this is
        used to build the Embedding layer weights accordingly. If not, no weights will be set.
        """

        word_emb = self.tok_x.emb_matrix
        case_emb = self.tok_c.emb_matrix
        trainable = self.params['train_emb']
        mask_zero = self.params['mask_zero']

        if word_emb is not None:
            word_vocab_size, word_embed_dim = word_emb.shape
        else:
            word_vocab_size, word_embed_dim = self.tok_x.vocab_size, self.params['w_embed_dim']

        if case_emb is not None:
            case_vocab_size, case_embed_dim = case_emb.shape
        else:
            case_vocab_size, case_embed_dim = self.tok_c.vocab_size, self.params['cs_embed_dim']

        word_embed_layer = Embedding(input_dim=word_vocab_size, output_dim=word_embed_dim,
                                     trainable=trainable, mask_zero=mask_zero)
        if word_emb is not None:
            word_embed_layer.build(None, )
            word_embed_layer.set_weights([word_emb])

        case_embed_layer = Embedding(input_dim=case_vocab_size, output_dim=case_embed_dim,
                                     trainable=trainable, mask_zero=mask_zero)
        if case_emb is not None:
            case_embed_layer.build(None, )
            case_embed_layer.set_weights([case_emb])

        return word_embed_layer, case_embed_layer

    def build(self):
        """
        Builds the deep sequence model based on set parameters and the tokenizers. Note that this is called only at
        training time.
        """
        if self.tok_x is None or self.tok_y is None or self.tok_c is None:
            raise RuntimeError("Please set tokenizers..")

        max_len = self.params['max_len']
        n_hidden = self.params['n_hidden']
        n_slots = self.tok_y.vocab_size

        word_embed_layer, case_embed_layer = self._build_embed()
        sentence_indices = Input(shape=(max_len,), dtype='int32')
        case_indices = Input(shape=(max_len,), dtype='int32')

        word_embs = word_embed_layer(sentence_indices)
        case_embs = case_embed_layer(case_indices)
        embeddings = tf.concat([word_embs, case_embs], axis=-1)

        lstm_seq = Bidirectional(LSTM(n_hidden, return_sequences=True, return_state=False))(embeddings)
        lstm_seq_d = Dropout(0.35)(lstm_seq)
        y_slot = TimeDistributed(Dense(n_slots, activation="softmax"))(lstm_seq_d)

        self.model = Model(inputs=[sentence_indices, case_indices], outputs=y_slot)
        self.model.summary()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def padded_seq(self, data_list):
        """
        Auxiliary helper function which takes a list of input indexed data and pads it. Used to simplify the sets of data
        associated with the training, validation and test set of the CONLL data used to train the model.
        :param data_list: list of input data which itself exists as list[list[int]]
        :return: list of padded sequence data as list[list[list[int]]]
        """
        max_len = self.params['max_len']
        padded_data = []
        for data in data_list:
            padded_data.append(pad_sequences(data, padding='post', maxlen=max_len, truncating='post'))

        return padded_data

    def train(self, train_data, val_data):
        """
        This method trains the sequence model. The training and validation data need to be passed as a list of list of
        ints; so this is data that has been tokenized and indexed by the tokenizer. The training history is saved as a
        pickle file and can be used to observe training and validation loss and accuracy.

        :param train_data: Indexed training data of the form list[list[int]]
        :param val_data: Indexed validation data of the form list[list[int]]
        """

        cfg = SolverConfig()

        train_sents_i, train_cases_i, train_tags_i = self.padded_seq(train_data)
        val_sents_i, val_cases_i, val_tags_i = self.padded_seq(val_data)

        history = self.model.fit([train_sents_i, train_cases_i], train_tags_i, validation_data=([val_sents_i, val_cases_i], val_tags_i),
                                 epochs=cfg.epochs, batch_size=cfg.batch_size, shuffle=True)

        pkl_dump(history.history, 'history')

    def predict(self, text):
        """
        This method is used to make predictions on the input text based on a trained sequence model. The model will have
        to be loaded before this method can be executed.
        :param text: Input text, can be a str, a list[str] or a list[list[str]]
        :return: Predicted NER tags as a list[list[str]]
        """
        if self.model is None:
            raise RuntimeError("NER Model is not loaded. Please load model.")

        if isinstance(text, str):
            text = [text]
        processed = self.tok_x.pre_process(text)
        cases = self.tok_c.text_to_indices(text)
        indices = self.tok_x.text_to_indices(processed)

        indices_pad, cases_pad = self.padded_seq([indices, cases])

        p_slots = self.model.predict([indices_pad, cases_pad])
        y_slots = np.argmax(p_slots, axis=-1)
        y_pred_i = []
        for i, sent in enumerate(processed):
            y_pred_i.append(list(y_slots[i, :len(sent)]))

        y_pred = self.tok_y.indices_to_text(y_pred_i)
        return y_pred