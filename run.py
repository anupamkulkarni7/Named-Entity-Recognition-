
"""
Named Entity Model training and evaluation based on the CONLL 2003 English dataset.

Usage:
    run.py train --train-src=<file> --val-src=<file> --glove-file=<file> [options]
    run.py test --test-src=<file> [options]

Options:
    -h --help                               Show this screen.
    --version                               1.0.1
    --train-src=<file>                      Train data source file
    --val-src=<file>                        Validation data source file
    --test-src=<file>                       Test source file
    --glove-file=<file>                     GloVe pre-trained vectors file (50 dim)
    --tok-tgt=<file>                        Tokenizer file source
    --model-tgt=<file>                      Model file source

"""


from docopt import docopt

from tokenizer import WordTokenizer, CaseTokenizer
from model import *
from utils import *
from seqeval.metrics import classification_report


def train(args):

    train_data_path = args["--train-src"]
    val_data_path = args["--val-src"]
    glove_file = args["--glove-file"]
    
    print("Reading and parsing in CONLL 2003 data...\nTrain data from {}\nValidation data from {}"
          .format(train_data_path, val_data_path))
    train_sents, train_tags = conll_parse(train_data_path)
    val_sents, val_tags = conll_parse(val_data_path)
    
    print("Initializing tokenizers for words, NER tags and word cases...\n")
    tok_x = WordTokenizer(from_pre=True)
    train_sents_p = tok_x.initialize(train_sents, glove_file)
    print("Vocab size for tok_x: {}".format(tok_x.vocab_size))
    tok_c = CaseTokenizer()
    print("Vocab size for tok_c: {}".format(tok_c.vocab_size))
    tok_y = WordTokenizer(oov_tok=None)
    _ = tok_y.initialize(train_tags)
    print("Vocab size for tok_y: {}\n".format(tok_y.vocab_size))

    train_sents_i = tok_x.text_to_indices(train_sents_p)
    train_cases_i = tok_c.text_to_indices(train_sents)
    train_tags_i = tok_y.text_to_indices(train_tags)

    val_sents_p = tok_x.pre_process(val_sents)
    val_sents_i = tok_x.text_to_indices(val_sents_p)
    val_cases_i = tok_c.text_to_indices(val_sents)
    val_tags_i = tok_y.text_to_indices(val_tags)

    train_data = [train_sents_i, train_cases_i, train_tags_i]
    val_data = [val_sents_i, val_cases_i, val_tags_i]

    print("Initializing NER model and beginning training...")
    ner_tagger = NERTagger(tok_x, tok_c, tok_y)
    ner_tagger.build()
    ner_tagger.train(train_data, val_data)

    if args["--model-tgt"] and args["--tok-tgt"]:
        ner_tagger.save(aux_file=args["--tok-tgt"], model_file=args["--model-tgt"])
    else:
        ner_tagger.save()


def evaluate(args):

    test_data_path = args['--test-src']
    print("Reading and parsing in CONLL 2003 data...\n")
    test_sents, test_tags = conll_parse(test_data_path)

    print("Initializing NER Model...")
    ner_tagger = NERTagger()
    if args["--model-tgt"] and args["--tok-tgt"]:
        ner_tagger.load(aux_file=args["--tok-tgt"], model_file=args["--model-tgt"])
        print("Model loaded successfully from {}".format(args["--model-tgt"]))
        print("Tokenizers loaded successfully from {}".format(args["--tok-tgt"]))
    else:
        ner_tagger.load()
        print("Model and tokenizers loaded successfully from default path.\n")

    pred_tags = ner_tagger.predict(test_sents)
    print(classification_report(test_tags, pred_tags))


def main():

    args = docopt(__doc__)

    if args['train']:
        train(args)
    elif args['test']:
        evaluate(args)


if __name__ == "__main__":
    main()