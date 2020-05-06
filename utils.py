import re
import pickle
import os


def conll_parse(fpath):
    """
    Parses the documents as seen in the CoNLL 2003 dataset to return a list of sentences and NER tags.
    Note that no removal of special characters, case manipulation or any other text processing is done here.
    Note that each sentence and corresponding tags are return as a list of strings, NOT a List[List[str]].
    :param fpath: File path of input file
    :return: sents(List[str]), tags(List[str])
    """
    sents, tags = [], []
    sent, tag = [], []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('-DOCSTART-'):
                continue
            if line[0] == '\n' or len(line) == 0:
                if len(sent) > 0:
                    sents.append(sent)
                    tags.append(tag)
                    sent, tag = [], []
            else:
                w, _, _, ner = line.split()
                wa = re.sub('[\W_]+', '', w)
                if wa:
                    sent.append(w)
                    tag.append(ner)

    sents.append(sent)
    tags.append(tag)
    f.close()

    # sents = [' '.join(s) for s in sents]
    # tags = [' '.join(t) for t in tags]

    return sents, tags


def pkl_load(filepath):
    with open(os.path.join(filepath), "rb") as f:
        pdata = pickle.load(f)
    return pdata


def pkl_dump(pdata, filepath):
    with open(os.path.join(filepath), "wb") as f:
        pickle.dump(pdata, f)
