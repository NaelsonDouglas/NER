import random

import pandas as pd
import spacy
from spacy.util import compounding, minibatch
from spacy.training import Example


from constants import MAKE, MODELNAME, TITLE
from data_preparation import TagMaker
from spacy_singleton import nlp


def disabled_pipes(nlp):
    keep = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
    disable = [pipe for pipe in nlp.pipe_names if pipe not in keep]
    return disable

def add_new_tags(train_data, ner_pipe):
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner_pipe.add_label(ent[2])
    return ner_pipe


if __name__ == '__main__':
    df = pd.read_csv('20220811.csv')
    df = df[[TITLE, MAKE, MODELNAME]].dropna()
    tag_maker = TagMaker()
    ner = nlp.get_pipe('ner')
    unaffected_pipes = disabled_pipes(nlp)

    TRAIN_DATA = tag_maker.build_tags(df, [MAKE])
    ner = add_new_tags(TRAIN_DATA, ner)

