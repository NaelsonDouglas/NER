import random

import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import compounding, minibatch

from constants import MAKE, MODELNAME, TITLE
from data_preparation import TagMaker
from spacy_singleton import nlp

if __name__ == '__main__':
    df = pd.read_csv('20220811.csv')
    df = df[[TITLE, MAKE, MODELNAME]].dropna()
    tag_maker = TagMaker()
    TRAIN_DATA = tag_maker.build_tags(df, [MAKE])

