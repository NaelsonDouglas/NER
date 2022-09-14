import random

import pandas as pd

from constants import KEEP_COLUMNS,FEATURES, TITLE, MAKE, MODELNAME, MODELNO, MODELNOQ
from data_preparation import TagMaker
from spacy_singleton import nlp


if __name__ == '__main__':
    # df = pd.read_csv('20220811.csv')
    df = pd.read_csv('20220912.csv',on_bad_lines='warn')
    df = df[KEEP_COLUMNS].dropna().sample(10)
    tag_maker = TagMaker()
    TRAIN_DATA = tag_maker.build_tags(df, [FEATURES])

