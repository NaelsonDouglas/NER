import pandas as pd
from spacy.matcher import Matcher
import spacy
import json

from sklearn.model_selection import train_test_split
from configs import configs
from constants import MAKE, TITLE, MODELNAME
from spacy_singleton import nlp
from tqdm import tqdm

class TagMaker:
    def build_tags(self, df:pd.DataFrame, ner_tags:list) -> list:
        patterns = dict()
        data = []
        for _, row in tqdm(list(df.iterrows())):
            title = row[TITLE]
            if title:
                all_ners = []
                for tag in ner_tags:
                    _ners = self._build_generic_taglist(title, row[tag], tag.upper())
                    all_ners = all_ners + _ners
                if all_ners:
                    train_cell = [title, {'entities': all_ners}]
                    data.append(train_cell)
        train, test = train_test_split(data)
        with open('train.json', 'w') as train_io, open('test.json', 'w') as test_io:
            train_io.write(json.dumps(train))
            test_io.write(json.dumps(test))
        return train, test


    def _build_generic_taglist(self, title, ner, ner_tag) -> list:
        matcher = Matcher(nlp.vocab)
        pattern = [{'LOWER': ner.lower()}]
        matcher.add(ner.lower(), [pattern])
        doc = nlp(title)
        matches = matcher(doc)
        ners = []
        if matches:
            ners = list()
            for _, start, end in matches:
                start = doc[start].idx
                try:
                    end = doc[end].idx-1
                except IndexError:
                    end = doc[end-1].idx-1
                ner = [start, end, ner_tag]
                ners.append(ner)
        return ners

    def _clear_empty_titles(self, df:pd.DataFrame) -> pd.DataFrame:
        return df[~df[TITLE].isna()]