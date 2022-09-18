import json
import random

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.matcher import Matcher
from spacy.tokens import DocBin
from tqdm import tqdm

from configs import configs
from constants import FEATURES, KEEP_COLUMNS, MAKE, MODELNAME, MODELNO, MODELNOQ, TITLE

nlp = spacy.load('en_core_web_sm')

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
                    # train_cell = [title, {'entities': all_ners}]
                    all_ners = [tuple(ner) for ner in all_ners]
                    train_cell = [title, all_ners]
                    data.append(train_cell)
        train, test_eval = train_test_split(data, test_size = 0.8)
        test, validation = train_test_split(data, test_size = 0.5)

        self.create_docbin(train, 'train')
        self.create_docbin(test, 'test')
        self.create_docbin(validation, 'validation')
        with open('corpus/train.json', 'w') as train_io,\
            open('corpus/test.json', 'w') as test_io,\
            open('corpus/validation.json', 'w') as validation_io:
            train_io.write(json.dumps(train))
            test_io.write(json.dumps(test))
            validation_io.write(json.dumps(validation))
        return train, test, validation

    def create_docbin(self, tagged_text:list, output_basename:str) -> DocBin:
        """
        Gets a list of tagged texts on the format (text, [(start1, end1, tag1), (start2, end2, tag2),...(startn,end, tagn) ])
        and bundles it inside a DocBin containing the docs built upont the `text` with the spancat tags

        `tagged_text` is buit by the method build_tags
        """
        doc_bin = DocBin()
        for text, spans in tagged_text:
            doc = nlp(text)
            spans_list = list()
            for (start, end, tag) in spans:
                span = doc.char_span(start, end, label=tag)
                if span:
                    spans_list.append(span)
                doc.set_ents(spans_list)
                doc.spans[tag] = list(doc.ents)
                doc_bin.add(doc)
        doc_bin.to_disk(f'corpus/{output_basename}.spacy')
        return doc_bin

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



if __name__ == '__main__':
    try:
        df = pd.read_csv('assets/ds_13k.csv',on_bad_lines='warn')
    except FileNotFoundError:
        df = pd.read_csv('ds_13k.csv',on_bad_lines='warn')
    df = df[KEEP_COLUMNS].dropna().sample(30)
    tag_maker = TagMaker()
    TRAIN_DATA, TEST_DATA, VALIDATION_DATA = tag_maker.build_tags(df, FEATURES)

