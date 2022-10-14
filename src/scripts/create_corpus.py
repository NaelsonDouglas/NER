import json
import random
import re
from itertools import product
import math

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.matcher import Matcher
from spacy.tokens import DocBin
from tqdm import tqdm
import regex

from configs import configs
from logger import log
from constants import FEATURES, KEEP_COLUMNS, TITLE, MODELNOQ
from preprocessor import preprocess

nlp = spacy.load('en_core_web_sm')

class TagMaker:
    def build_tags(self, df:pd.DataFrame, ner_tags:list) -> list:
        patterns = dict()
        data = []
        for _, row in tqdm(list(df.iterrows())):
            title = row[TITLE]
            title = preprocess(title)
            if title:
                all_ners = []
                for tag in ner_tags:
                    entity = row[tag]
                    if entity:
                        _tag_ner = self._build_generic_taglist(title, entity, str(tag).upper())
                        all_ners = all_ners + [_tag_ner]
                if all_ners:
                    # train_cell = [title, {'entities': all_ners}]
                    all_ners = [tuple(ner) for ner in all_ners if ner]
                    train_cell = [title, all_ners]
                    data.append(train_cell)
        log.info('Spliting train and test_eval')
        # Tiny test dataset, small evaluation dataset and huge test dataset
        train, test_eval = train_test_split(data, test_size = 0.9)
        log.info('Spliting test and dev')
        dev, test = train_test_split(data, test_size = 0.95)
        train, test = train_test_split(data, test_size = 0.3)
        log.info('Persisting')
        train, test, dev = self.persist(train, test, dev)
        return train, test, dev

    def persist(self, train, test, dev) -> tuple:
        log.info('Persisting as json')
        with open('corpus/train.json', 'w') as train_io,\
            open('corpus/test.json', 'w') as test_io,\
            open('corpus/validation.json', 'w') as validation_io:
            train_io.write(json.dumps(train))
            test_io.write(json.dumps(test))
            validation_io.write(json.dumps(dev))
        log.info('Persisting as spacy')
        train = self.create_docbin(train, 'train')
        test = self.create_docbin(test, 'test')
        dev = self.create_docbin(dev, 'dev')
        return (train, test, dev)

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
                try:
                    span = doc.char_span(start, end, label=tag)
                except IndexError:
                    span = None
                if span:
                    spans_list.append(span)
                    doc.spans['sc'] = list(spans_list)
            if spans_list:
                doc_bin.add(doc)
        doc_bin.to_disk(f'corpus/{output_basename}.spacy')
        return doc_bin

    def find_match(self, ner:str, title:str):
        match = self._find_fuzzy_match(ner, title)
        return match

    def _build_generic_taglist(self, title, ner, ner_tag) -> list:
        ners = []
        match = self.find_match(ner,title)
        if match:
            start, end = match.span()
            ners = [start, end, ner_tag]
        return ners

    def _clear_empty_titles(self, df:pd.DataFrame) -> pd.DataFrame:
        return df[~df[TITLE].isna()]

    # def _find_match(self, ner:str, title:str):
    #     """
    #     Takes a ner text, like  PA-28-180, creates many variations with it replacing special characters (e.g. PA28180, PA.28.180)
    #     Then it finds if any of its variations are present on the title.
    #     """
    #     _ner = re.escape(ner.lower().replace(' ','\s'))
    #     title = title.lower()
    #     replaces = [' ', '-', '/', '.']
    #     replaces = product(replaces, replaces)
    #     replaces = [r for r in replaces if r[0]!=r[1]]
    #     patterns = []
    #     for (original, replacement) in replaces:
    #         _ner = re.escape(_ner.replace(original, replacement))
    #         patterns.append(_ner)
    #         _ner = re.escape(_ner.replace(original, ''))
    #         patterns.append(_ner)
    #     patterns.append(_ner)
    #     patterns = set(patterns)
    #     patterns = '|'.join(patterns)
    #     match = re.search(re.compile(patterns), title)
    #     return match

    def _find_fuzzy_match(self, ner:str, title:str) -> regex.Match:
        BEST_MATCH_FLAG = '(?b)'
        max_errors = math.floor(len(ner)/2)
        constraints = '{'+f'i<=1,d<={max_errors},s<=1,e<={max_errors}'+'}' #Max of 8 deletions and 3 substitutions (?b) filters to the best match
        pattern = f'({ner.lower()}){constraints}{BEST_MATCH_FLAG}'
        match = regex.search(pattern, title.lower())
        # if not match:
        #     print()
        #     print(title)
        #     print(ner)
        #     print()
        return match



if __name__ == '__main__':
    dataset = 'ds_13k.csv'
    dataset = 'big_dataset.csv'
    dataset = 'final_dataset.csv'
    try:
        df = pd.read_csv(f'assets/{dataset}',on_bad_lines='warn')
    except FileNotFoundError:
        df = pd.read_csv(f'../assets/{dataset}',on_bad_lines='warn')
    # df = df[KEEP_COLUMNS].dropna()
    df = df.query('~title.isnull()').sample(1200)
    df = df.fillna('')
    df = df.groupby([TITLE,MODELNOQ]).first().reset_index(drop=False)
    tag_maker = TagMaker()
    train, test, val = tag_maker.build_tags(df, FEATURES)
    # train = list(train.get_docs(nlp.vocab))


