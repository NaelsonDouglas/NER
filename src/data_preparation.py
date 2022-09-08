import pandas as pd
from spacy.matcher import Matcher
import spacy

from configs import configs
from constants import MAKE, TITLE
from spacy_singleton import nlp
from tqdm import tqdm

class TagMaker:
    def build_make_tags(self, df:pd.DataFrame) -> list:
        patterns = dict()
        data = []
        for _, row in tqdm(df.iterrows()):
            title = row[TITLE]
            matcher = Matcher(nlp.vocab)
            if title:
                make = row[MAKE]
                pattern = [{'LOWER': make.lower()}]
                matcher.add(make.lower(), [pattern])
                doc = nlp(title)
                matches = matcher(doc)
                ners = []
                if matches:
                    ners = list()
                    for _, start, end in matches:
                        start = doc[start].idx
                        end = doc[end].idx-1
                        ner = (start, end, 'MAKE')
                        ners.append(ner)
                    data.append((title, {'entities': ners}))
        return data

    def _clear_empty_titles(self, df:pd.DataFrame) -> pd.DataFrame:
        return df[~df[TITLE].isna()]

if __name__ == '__main__':
    tag_maker = TagMaker()
    df = pd.read_csv('20220811.csv')
    df = df[[TITLE, MAKE]].dropna().head(150)
    s = tag_maker.build_make_tags(df)
