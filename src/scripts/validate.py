import spacy
import pandas as pd
from constants import KEEP_COLUMNS

model = spacy.load('../training/model-last')


def get_tags(df:pd.DataFrame) -> pd.Series:
    tags = df.title.apply(_get_tags)
    tags = pd.DataFrame(list(tags))
    tags = tags.fillna('')
    tags['title'] = df.title.values
    result = pd.merge(tags,df, on='title')
    return result

def _get_tags(title:str) -> dict:
    doc = model(title)
    labels = [span.label_ for span in doc.spans['sc']]
    tags = [span.text for span in doc.spans['sc']]
    return dict(zip(labels, tags))


if __name__ == '__main__':
    dataset = 'ds_13k.csv'
    dataset = 'big_dataset.csv'
    # dataset = 'temp.csv'
    try:
        df = pd.read_csv(f'../assets/{dataset}',on_bad_lines='warn')
    except FileNotFoundError:
        df = pd.read_csv(f'{dataset}',on_bad_lines='warn')
    df = df[KEEP_COLUMNS].dropna()
    df = df.loc[~df.title.isna()].reset_index(drop=True)
    t = get_tags(df.sample(100))
