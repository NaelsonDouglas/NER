import pandas as pd
import spacy
from scripts.constants import KEEP_COLUMNS


if __name__ == '__main__':
    model = spacy.load('training/model-last')
    dataset = 'ds_13k.csv'
    dataset = 'big_dataset.csv'
    df = pd.read_csv(f'assets/{dataset}',on_bad_lines='warn')[KEEP_COLUMNS]
    df = df.loc[~df.title.isna()].reset_index(drop=True)
    sample = df.sample(500)
    sample['doc'] = sample.title.apply(model)
    sample['span'] = sample['doc'].apply(lambda x: x.spans['sc'])
    sample['tags'] = sample['span'].apply(lambda x: [tag.label_ for tag in x])
    for _, row in sample.iterrows():
        print(row.title,dict(zip(row.tags, row.span)), sep='\n--->',end='\n\n')