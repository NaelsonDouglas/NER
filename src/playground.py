import pandas as pd
import spacy
from scripts.constants import KEEP_COLUMNS, TITLE, MODELNOQ


def apply(sample:pd.DataFrame, model, lowercase:bool=False) -> None:
    if lowercase:
        sample['doc'] = sample.title.apply(lambda x: model(x.lower()))
    else:
        sample['doc'] = sample.title.apply(model)
    sample['span'] = sample['doc'].apply(lambda x: x.spans['sc'])
    sample['tags'] = sample['span'].apply(lambda x: [tag.label_ for tag in x])
    for _, row in sample.iterrows():
        print(row.title,dict(zip(row.tags, row.span)), sep='\n--->',end='\n\n')
    return sample

if __name__ == '__main__':
    dataset = 'final_dataset.csv'
    df = pd.read_csv(f'assets/{dataset}',on_bad_lines='warn')[KEEP_COLUMNS]
    df = df.query('~title.isnull()')
    df = df.fillna('')
    df = df.groupby([TITLE,MODELNOQ]).first().reset_index(drop=False)
