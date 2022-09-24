import pandas as pd
import spacy
from scripts.constants import KEEP_COLUMNS, TITLE, MODELNOQ, FEATURES


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
    model = spacy.load('training/model-best')
    df = pd.read_csv(f'assets/{dataset}',on_bad_lines='warn')[KEEP_COLUMNS]
    df = df.query('~title.isnull()')
    df = df.fillna('')
    df = df.groupby([TITLE,MODELNOQ]).first().reset_index(drop=False)
    sample = df.sample(1500)
    sample['doc'] = sample.title.apply(model)
    sample['span'] = sample.doc.apply(lambda x: dict([(span.label_,str(span)) for span in x.spans['sc'] if str(span)]))
    sample = sample.drop(columns=['doc'])
    print('-------\n')
    for _, row in sample.iterrows():
        print('========= TITLE ==============')
        print(row.title)
        print()
        print('========= ANOTATIONS =========')
        print(row[FEATURES])
        print()
        print('========= DETECTED ===========')
        print(row.span)
        print('=============================')
        print()
        print('-------\n')