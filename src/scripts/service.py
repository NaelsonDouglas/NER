import pandas as pd
from constants import FEATURES, TITLE, MAKE_NORMALIZED, MODELNAME_NORMALIZED, MODELNO_NORMALIZED
from pipeline import model
from preprocessor import preprocess

def extract_tags(title:str) -> pd.DataFrame:
    df = pd.DataFrame([{'title':title, 'doc':None, 'span': None, MODELNAME_NORMALIZED:None, MAKE_NORMALIZED: None}])
    df['doc'] = df.title.apply(lambda x: model(preprocess(x)))
    df['span'] = _add_span(df)
    for tag in FEATURES:
        df[tag.upper()] = df['span'].apply(lambda x: x.get(tag.upper())).fillna('')
    df = df.drop(columns=['span'])
    df[MAKE_NORMALIZED] = df.doc.apply(lambda x: x._.make_normalized)
    df[MODELNAME_NORMALIZED] = df.doc.apply(lambda x: x._.model_normalized)
    df[MODELNO_NORMALIZED] = df.doc.apply(lambda x: x._.modelno_normalized)
    columns = [feature.upper() for feature in FEATURES] + [MAKE_NORMALIZED, MODELNAME_NORMALIZED, MODELNO_NORMALIZED]
    columns = sorted(columns)
    columns = [TITLE]+columns
    df = df[columns]
    return df

def _add_span(df:pd.DataFrame) -> pd.Series:
    return df.doc.apply(lambda x: dict([(span.label_,str(span)) for span in x.spans['sc'] if str(span)]))