import pandas as pd
import spacy
import os
from spacy.tokens import Doc
from thefuzz import fuzz, process
import constants

this_path = os.path.dirname(os.path.abspath(__file__))
makes = pd.read_csv(this_path+'/makes.csv').iloc[:, 0].values
models = pd.read_csv(this_path+'/models.csv').iloc[:, 0].values

def extract_similar_make(doc:spacy.tokens.doc.Doc) -> tuple:
    return _extract_similar_tag(constants.MAKE, doc, makes, fallback_on_title=True)

def extract_similar_modelname(doc:spacy.tokens.doc.Doc) -> tuple:
    return _extract_similar_tag(constants.MODELNAME, doc, models, fallback_on_title=True)

def extract_similar_modelno(doc:spacy.tokens.doc.Doc) -> tuple:
    return _extract_similar_tag(constants.MODELNOQ, doc, models, fallback_on_title=False)

def _extract_similar_tag(feature:str, doc: spacy.tokens.doc.Doc, lookup_table:list, fallback_on_title:bool) -> str:
    spancat_text = [span.text for span in doc.spans['sc'] if span.label_ == feature.upper()]
    extracted = ('',0)
    if spancat_text:
        extracted = process.extractOne(spancat_text[0], lookup_table, scorer=fuzz.partial_ratio)
    elif fallback_on_title:
        extracted = process.extractOne(doc.text, lookup_table, scorer=fuzz.partial_ratio)
    if extracted and extracted[1] < constants.MIN_FUZZY_RATIO:
        extracted = ('',0)
    return extracted

if __name__ == '__main__':
    Doc.set_extension('make_normalized', getter=extract_similar_make)
    Doc.set_extension('model_normalized', getter=extract_similar_modelname)
    txt = '1990 eurocopterz\ AS 350BA Ecureuil for sale'
    model = spacy.load('../../training/model-last')
    # model.add_pipe('harmonizer')

    doc = model(txt)
    print(doc._.make_normalized)
    print(doc._.model_normalized)