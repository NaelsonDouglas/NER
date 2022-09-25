import pandas as pd
import spacy
import os
from spacy.tokens import Doc
from thefuzz import fuzz, process

this_path = os.path.dirname(os.path.abspath(__file__))
makes = pd.read_csv(this_path+'/makes.csv').iloc[:, 0].values
models = pd.read_csv(this_path+'/models.csv').iloc[:, 0].values

def extract_similar_make(doc:spacy.tokens.doc.Doc) -> tuple:
    return process.extractOne(doc.text, makes, scorer=fuzz.partial_ratio)

def extract_similar_model(doc:spacy.tokens.doc.Doc) -> tuple:
    return process.extractOne(doc.text, models, scorer=fuzz.partial_ratio)

if __name__ == '__main__':
    Doc.set_extension('make_normalized', getter=extract_similar_make)
    Doc.set_extension('model_normalized', getter=extract_similar_model)
    txt = '1990 eurocopterz\ AS 350BA Ecureuil for sale'
    model = spacy.load('../../training/model-last')
    # model.add_pipe('harmonizer')

    doc = model(txt)
    print(doc._.make_normalized)
    print(doc._.model_normalized)