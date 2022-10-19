import spacy
from harmonizer.harmonizer import extract_similar_modelname, extract_similar_make, extract_similar_modelno
from spacy.tokens import Doc

Doc.set_extension('make_normalized', getter=extract_similar_make)
Doc.set_extension('model_normalized', getter=extract_similar_modelname)
Doc.set_extension('modelno_normalized', getter=extract_similar_modelno)
model = spacy.load('training/model-best')