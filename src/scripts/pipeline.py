import spacy
from harmonizer.harmonizer import extract_similar_model, extract_similar_make
from spacy.tokens import Doc

Doc.set_extension('make_normalized', getter=extract_similar_make)
Doc.set_extension('model_normalized', getter=extract_similar_model)
model = spacy.load('training/model-last')