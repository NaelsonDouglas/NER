import spacy
from spacy.tokens import DocBin
nlp = spacy.blank("en")

with open("corpus/train.spacy", "rb") as f:
    doc_bin = DocBin().from_bytes(f.read())
docs = list(doc_bin.get_docs(nlp.vocab))

for doc in docs:
    print(doc)
    for span in doc.spans['sc']:
        print(span,span.label_, sep=': ',end='\n')
    print(10*'=',end='\n\n')