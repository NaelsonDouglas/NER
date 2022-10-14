import json

import spacy
from spacy.tokens import DocBin

nlp = spacy.blank('en')

def convert(training_data:list, train:bool) -> bool:
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        try:
            doc.ents = ents
        except:
            breakpoint()
        db.add(doc)
    if train:
        db.to_disk('./train.spacy')
    else:
        db.to_disk('./test.spacy')
    return True


if __name__ == '__main__':
    with open('train.json', 'r') as train, open('test.json', 'r') as test:
        train = json.load(train)
        test = json.load(test)
        convert(train, True)
        convert(test, False)
