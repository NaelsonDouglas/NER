import spacy
from configs import configs
from singleton import Singleton

from logger import log

class _GenericPipeline(Singleton):
    @classmethod
    def get_instance(cls):
        return cls.instance

    def tokenize(self, text):
        if text is not None and len(text) > 0:
            result = self.nlp(text)
        else:
            result = self.nlp('')
        return result

    def build_sentences(self, text):
        doc = self.tokenize(text)
        sents = list(doc.sents)
        return sents

    def get(self):
        return self.nlp

    def _load_model(self):
        try:
            model = configs.SPACY_MODEL
            nlp = spacy.load(model)
        except OSError:
            msg = f'Error while trying to create instance the model {model}. You could install it using python -m spacy download {model}'
            log.error(msg)
            raise OSError()
        return nlp

class _Pipeline(_GenericPipeline):
    def __init__(self):
        if not hasattr(self, 'nlp'):
            self.nlp = self._load_model()
        else:
            self.get_instance()

nlp = _Pipeline().nlp