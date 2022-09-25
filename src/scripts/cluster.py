import pandas as pd
from thefuzz import process


if __name__ == '__main__':
    # model = spacy.load('../training/model-last')
    makes = pd.read_csv('../assets/makes.csv')
    data = makes.MAKE.values
    process.extractOne("beechcraft", data)