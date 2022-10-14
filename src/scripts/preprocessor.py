import re
try:
    from nltk.corpus import stopwords
except ImportError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
from constants import LOCAL_STOPWORDS

STOPWORDS = set([word for word in stopwords.words('english')])


def preprocess(title:str) -> str:
    title = title.strip()
    title = re.sub(r'\n|\t|\s{2,}', ' ', title)
    starting_year = r'^(19|20)\d{2}\s*' #starting
    title = re.sub(starting_year,'', title)
    title = [word for word in title.split(' ') if word.lower() not in STOPWORDS and word.lower() not in LOCAL_STOPWORDS]
    title = ' '.join(title)
    title = title.strip()
    return title


if __name__ == '__main__':
    txt = '1980 Nekter PA-34-220T Seneca III for sale'
    print(preprocess(txt))