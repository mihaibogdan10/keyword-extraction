from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        allowed_chars = '[0-9]|[a-z]|[\$\#\%\&\*\-\+\/\<\>\!\@\?\^]'
        self.tkn = RegexpTokenizer( r'(?:{0})*[a-z]+(?:{0})*'.format(allowed_chars))
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tkn.tokenize(doc)]
