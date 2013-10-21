import csv
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DOCUMENTS_COUNT = 13000
#Will consider that tokens appearing in more than 
#RELEVANCE_THRESHOLD * DOCUMENTS_COUNT documents
#are not relevant to the meaning of any document
RELEVANCE_THRESHOLD = 0.05

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tkn = RegexpTokenizer(r'(?:[0-9]|[a-z]|[\$\#\%\&\*\-])*[a-z]+(?:[0-9]|[a-z]|[\$\#\%\&\*\-])*')
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tkn.tokenize(doc)]

class VocabularyTokenizer(object):
    def __init__(self, vocabulary):
        self.wnl = WordNetLemmatizer()
        self.tkn = RegexpTokenizer(r'(?:[0-9]|[a-z]|[\$\#\%\&\*\-])*[a-z]+(?:[0-9]|[a-z]|[\$\#\%\&\*\-])*')
        self.vocabulary = vocabulary
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tkn.tokenize(doc) if self.wnl.lemmatize(t) in vocabulary]


def getRelevantVocabulary(corpus):
    tokenize = LemmaTokenizer()
    token_frequency = {}
    for index, document in enumerate(corpus):
        tokens = set(tokenize(document))
        for token in tokens:
            if token not in token_frequency:
                token_frequency[token] = 1
            else:
                token_frequency[token] += 1

    token_frequency_list = []
    for key in token_frequency.iterkeys():
        token_frequency_list.append((token_frequency[key], key))

    token_frequency_list = sorted(token_frequency_list, reverse=True)
    return set([token for count, token in token_frequency_list if count < RELEVANCE_THRESHOLD * DOCUMENTS_COUNT])


def euclidianDistanceSquared(x, y):
    assert(len(x) == len(y))
    return sum((x[k] - y[k])**2 for k in xrange(len(x)))


corpus = []
keywords = []
for index, row in enumerate(csv.reader(open("Train_no_duplicates.csv"))):
    if index == 0:
        columns = row 
    else:
        title = row[1].lower()
        #print title
        description = row[2].lower()
        keywords.append(row[3])
        corpus.append(title)
        #corpus.append(title + ' ' + description)
        #print title
        #print description
        if index == DOCUMENTS_COUNT:
            break

vocabulary = getRelevantVocabulary(corpus)

vectorizer = CountVectorizer(tokenizer=VocabularyTokenizer(vocabulary))
transformer = TfidfTransformer()
tfvectorizer = TfidfVectorizer(tokenizer=VocabularyTokenizer(vocabulary))
neighbors = NearestNeighbors(n_neighbors=5)
columns = []
all_tokens = {}


X = vectorizer.fit_transform(corpus)
neighbors.fit(X)
distances, indeces = neighbors.kneighbors(X)
neighbours_group = indeces[0]
#print neighbours_group
#vocabulary = vectorizer.get_feature_names()
#temp_X = X.toarray()

for index in neighbours_group:
    #for i, feature in enumerate(temp_X[index]):
    #    if feature != 0:
    #        print feature, vocabulary[i]
    print corpus[index], ": ", keywords[index], "\n\n\n"

#first_feature_vector = temp_X[0]
#distances = []

#for index, feature_vector in enumerate(temp_X):
#    new_dist =  euclidianDistanceSquared(first_feature_vector, feature_vector)
#    distances.append((new_dist, index))

#sorted_distances = sorted(distances)
#for dist, index in sorted_distances[:5]:
#    for i, feature in enumerate(temp_X[index]):
#        if feature != 0:
#            print feature, vocabulary[i]
#    print corpus[index], ": ", keywords[index]
#    print 'distance: ', dist, '\n\n\n'

#print X.toarray()
#TX = transformer.fit_transform(X.toarray()).toarray()
#print TX

