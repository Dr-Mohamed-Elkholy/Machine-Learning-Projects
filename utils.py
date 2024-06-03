import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords and tokenizer
sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

# Utility functions
def save_file(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load_file(name):
    return pickle.load(open(name, "rb"))

# Text preprocessing function
def process_text(review, stem='p'):
    review = review.lower()
    tokens = word_tokenize(review)
    tokens = [t for t in tokens if t not in sw]
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 0]
    tokens = ["".join(t) for t in tokens]
    if stem == 'p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()
    else:
        raise Exception("stem has to be either 'p' for Porter or 'l' for Lancaster")
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Vectorization function
def vectorize(token_list, y, vect='bow', min_df=5, ng_low=1, ng_high=3, test_size=0.2, rs=42):
    if vect == 'bow':
        vectorizer = CountVectorizer(min_df=min_df)
    elif vect == 'bowb':
        vectorizer = CountVectorizer(binary=True, min_df=min_df)
    elif vect == 'ng':
        vectorizer = CountVectorizer(min_df=min_df, ngram_range=(ng_low, ng_high))
    elif vect == 'tf':
        vectorizer = TfidfVectorizer(min_df=min_df)
    else:
        raise Exception("vect has to be one of 'bow', 'bowb', 'ng', 'tf'")
    X = vectorizer.fit_transform(token_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)
    return X_train, X_test, y_train, y_test, vectorizer
