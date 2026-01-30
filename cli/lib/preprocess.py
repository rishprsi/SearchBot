import string
from .search_utils import load_stopwords


def preprocess(word: str):
    # Change the case of all characters
    word = word.lower()

    # Create a maketrans that removes all the punctuations
    table = str.maketrans("", "", string.punctuation)
    # Remove all the punctuations
    word = word.translate(table)

    # Tokenize the words into a list of strings
    tokens = [x for x in word.split(" ") if x]

    # Remove st op words
    stopwords = load_stopwords()
    tokens = list(filter(lambda x: x not in stopwords, tokens))
    return tokens
