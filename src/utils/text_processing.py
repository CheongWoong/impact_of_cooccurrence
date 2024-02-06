from nltk.corpus import stopwords
from nltk import word_tokenize

stopword_list = stopwords.words("english")

filter = {}
for w in stopword_list:
    filter[w] = w
punctuations = {
    "?": "?",
    ":": ":",
    "!": "!",
    ".": ".",
    ",": ",",
    ";": ";"
}
filter.update(punctuations)
def filtering(text):
    if text in filter:
        return True

def text_normalization_without_lemmatization(text):
    result = []
    tokens = word_tokenize(text)
    
    for token in tokens:
        token_low = token.lower()
        if filtering(token_low):
            continue
        result.append(token_low)
    return result