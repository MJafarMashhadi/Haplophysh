import tensorflow.keras as k
from sklearn.feature_extraction.text import CountVectorizer


def create_char_vectorizer(training_corpus):
    to_characters = k.preprocessing.text.Tokenizer(char_level=True, oov_token='<OOV>', filters='\t\n')
    to_characters.fit_on_texts(training_corpus)

    return to_characters


def create_word_vectorizer(training_corpus):
    word_vectorizer = CountVectorizer(
        stop_words=None,
        min_df=5,
        token_pattern=r'&\w+;|[:/&?=.\[\]\\]|%\w{2}|[-_\w\d]+',
        analyzer='word',
        max_features=500
    )
    word_vectorizer.fit(training_corpus)

    return word_vectorizer
