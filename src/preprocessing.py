import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


# Create reusable preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"       # adjective
    elif tag.startswith("V"):
        return "v"       # verb
    elif tag.startswith("N"):
        return "n"       # noun
    elif tag.startswith("R"):
        return "r"       # adverb
    else:
        return "n"       # noun


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = []

    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    # POS tagging
    pos_tags = pos_tag(filtered_tokens)

    # POS-aware lemmatization
    lemmatized_tokens = []

    for token, (_, tag) in zip(filtered_tokens, pos_tags):

        lemmatized_token = lemmatizer.lemmatize(
            token,
            get_wordnet_pos(tag)
        )

        lemmatized_tokens.append(lemmatized_token)

    return " ".join(lemmatized_tokens)

if __name__ == "__main__":

    sample_text = "I absolutely LOVED this movie!!!"

    result = preprocess_text(sample_text)

    print(result)