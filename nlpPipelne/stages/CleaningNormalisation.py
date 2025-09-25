import re
# import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import json
# from nlpPipelne.stages.TextExtraction import extract_text
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from googletrans import Translator

# Download necessary resources (run once)
# nltk.download("punkt", quiet=True)
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

translator = Translator()

async def translate_to_english(text: str) -> str:
    """Translate input text to English."""
    if not text:
        return ""
    translated = await translator.translate(text, dest='en')
    print(translated)
    return translated.text

def clean_text(text: str) -> str:
    """
    Basic cleaning of text:
    - Remove extra spaces, newlines
    - Remove special characters (except .,!? for sentence structure)
    - Convert to lowercase
    """
    if not text:
        return ""

    # Remove multiple spaces & newlines
    text = re.sub(r"\s+", " ", text)

    # Keep letters (all unicode letters), numbers, and basic punctuation
    text = re.sub(r"[^\w\s.,!?;:()\-]", "", text, flags=re.UNICODE)

    # Lowercase
    text = text.lower()

    return text


async def clean_normalise(stage1_result: dict) -> dict:
    """
    Update the original dict with translated, cleaned, and tokenized text info.
    """
    # Step 0: Translate to English first
    translated_text = await translate_to_english(stage1_result["raw_text"])

    # Step 1: Clean
    cleaned = clean_text(translated_text)

    # Step 2: Sentence tokenization
    sentences = sent_tokenize(cleaned)

    # Step 3: Word tokenization
    words = word_tokenize(cleaned)

    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) if re.match(r"[a-zA-Z]", w) else w for w in words]

    # Remove punctuation-only tokens & normalize numbers
    words = [re.sub(r"\d+", "<NUM>", w) for w in words if re.match(r"[a-zA-Z0-9]", w)]

    # Update the original dict
    stage1_result.update({
        "translated_text": translated_text,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "words": words,
        "num_sentences": len(sentences),
        "num_words": len(words),
    })

    return stage1_result
