"""Text preprocessing for medical NLP."""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from loguru import logger


class TextPreprocessor:
    """Preprocess medical text for NLP analysis."""

    def __init__(
        self,
        remove_stopwords: bool = False,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        lemmatize: bool = True
    ):
        """
        Initialize text preprocessor.

        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            lemmatize: Whether to lemmatize words
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Medical terms to preserve (not remove even if stopwords)
        self.medical_terms_preserve = {
            'pain', 'ache', 'fever', 'cough', 'cold', 'hot',
            'blood', 'heart', 'lung', 'stomach', 'head'
        }

        logger.info("Initialized TextPreprocessor")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove numbers (but keep if part of medical terms like "24 hours")
        # text = re.sub(r'\b\d+\b', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_punctuation_func(self, text: str) -> str:
        """
        Remove punctuation from text.

        Args:
            text: Input text

        Returns:
            Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return word_tokenize(text)

    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        # Keep medical terms even if they're stopwords
        filtered = [
            token for token in tokens
            if token.lower() not in self.stop_words or token.lower() in self.medical_terms_preserve
        ]
        return filtered

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.

        Args:
            tokens: List of tokens

        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = self.remove_punctuation_func(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)

        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)

        # Join back to string
        processed_text = ' '.join(tokens)

        return processed_text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptom keywords from text.

        Args:
            text: Input text

        Returns:
            List of extracted symptoms
        """
        # Common symptom keywords
        symptom_keywords = {
            'pain', 'ache', 'fever', 'cough', 'nausea', 'vomiting',
            'diarrhea', 'constipation', 'headache', 'dizziness', 'fatigue',
            'weakness', 'shortness', 'breath', 'chest', 'abdominal',
            'swelling', 'rash', 'itching', 'bleeding', 'discharge',
            'burning', 'numbness', 'tingling', 'cramping', 'stiffness'
        }

        # Tokenize and lowercase
        tokens = [token.lower() for token in self.tokenize(text)]

        # Extract symptoms
        symptoms = [token for token in tokens if token in symptom_keywords]

        return list(set(symptoms))  # Remove duplicates

    def extract_body_parts(self, text: str) -> List[str]:
        """
        Extract body part mentions from text.

        Args:
            text: Input text

        Returns:
            List of extracted body parts
        """
        # Common body parts
        body_parts = {
            'head', 'neck', 'chest', 'abdomen', 'back', 'arm', 'leg',
            'hand', 'foot', 'heart', 'lung', 'stomach', 'liver', 'kidney',
            'brain', 'eye', 'ear', 'nose', 'throat', 'skin', 'bone', 'muscle'
        }

        # Tokenize and lowercase
        tokens = [token.lower() for token in self.tokenize(text)]

        # Extract body parts
        parts = [token for token in tokens if token in body_parts]

        return list(set(parts))  # Remove duplicates
