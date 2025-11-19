"""
Multi-Language Support Module
Handles language detection and translation
"""

from deep_translator import GoogleTranslator
from langdetect import detect, detect_langs, LangDetectException
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageManager:
    """
    Manages multi-language support including:
    - Language detection
    - Translation to/from English
    - Language-specific formatting
    """

    # Supported languages
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'ru': 'Russian',
        'ja': 'Japanese',
        'zh-CN': 'Chinese (Simplified)',
        'zh-TW': 'Chinese (Traditional)',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'cs': 'Czech',
        'ro': 'Romanian',
        'hu': 'Hungarian',
        'el': 'Greek',
        'he': 'Hebrew',
        'id': 'Indonesian',
        'ms': 'Malay',
        'uk': 'Ukrainian'
    }

    def __init__(self, default_language: str = 'en'):
        """
        Initialize language manager

        Args:
            default_language: Default language code (ISO 639-1)
        """
        self.default_language = default_language
        logger.info(f"Language manager initialized with default: {default_language}")

    def detect_language(self, text: str, return_all: bool = False) -> Dict:
        """
        Detect language of text

        Args:
            text: Input text
            return_all: If True, return all detected languages with probabilities

        Returns:
            Dictionary with language detection results
        """
        if not text or not text.strip():
            return {
                'language': self.default_language,
                'confidence': 0.0,
                'all_languages': []
            }

        try:
            if return_all:
                # Detect all possible languages
                langs = detect_langs(text)
                all_langs = [
                    {
                        'language': lang.lang,
                        'language_name': self.SUPPORTED_LANGUAGES.get(lang.lang, lang.lang),
                        'probability': round(lang.prob, 4)
                    }
                    for lang in langs
                ]

                primary = all_langs[0] if all_langs else {
                    'language': self.default_language,
                    'language_name': self.SUPPORTED_LANGUAGES[self.default_language],
                    'probability': 0.0
                }

                return {
                    'language': primary['language'],
                    'language_name': primary['language_name'],
                    'confidence': primary['probability'],
                    'all_languages': all_langs
                }
            else:
                # Detect primary language
                lang_code = detect(text)
                lang_name = self.SUPPORTED_LANGUAGES.get(lang_code, lang_code)

                return {
                    'language': lang_code,
                    'language_name': lang_name,
                    'confidence': 1.0,  # Single detection doesn't provide probability
                    'all_languages': []
                }

        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                'language': self.default_language,
                'language_name': self.SUPPORTED_LANGUAGES[self.default_language],
                'confidence': 0.0,
                'all_languages': []
            }

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict:
        """
        Translate text to target language

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)

        Returns:
            Dictionary with translation results
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': self.default_language,
                'target_language': target_language,
                'success': False
            }

        # Auto-detect source language if not provided
        if source_language is None:
            detection = self.detect_language(text)
            source_language = detection['language']

        # No translation needed if same language
        if source_language == target_language:
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language,
                'target_language': target_language,
                'success': True,
                'note': 'No translation needed - same language'
            }

        try:
            # Perform translation
            translator = GoogleTranslator(source=source_language, target=target_language)
            translated = translator.translate(text)

            return {
                'original_text': text,
                'translated_text': translated,
                'source_language': source_language,
                'source_language_name': self.SUPPORTED_LANGUAGES.get(source_language, source_language),
                'target_language': target_language,
                'target_language_name': self.SUPPORTED_LANGUAGES.get(target_language, target_language),
                'success': True
            }

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language,
                'target_language': target_language,
                'success': False,
                'error': str(e)
            }

    def translate_to_english(self, text: str, source_language: Optional[str] = None) -> Dict:
        """
        Translate text to English

        Args:
            text: Text to translate
            source_language: Source language (auto-detect if None)

        Returns:
            Translation result
        """
        return self.translate(text, 'en', source_language)

    def translate_from_english(self, text: str, target_language: str) -> Dict:
        """
        Translate from English to target language

        Args:
            text: English text
            target_language: Target language code

        Returns:
            Translation result
        """
        return self.translate(text, target_language, 'en')

    def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[Dict]:
        """
        Translate multiple texts

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language (auto-detect if None)

        Returns:
            List of translation results
        """
        results = []
        for text in texts:
            result = self.translate(text, target_language, source_language)
            results.append(result)

        return results

    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()

    def get_language_name(self, language_code: str) -> str:
        """Get language name from code"""
        return self.SUPPORTED_LANGUAGES.get(language_code, language_code)

    def create_multilingual_response(
        self,
        english_text: str,
        user_language: str
    ) -> Dict:
        """
        Create response in user's language

        Args:
            english_text: Response text in English
            user_language: User's detected language

        Returns:
            Response in both English and user's language
        """
        # If user speaks English, return as-is
        if user_language == 'en':
            return {
                'english': english_text,
                'user_language': english_text,
                'language_code': 'en',
                'translated': False
            }

        # Translate to user's language
        translation = self.translate_from_english(english_text, user_language)

        return {
            'english': english_text,
            'user_language': translation.get('translated_text', english_text),
            'language_code': user_language,
            'language_name': self.get_language_name(user_language),
            'translated': translation.get('success', False)
        }

    def process_user_input(self, text: str, session_language: Optional[str] = None) -> Dict:
        """
        Process user input for multilingual support

        Args:
            text: User input text
            session_language: Previously detected session language

        Returns:
            Processed input with language detection and translation
        """
        # Detect language
        detection = self.detect_language(text, return_all=True)
        detected_lang = detection['language']

        # Update session language if confidence is high
        if detection['confidence'] > 0.8:
            current_language = detected_lang
        elif session_language:
            current_language = session_language
        else:
            current_language = self.default_language

        # Translate to English for processing if needed
        if current_language != 'en':
            translation = self.translate_to_english(text, current_language)
            english_text = translation.get('translated_text', text)
            translation_success = translation.get('success', False)
        else:
            english_text = text
            translation_success = True

        return {
            'original_text': text,
            'english_text': english_text,
            'detected_language': detected_lang,
            'detected_language_name': detection['language_name'],
            'detection_confidence': detection['confidence'],
            'current_language': current_language,
            'translation_success': translation_success,
            'all_detected_languages': detection.get('all_languages', [])
        }

    def format_greeting(self, language: str) -> str:
        """Get greeting in specified language"""
        greetings = {
            'en': 'Hello! How can I help you today?',
            'es': '¡Hola! ¿Cómo puedo ayudarte hoy?',
            'fr': 'Bonjour! Comment puis-je vous aider aujourd\'hui?',
            'de': 'Hallo! Wie kann ich Ihnen heute helfen?',
            'it': 'Ciao! Come posso aiutarti oggi?',
            'pt': 'Olá! Como posso ajudá-lo hoje?',
            'nl': 'Hallo! Hoe kan ik u vandaag helpen?',
            'pl': 'Cześć! Jak mogę Ci dzisiaj pomóc?',
            'ru': 'Здравствуйте! Как я могу вам помочь сегодня?',
            'ja': 'こんにちは！今日はどのようにお手伝いできますか？',
            'zh-CN': '你好！今天我能帮你什么？',
            'ko': '안녕하세요! 오늘 무엇을 도와드릴까요?',
            'ar': 'مرحبا! كيف يمكنني مساعدتك اليوم؟',
            'hi': 'नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूँ?',
            'tr': 'Merhaba! Bugün size nasıl yardımcı olabilirim?',
            'vi': 'Xin chào! Tôi có thể giúp gì cho bạn hôm nay?',
        }

        return greetings.get(language, greetings['en'])


# Testing function
def test_language_manager():
    """Test language manager"""
    lm = LanguageManager()

    print("Language Manager Test")
    print("=" * 70)

    # Test language detection
    test_texts = [
        ("Hello, how are you?", "English"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hallo, wie geht es dir?", "German"),
        ("こんにちは、お元気ですか？", "Japanese"),
        ("你好，你好吗？", "Chinese"),
    ]

    print("\n1. Language Detection Test")
    print("-" * 70)
    for text, expected in test_texts:
        result = lm.detect_language(text)
        print(f"Text: {text}")
        print(f"Detected: {result['language_name']} ({result['language']}) "
              f"- Confidence: {result['confidence']:.3f}")
        print()

    # Test translation
    print("\n2. Translation Test")
    print("-" * 70)

    test_translation = "I need help with my account"
    target_languages = ['es', 'fr', 'de', 'ja']

    for lang in target_languages:
        result = lm.translate(test_translation, lang, 'en')
        if result['success']:
            print(f"{result['source_language_name']} → {result['target_language_name']}:")
            print(f"  Original: {result['original_text']}")
            print(f"  Translated: {result['translated_text']}")
            print()

    # Test multilingual response
    print("\n3. Multilingual Response Test")
    print("-" * 70)

    english_response = "Your account has been successfully updated."
    for lang in ['es', 'fr', 'ja']:
        response = lm.create_multilingual_response(english_response, lang)
        print(f"Language: {response['language_name']}")
        print(f"Response: {response['user_language']}")
        print()

    # Test greetings
    print("\n4. Greetings in Different Languages")
    print("-" * 70)
    for lang_code in ['en', 'es', 'fr', 'de', 'ja', 'zh-CN']:
        greeting = lm.format_greeting(lang_code)
        lang_name = lm.get_language_name(lang_code)
        print(f"{lang_name}: {greeting}")


if __name__ == "__main__":
    test_language_manager()
