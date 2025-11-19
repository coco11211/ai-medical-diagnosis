"""
Entity Extraction Module
Extracts named entities and custom entities from text
"""

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import spacy
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Multi-method entity extractor using:
    - Transformer-based NER (BERT/RoBERTa)
    - spaCy NER
    - Custom regex patterns for domain-specific entities
    """

    # Custom entity patterns
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
        'order_number': r'\b(?:order|#|no\.?|number)\s*[:#]?\s*([A-Z0-9]{6,20})\b',
        'account_number': r'\b(?:account|acct)\s*[:#]?\s*([A-Z0-9]{6,15})\b',
        'tracking_number': r'\b(?:tracking|track)\s*[:#]?\s*([A-Z0-9]{10,30})\b',
        'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        'currency': r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?)',
        'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        'time': r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
        'product_code': r'\b(?:SKU|sku|product)\s*[:#]?\s*([A-Z0-9-]{4,15})\b',
    }

    def __init__(
        self,
        use_transformer: bool = True,
        transformer_model: str = "dslim/bert-base-NER",
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize entity extractor

        Args:
            use_transformer: Whether to use transformer-based NER
            transformer_model: HuggingFace model for NER
            use_spacy: Whether to use spaCy NER
            spacy_model: spaCy model name
        """
        self.use_transformer = use_transformer
        self.use_spacy = use_spacy

        # Initialize transformer NER
        if use_transformer:
            try:
                logger.info(f"Loading transformer NER model: {transformer_model}")
                self.ner_pipeline = pipeline(
                    "ner",
                    model=transformer_model,
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Transformer NER loaded successfully")
            except Exception as e:
                logger.error(f"Error loading transformer NER: {e}")
                self.use_transformer = False

        # Initialize spaCy NER
        if use_spacy:
            try:
                logger.info(f"Loading spaCy model: {spacy_model}")
                self.nlp = spacy.load(spacy_model)
                logger.info("spaCy NER loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading spaCy model: {e}")
                logger.info("Run: python -m spacy download en_core_web_sm")
                self.use_spacy = False

    def extract(self, text: str) -> Dict:
        """
        Extract all entities from text

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        if not text or not text.strip():
            return self._empty_result()

        entities = {
            'transformer_entities': [],
            'spacy_entities': [],
            'custom_entities': [],
            'consolidated_entities': []
        }

        try:
            # Transformer-based NER
            if self.use_transformer:
                entities['transformer_entities'] = self._extract_transformer(text)

            # spaCy NER
            if self.use_spacy:
                entities['spacy_entities'] = self._extract_spacy(text)

            # Custom pattern-based extraction
            entities['custom_entities'] = self._extract_custom(text)

            # Consolidate all entities
            entities['consolidated_entities'] = self._consolidate_entities(entities)

            logger.debug(f"Extracted {len(entities['consolidated_entities'])} entities")

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")

        return entities

    def _extract_transformer(self, text: str) -> List[Dict]:
        """Extract entities using transformer model"""
        try:
            results = self.ner_pipeline(text)

            entities = []
            for entity in results:
                entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'source': 'transformer'
                })

            return entities

        except Exception as e:
            logger.error(f"Error in transformer extraction: {e}")
            return []

    def _extract_spacy(self, text: str) -> List[Dict]:
        """Extract entities using spaCy"""
        try:
            doc = self.nlp(text)

            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'source': 'spacy'
                })

            return entities

        except Exception as e:
            logger.error(f"Error in spaCy extraction: {e}")
            return []

    def _extract_custom(self, text: str) -> List[Dict]:
        """Extract entities using custom patterns"""
        entities = []

        for entity_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                # Extract the actual value (use group 1 if it exists, else full match)
                value = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)

                entities.append({
                    'text': value.strip(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'source': 'pattern',
                    'full_match': match.group(0)
                })

        return entities

    def _consolidate_entities(self, entities_dict: Dict) -> List[Dict]:
        """
        Consolidate entities from multiple sources

        Removes duplicates and merges overlapping entities
        """
        all_entities = (
            entities_dict.get('transformer_entities', []) +
            entities_dict.get('spacy_entities', []) +
            entities_dict.get('custom_entities', [])
        )

        if not all_entities:
            return []

        # Sort by start position
        all_entities.sort(key=lambda x: x['start'])

        # Remove duplicates and overlaps
        consolidated = []
        for entity in all_entities:
            # Check if overlaps with existing entities
            overlaps = False
            for existing in consolidated:
                if self._entities_overlap(entity, existing):
                    # Keep the one with higher confidence or from better source
                    if self._entity_priority(entity) > self._entity_priority(existing):
                        consolidated.remove(existing)
                        consolidated.append(entity)
                    overlaps = True
                    break

            if not overlaps:
                consolidated.append(entity)

        # Sort again after consolidation
        consolidated.sort(key=lambda x: x['start'])

        return consolidated

    def _entities_overlap(self, e1: Dict, e2: Dict) -> bool:
        """Check if two entities overlap"""
        return not (e1['end'] <= e2['start'] or e2['end'] <= e1['start'])

    def _entity_priority(self, entity: Dict) -> int:
        """
        Get priority score for entity (higher is better)

        Priority order: pattern > transformer > spacy
        """
        source_priority = {
            'pattern': 3,  # Custom patterns are most reliable for domain entities
            'transformer': 2,
            'spacy': 1
        }

        priority = source_priority.get(entity.get('source', ''), 0)

        # Boost priority if entity has high confidence score
        if 'score' in entity:
            priority += entity['score']

        return priority

    def _empty_result(self) -> Dict:
        """Return empty result"""
        return {
            'transformer_entities': [],
            'spacy_entities': [],
            'custom_entities': [],
            'consolidated_entities': []
        }

    def extract_by_type(self, text: str, entity_type: str) -> List[str]:
        """
        Extract entities of a specific type

        Args:
            text: Input text
            entity_type: Type of entity to extract

        Returns:
            List of extracted entity values
        """
        entities = self.extract(text)
        consolidated = entities.get('consolidated_entities', [])

        return [
            e['text']
            for e in consolidated
            if e['type'].lower() == entity_type.lower()
        ]

    def get_entity_summary(self, text: str) -> Dict:
        """
        Get a summary of entities grouped by type

        Args:
            text: Input text

        Returns:
            Dictionary mapping entity types to lists of values
        """
        entities = self.extract(text)
        consolidated = entities.get('consolidated_entities', [])

        summary = {}
        for entity in consolidated:
            entity_type = entity['type']
            if entity_type not in summary:
                summary[entity_type] = []

            summary[entity_type].append({
                'text': entity['text'],
                'confidence': entity.get('score', 1.0)
            })

        return summary

    def highlight_entities(self, text: str) -> str:
        """
        Return text with entities highlighted

        Args:
            text: Input text

        Returns:
            Text with entities marked with [TYPE: value]
        """
        entities = self.extract(text)
        consolidated = entities.get('consolidated_entities', [])

        # Sort entities by start position in reverse to maintain positions
        consolidated.sort(key=lambda x: x['start'], reverse=True)

        highlighted = text
        for entity in consolidated:
            start, end = entity['start'], entity['end']
            entity_text = text[start:end]
            entity_type = entity['type']

            replacement = f"[{entity_type.upper()}: {entity_text}]"
            highlighted = highlighted[:start] + replacement + highlighted[end:]

        return highlighted


# Convenience import guard for torch
try:
    import torch
except ImportError:
    logger.warning("PyTorch not installed. Transformer NER will not be available.")


# Testing function
def test_entity_extractor():
    """Test entity extractor"""
    extractor = EntityExtractor()

    test_texts = [
        "My order number is #ABC123456 and I haven't received it yet.",
        "Contact me at john.doe@email.com or call 555-123-4567",
        "I was charged $149.99 on 12/15/2023 for account #ACC789012",
        "Track my package with tracking number 1Z999AA10123456784",
        "The issue started on January 15, 2024 at 3:30 PM",
        "I need help with product SKU: PROD-12345-XL"
    ]

    print("Entity Extraction Test")
    print("=" * 70)

    for text in test_texts:
        print(f"\nText: {text}")

        # Get entity summary
        summary = extractor.get_entity_summary(text)

        print("\nExtracted Entities:")
        for entity_type, entities in summary.items():
            print(f"  {entity_type}:")
            for entity in entities:
                conf = entity.get('confidence', 1.0)
                print(f"    - {entity['text']} (confidence: {conf:.3f})")

        # Show highlighted text
        highlighted = extractor.highlight_entities(text)
        print(f"\nHighlighted: {highlighted}")
        print("-" * 70)


if __name__ == "__main__":
    test_entity_extractor()
