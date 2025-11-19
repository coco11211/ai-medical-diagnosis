"""
Knowledge Base System with Vector Search
Uses sentence transformers and FAISS for semantic search
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Vector-based knowledge base for customer service
    Supports semantic search using sentence embeddings
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        data_path: Optional[str] = None
    ):
        """
        Initialize knowledge base

        Args:
            embedding_model: Sentence transformer model name
            index_path: Path to saved FAISS index
            data_path: Path to saved knowledge base data
        """
        logger.info(f"Initializing knowledge base with model: {embedding_model}")

        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_with_ids = faiss.IndexIDMap(self.index)

        # Storage for knowledge entries
        self.entries: List[Dict] = []
        self.id_counter = 0

        # Load existing index if provided
        if index_path and data_path:
            self.load(index_path, data_path)

        logger.info(f"Knowledge base initialized with {len(self.entries)} entries")

    def add_entry(
        self,
        question: str,
        answer: str,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add an entry to the knowledge base

        Args:
            question: Question or topic
            answer: Answer or information
            category: Category/intent this belongs to
            keywords: Related keywords
            metadata: Additional metadata

        Returns:
            Entry ID
        """
        # Create entry
        entry = {
            'id': self.id_counter,
            'question': question,
            'answer': answer,
            'category': category or 'general',
            'keywords': keywords or [],
            'metadata': metadata or {},
            'embedding': None  # Will be computed
        }

        # Compute embedding
        text_to_embed = f"{question} {answer}"
        embedding = self.model.encode([text_to_embed])[0]
        entry['embedding'] = embedding

        # Add to FAISS index
        self.index_with_ids.add_with_ids(
            np.array([embedding], dtype=np.float32),
            np.array([self.id_counter], dtype=np.int64)
        )

        # Store entry
        self.entries.append(entry)

        logger.debug(f"Added entry {self.id_counter}: {question[:50]}...")

        self.id_counter += 1
        return entry['id']

    def add_entries_batch(self, entries: List[Dict]) -> List[int]:
        """
        Add multiple entries in batch

        Args:
            entries: List of entry dictionaries with 'question' and 'answer' keys

        Returns:
            List of entry IDs
        """
        ids = []
        for entry in entries:
            entry_id = self.add_entry(
                question=entry.get('question', ''),
                answer=entry.get('answer', ''),
                category=entry.get('category'),
                keywords=entry.get('keywords'),
                metadata=entry.get('metadata')
            )
            ids.append(entry_id)

        logger.info(f"Added {len(ids)} entries in batch")
        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search knowledge base using semantic similarity

        Args:
            query: Search query
            k: Number of results to return
            category_filter: Filter by category
            score_threshold: Minimum similarity threshold

        Returns:
            List of matching entries with scores
        """
        if not self.entries:
            logger.warning("Knowledge base is empty")
            return []

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Search in FAISS
        distances, indices = self.index_with_ids.search(
            np.array([query_embedding], dtype=np.float32),
            min(k * 2, len(self.entries))  # Get more for filtering
        )

        # Convert distances to similarity scores (cosine-like)
        # Lower distance = higher similarity
        max_dist = distances[0].max() if len(distances[0]) > 0 else 1.0
        similarities = 1 - (distances[0] / max(max_dist, 1.0))

        # Collect results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx == -1:  # Invalid index
                continue

            entry = self.entries[int(idx)]

            # Apply filters
            if category_filter and entry['category'] != category_filter:
                continue

            if similarity < score_threshold:
                continue

            results.append({
                'id': entry['id'],
                'question': entry['question'],
                'answer': entry['answer'],
                'category': entry['category'],
                'keywords': entry['keywords'],
                'metadata': entry['metadata'],
                'score': float(similarity)
            })

            if len(results) >= k:
                break

        logger.debug(f"Found {len(results)} results for query: {query[:50]}")

        return results

    def get_entry(self, entry_id: int) -> Optional[Dict]:
        """Get entry by ID"""
        for entry in self.entries:
            if entry['id'] == entry_id:
                return {
                    'id': entry['id'],
                    'question': entry['question'],
                    'answer': entry['answer'],
                    'category': entry['category'],
                    'keywords': entry['keywords'],
                    'metadata': entry['metadata']
                }
        return None

    def update_entry(self, entry_id: int, **kwargs):
        """Update an existing entry"""
        for entry in self.entries:
            if entry['id'] == entry_id:
                # Update fields
                for key, value in kwargs.items():
                    if key in entry and key != 'id':
                        entry[key] = value

                # Recompute embedding if question or answer changed
                if 'question' in kwargs or 'answer' in kwargs:
                    text_to_embed = f"{entry['question']} {entry['answer']}"
                    embedding = self.model.encode([text_to_embed])[0]
                    entry['embedding'] = embedding

                    # Update in FAISS
                    # Note: FAISS doesn't support direct update, so we'd need to rebuild
                    # For now, just update the entry
                    logger.warning("Embedding updated but FAISS index not rebuilt")

                logger.info(f"Updated entry {entry_id}")
                return True

        return False

    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry"""
        for i, entry in enumerate(self.entries):
            if entry['id'] == entry_id:
                self.entries.pop(i)
                logger.info(f"Deleted entry {entry_id}")
                # Note: Would need to rebuild FAISS index
                logger.warning("FAISS index not rebuilt after deletion")
                return True

        return False

    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        return list(set(entry['category'] for entry in self.entries))

    def get_entries_by_category(self, category: str) -> List[Dict]:
        """Get all entries in a category"""
        return [
            {
                'id': entry['id'],
                'question': entry['question'],
                'answer': entry['answer'],
                'category': entry['category'],
                'keywords': entry['keywords'],
                'metadata': entry['metadata']
            }
            for entry in self.entries
            if entry['category'] == category
        ]

    def save(self, index_path: str, data_path: str):
        """
        Save knowledge base to disk

        Args:
            index_path: Path to save FAISS index
            data_path: Path to save data (entries)
        """
        # Save FAISS index
        faiss.write_index(self.index_with_ids, index_path)

        # Save entries (without embeddings to save space)
        entries_to_save = []
        for entry in self.entries:
            entry_copy = entry.copy()
            entry_copy.pop('embedding', None)  # Remove embedding
            entries_to_save.append(entry_copy)

        with open(data_path, 'wb') as f:
            pickle.dump({
                'entries': entries_to_save,
                'id_counter': self.id_counter
            }, f)

        logger.info(f"Saved knowledge base to {index_path} and {data_path}")

    def load(self, index_path: str, data_path: str):
        """
        Load knowledge base from disk

        Args:
            index_path: Path to FAISS index
            data_path: Path to data file
        """
        try:
            # Load FAISS index
            self.index_with_ids = faiss.read_index(index_path)

            # Load entries
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.entries = data['entries']
                self.id_counter = data['id_counter']

            # Recompute embeddings (needed for updates)
            for entry in self.entries:
                text_to_embed = f"{entry['question']} {entry['answer']}"
                entry['embedding'] = self.model.encode([text_to_embed])[0]

            logger.info(f"Loaded knowledge base with {len(self.entries)} entries")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise

    def export_to_json(self, filepath: str):
        """Export knowledge base to JSON"""
        entries_export = []
        for entry in self.entries:
            entries_export.append({
                'id': entry['id'],
                'question': entry['question'],
                'answer': entry['answer'],
                'category': entry['category'],
                'keywords': entry['keywords'],
                'metadata': entry['metadata']
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entries_export, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported knowledge base to {filepath}")

    def import_from_json(self, filepath: str):
        """Import knowledge base from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        ids = self.add_entries_batch(entries)
        logger.info(f"Imported {len(ids)} entries from {filepath}")

        return ids

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        categories = {}
        for entry in self.entries:
            cat = entry['category']
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_entries': len(self.entries),
            'categories': categories,
            'embedding_dimension': self.embedding_dim,
            'model': self.model._modules['0'].auto_model.name_or_path
        }


# Testing function
def test_knowledge_base():
    """Test knowledge base"""
    kb = KnowledgeBase()

    # Add sample entries
    sample_entries = [
        {
            'question': 'How do I reset my password?',
            'answer': 'To reset your password, click on "Forgot Password" on the login page and follow the instructions sent to your email.',
            'category': 'account_issue',
            'keywords': ['password', 'reset', 'login']
        },
        {
            'question': 'What are your business hours?',
            'answer': 'Our customer service is available Monday-Friday 9AM-5PM EST, and Saturday 10AM-2PM EST.',
            'category': 'general_inquiry',
            'keywords': ['hours', 'time', 'when']
        },
        {
            'question': 'How can I track my order?',
            'answer': 'You can track your order by logging into your account and clicking on "Order History", or use the tracking number sent to your email.',
            'category': 'order_status',
            'keywords': ['track', 'order', 'shipping']
        },
        {
            'question': 'What is your refund policy?',
            'answer': 'We offer full refunds within 30 days of purchase. The item must be unused and in original packaging.',
            'category': 'return_exchange',
            'keywords': ['refund', 'return', 'money back']
        },
        {
            'question': 'How do I cancel my subscription?',
            'answer': 'To cancel your subscription, go to Account Settings > Billing > Cancel Subscription. You can also contact support for assistance.',
            'category': 'billing_payment',
            'keywords': ['cancel', 'subscription', 'billing']
        }
    ]

    print("Knowledge Base Test")
    print("=" * 70)

    # Add entries
    print("\nAdding entries...")
    kb.add_entries_batch(sample_entries)

    # Test search
    test_queries = [
        "I forgot my password, what should I do?",
        "When are you open?",
        "Where is my package?",
        "I want my money back",
        "How to stop my subscription"
    ]

    print("\nSearching knowledge base:")
    print("-" * 70)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = kb.search(query, k=2)

        if results:
            print(f"Top result (score: {results[0]['score']:.3f}):")
            print(f"  Q: {results[0]['question']}")
            print(f"  A: {results[0]['answer']}")
        else:
            print("  No results found")

    # Stats
    print("\n" + "=" * 70)
    print("Knowledge Base Stats:")
    stats = kb.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_knowledge_base()
