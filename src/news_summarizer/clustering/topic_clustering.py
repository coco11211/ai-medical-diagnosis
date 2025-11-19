"""
Topic Clustering Module
Implements K-Means, LDA, and other topic modeling/clustering techniques.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class TopicClusterer:
    """Topic clustering and modeling for news articles."""

    def __init__(self, n_topics: int = 5, random_state: int = 42):
        """
        Initialize topic clusterer.

        Args:
            n_topics: Number of topics/clusters
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = None
        self.model = None

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for clustering."""
        return [text.strip() for text in texts if text and text.strip()]

    def kmeans_cluster(self, texts: List[str], n_clusters: Optional[int] = None,
                      max_features: int = 1000) -> Dict:
        """
        Cluster texts using K-Means on TF-IDF vectors.

        Args:
            texts: List of text documents
            n_clusters: Number of clusters (uses self.n_topics if None)
            max_features: Maximum number of features for TF-IDF

        Returns:
            Dictionary with clustering results
        """
        if n_clusters is None:
            n_clusters = self.n_topics

        texts = self._preprocess_texts(texts)

        if len(texts) < n_clusters:
            return {
                'method': 'kmeans',
                'error': f'Not enough texts ({len(texts)}) for {n_clusters} clusters',
                'success': False
            }

        try:
            # Create TF-IDF vectors
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                max_df=0.8,
                min_df=2
            )

            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Perform K-Means clustering
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )

            clusters = self.model.fit_predict(tfidf_matrix)

            # Calculate silhouette score
            if len(set(clusters)) > 1:
                silhouette = silhouette_score(tfidf_matrix, clusters)
            else:
                silhouette = 0.0

            # Get top terms for each cluster
            cluster_terms = self._get_kmeans_top_terms(n_clusters, top_n=10)

            # Organize texts by cluster
            clustered_texts = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_texts:
                    clustered_texts[cluster_id] = []
                clustered_texts[cluster_id].append({
                    'text': texts[idx],
                    'index': idx
                })

            return {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'clusters': clusters.tolist(),
                'cluster_terms': cluster_terms,
                'clustered_texts': clustered_texts,
                'silhouette_score': float(silhouette),
                'success': True
            }

        except Exception as e:
            return {
                'method': 'kmeans',
                'error': str(e),
                'success': False
            }

    def _get_kmeans_top_terms(self, n_clusters: int, top_n: int = 10) -> Dict:
        """Get top terms for each K-Means cluster."""
        cluster_terms = {}
        feature_names = self.vectorizer.get_feature_names_out()

        for cluster_id in range(n_clusters):
            center = self.model.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-top_n:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [float(center[i]) for i in top_indices]

            cluster_terms[cluster_id] = {
                'terms': top_words,
                'scores': top_scores
            }

        return cluster_terms

    def lda_topics(self, texts: List[str], n_topics: Optional[int] = None,
                  max_features: int = 1000, max_iter: int = 20) -> Dict:
        """
        Extract topics using Latent Dirichlet Allocation (LDA).

        Args:
            texts: List of text documents
            n_topics: Number of topics (uses self.n_topics if None)
            max_features: Maximum number of features
            max_iter: Maximum iterations for LDA

        Returns:
            Dictionary with topic modeling results
        """
        if n_topics is None:
            n_topics = self.n_topics

        texts = self._preprocess_texts(texts)

        if len(texts) < 2:
            return {
                'method': 'lda',
                'error': 'Need at least 2 texts for LDA',
                'success': False
            }

        try:
            # Create document-term matrix
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                max_df=0.8,
                min_df=2
            )

            doc_term_matrix = self.vectorizer.fit_transform(texts)

            # Perform LDA
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=self.random_state,
                max_iter=max_iter,
                learning_method='batch'
            )

            doc_topics = self.model.fit_transform(doc_term_matrix)

            # Get top terms for each topic
            topics = self._get_lda_topics(n_topics, top_n=10)

            # Assign dominant topic to each document
            dominant_topics = doc_topics.argmax(axis=1).tolist()

            # Organize texts by topic
            topic_texts = {}
            for idx, topic_id in enumerate(dominant_topics):
                if topic_id not in topic_texts:
                    topic_texts[topic_id] = []
                topic_texts[topic_id].append({
                    'text': texts[idx],
                    'index': idx,
                    'topic_distribution': doc_topics[idx].tolist()
                })

            return {
                'method': 'lda',
                'n_topics': n_topics,
                'topics': topics,
                'dominant_topics': dominant_topics,
                'topic_texts': topic_texts,
                'doc_topic_distribution': doc_topics.tolist(),
                'perplexity': float(self.model.perplexity(doc_term_matrix)),
                'success': True
            }

        except Exception as e:
            return {
                'method': 'lda',
                'error': str(e),
                'success': False
            }

    def _get_lda_topics(self, n_topics: int, top_n: int = 10) -> Dict:
        """Get top terms for each LDA topic."""
        topics = {}
        feature_names = self.vectorizer.get_feature_names_out()

        for topic_id in range(n_topics):
            topic_dist = self.model.components_[topic_id]
            top_indices = topic_dist.argsort()[-top_n:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [float(topic_dist[i]) for i in top_indices]

            topics[topic_id] = {
                'terms': top_words,
                'scores': top_scores
            }

        return topics

    def nmf_topics(self, texts: List[str], n_topics: Optional[int] = None,
                  max_features: int = 1000) -> Dict:
        """
        Extract topics using Non-negative Matrix Factorization (NMF).

        Args:
            texts: List of text documents
            n_topics: Number of topics
            max_features: Maximum number of features

        Returns:
            Dictionary with NMF topic modeling results
        """
        if n_topics is None:
            n_topics = self.n_topics

        texts = self._preprocess_texts(texts)

        if len(texts) < 2:
            return {
                'method': 'nmf',
                'error': 'Need at least 2 texts for NMF',
                'success': False
            }

        try:
            # Create TF-IDF matrix
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                max_df=0.8,
                min_df=2
            )

            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Perform NMF
            self.model = NMF(
                n_components=n_topics,
                random_state=self.random_state,
                init='nndsvda',
                max_iter=400
            )

            doc_topics = self.model.fit_transform(tfidf_matrix)

            # Get top terms for each topic
            topics = self._get_nmf_topics(n_topics, top_n=10)

            # Assign dominant topic
            dominant_topics = doc_topics.argmax(axis=1).tolist()

            # Organize texts by topic
            topic_texts = {}
            for idx, topic_id in enumerate(dominant_topics):
                if topic_id not in topic_texts:
                    topic_texts[topic_id] = []
                topic_texts[topic_id].append({
                    'text': texts[idx],
                    'index': idx,
                    'topic_distribution': doc_topics[idx].tolist()
                })

            return {
                'method': 'nmf',
                'n_topics': n_topics,
                'topics': topics,
                'dominant_topics': dominant_topics,
                'topic_texts': topic_texts,
                'reconstruction_error': float(self.model.reconstruction_err_),
                'success': True
            }

        except Exception as e:
            return {
                'method': 'nmf',
                'error': str(e),
                'success': False
            }

    def _get_nmf_topics(self, n_topics: int, top_n: int = 10) -> Dict:
        """Get top terms for each NMF topic."""
        topics = {}
        feature_names = self.vectorizer.get_feature_names_out()

        for topic_id in range(n_topics):
            topic_dist = self.model.components_[topic_id]
            top_indices = topic_dist.argsort()[-top_n:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [float(topic_dist[i]) for i in top_indices]

            topics[topic_id] = {
                'terms': top_words,
                'scores': top_scores
            }

        return topics

    def cluster_and_analyze(self, texts: List[str],
                           method: str = 'lda') -> Dict:
        """
        Perform topic clustering and analysis.

        Args:
            texts: List of text documents
            method: 'kmeans', 'lda', or 'nmf'

        Returns:
            Dictionary with clustering results
        """
        if method == 'kmeans':
            return self.kmeans_cluster(texts)
        elif method == 'lda':
            return self.lda_topics(texts)
        elif method == 'nmf':
            return self.nmf_topics(texts)
        else:
            raise ValueError(f"Unknown method: {method}")

    def find_optimal_clusters(self, texts: List[str],
                            min_clusters: int = 2,
                            max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            texts: List of text documents
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try

        Returns:
            Dictionary with optimal cluster analysis
        """
        texts = self._preprocess_texts(texts)

        if len(texts) < max_clusters:
            max_clusters = len(texts) - 1

        scores = []
        cluster_range = range(min_clusters, max_clusters + 1)

        for n in cluster_range:
            result = self.kmeans_cluster(texts, n_clusters=n)
            if result.get('success'):
                scores.append({
                    'n_clusters': n,
                    'silhouette_score': result['silhouette_score']
                })

        if scores:
            best = max(scores, key=lambda x: x['silhouette_score'])
            return {
                'optimal_clusters': best['n_clusters'],
                'optimal_score': best['silhouette_score'],
                'all_scores': scores
            }
        else:
            return {
                'optimal_clusters': self.n_topics,
                'error': 'Could not determine optimal clusters'
            }
