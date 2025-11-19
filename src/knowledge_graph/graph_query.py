"""Medical knowledge graph query engine."""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from loguru import logger


class KnowledgeGraphQuery:
    """Query medical knowledge graph."""

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize knowledge graph query engine.

        Args:
            graph: Medical knowledge graph
        """
        self.graph = graph
        logger.info("Initialized KnowledgeGraphQuery")

    def get_disease_by_symptoms(
        self,
        symptoms: List[str],
        threshold: float = 0.5
    ) -> List[Dict[str, any]]:
        """
        Find diseases matching given symptoms.

        Args:
            symptoms: List of symptom names or IDs
            threshold: Minimum matching score (0-1)

        Returns:
            List of matching diseases with scores
        """
        # Find symptom nodes
        symptom_nodes = self._find_nodes_by_name(symptoms, 'symptom')

        if not symptom_nodes:
            logger.warning(f"No symptom nodes found for: {symptoms}")
            return []

        # Find diseases connected to these symptoms
        disease_scores = {}

        for symptom_node in symptom_nodes:
            # Get all diseases with this symptom
            for pred in self.graph.predecessors(symptom_node):
                if self.graph.nodes[pred].get('type') == 'disease':
                    edge_data = self.graph.get_edge_data(pred, symptom_node)
                    weight = edge_data.get('weight', 1.0)

                    if pred not in disease_scores:
                        disease_scores[pred] = 0
                    disease_scores[pred] += weight

        # Normalize scores
        max_score = len(symptom_nodes)
        normalized_scores = {
            disease: score / max_score
            for disease, score in disease_scores.items()
        }

        # Filter by threshold and prepare results
        results = []
        for disease_id, score in normalized_scores.items():
            if score >= threshold:
                disease_data = self.graph.nodes[disease_id]
                results.append({
                    'disease_id': disease_id,
                    'name': disease_data.get('name', 'Unknown'),
                    'category': disease_data.get('category'),
                    'match_score': score,
                    'description': disease_data.get('description')
                })

        # Sort by score
        results.sort(key=lambda x: x['match_score'], reverse=True)

        logger.info(f"Found {len(results)} diseases matching symptoms")
        return results

    def get_symptoms_by_disease(
        self,
        disease_name: str
    ) -> List[Dict[str, any]]:
        """
        Get symptoms for a specific disease.

        Args:
            disease_name: Disease name or ID

        Returns:
            List of symptoms
        """
        # Find disease node
        disease_nodes = self._find_nodes_by_name([disease_name], 'disease')

        if not disease_nodes:
            logger.warning(f"Disease not found: {disease_name}")
            return []

        disease_node = disease_nodes[0]

        # Get symptoms
        symptoms = []
        for successor in self.graph.successors(disease_node):
            if self.graph.nodes[successor].get('type') == 'symptom':
                edge_data = self.graph.get_edge_data(disease_node, successor)
                symptom_data = self.graph.nodes[successor]

                symptoms.append({
                    'symptom_id': successor,
                    'name': symptom_data.get('name', 'Unknown'),
                    'frequency': edge_data.get('weight'),
                    'severity': edge_data.get('severity'),
                    'description': symptom_data.get('description')
                })

        # Sort by frequency
        symptoms.sort(key=lambda x: x['frequency'] or 0, reverse=True)

        return symptoms

    def get_treatments_by_disease(
        self,
        disease_name: str
    ) -> List[Dict[str, any]]:
        """
        Get treatments for a specific disease.

        Args:
            disease_name: Disease name or ID

        Returns:
            List of treatments
        """
        # Find disease node
        disease_nodes = self._find_nodes_by_name([disease_name], 'disease')

        if not disease_nodes:
            logger.warning(f"Disease not found: {disease_name}")
            return []

        disease_node = disease_nodes[0]

        # Get treatments
        treatments = []
        for successor in self.graph.successors(disease_node):
            if self.graph.nodes[successor].get('type') == 'treatment':
                edge_data = self.graph.get_edge_data(disease_node, successor)
                treatment_data = self.graph.nodes[successor]

                treatments.append({
                    'treatment_id': successor,
                    'name': treatment_data.get('name', 'Unknown'),
                    'type': treatment_data.get('treatment_type'),
                    'effectiveness': edge_data.get('weight'),
                    'first_line': edge_data.get('first_line', False),
                    'description': treatment_data.get('description')
                })

        # Sort by effectiveness and first-line
        treatments.sort(key=lambda x: (x['first_line'], x['effectiveness'] or 0), reverse=True)

        return treatments

    def get_related_diseases(
        self,
        disease_name: str,
        max_results: int = 5
    ) -> List[Dict[str, any]]:
        """
        Find diseases related to the given disease (based on shared symptoms).

        Args:
            disease_name: Disease name or ID
            max_results: Maximum number of results

        Returns:
            List of related diseases
        """
        # Get symptoms of the disease
        symptoms = self.get_symptoms_by_disease(disease_name)

        if not symptoms:
            return []

        # Get symptom IDs
        symptom_ids = [s['symptom_id'] for s in symptoms]

        # Find other diseases with these symptoms
        disease_scores = {}

        for symptom_id in symptom_ids:
            for pred in self.graph.predecessors(symptom_id):
                pred_data = self.graph.nodes[pred]
                if pred_data.get('type') == 'disease' and pred_data.get('name') != disease_name:
                    edge_data = self.graph.get_edge_data(pred, symptom_id)
                    weight = edge_data.get('weight', 1.0)

                    if pred not in disease_scores:
                        disease_scores[pred] = 0
                    disease_scores[pred] += weight

        # Prepare results
        results = []
        for disease_id, score in disease_scores.items():
            disease_data = self.graph.nodes[disease_id]
            results.append({
                'disease_id': disease_id,
                'name': disease_data.get('name', 'Unknown'),
                'category': disease_data.get('category'),
                'similarity_score': score,
                'description': disease_data.get('description')
            })

        # Sort by score and limit
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results[:max_results]

    def get_differential_diagnosis(
        self,
        symptoms: List[str],
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Get differential diagnosis based on symptoms.

        Args:
            symptoms: List of symptoms
            top_k: Number of top diagnoses to return

        Returns:
            List of differential diagnoses
        """
        # Get all matching diseases
        diseases = self.get_disease_by_symptoms(symptoms, threshold=0.0)

        # Get additional information for each disease
        for disease in diseases:
            # Get all symptoms
            all_symptoms = self.get_symptoms_by_disease(disease['disease_id'])

            # Calculate symptom coverage
            matched_symptoms = len(symptoms)
            total_symptoms = len(all_symptoms)
            coverage = matched_symptoms / total_symptoms if total_symptoms > 0 else 0

            disease['symptom_coverage'] = coverage
            disease['total_symptoms'] = total_symptoms
            disease['matched_symptoms'] = matched_symptoms

            # Get treatments
            treatments = self.get_treatments_by_disease(disease['disease_id'])
            disease['available_treatments'] = len(treatments)

        # Sort by match score and coverage
        diseases.sort(key=lambda x: (x['match_score'], x['symptom_coverage']), reverse=True)

        return diseases[:top_k]

    def _find_nodes_by_name(
        self,
        names: List[str],
        node_type: Optional[str] = None
    ) -> List[str]:
        """
        Find nodes by name.

        Args:
            names: List of names to search for
            node_type: Optional node type filter

        Returns:
            List of node IDs
        """
        found_nodes = []

        for node, data in self.graph.nodes(data=True):
            # Check type filter
            if node_type and data.get('type') != node_type:
                continue

            # Check if name matches
            node_name = data.get('name', '').lower()
            node_id = str(node).lower()

            for name in names:
                name_lower = name.lower()
                if name_lower == node_name or name_lower == node_id:
                    found_nodes.append(node)
                    break

        return found_nodes

    def search_by_category(
        self,
        category: str,
        node_type: str = 'disease'
    ) -> List[Dict[str, any]]:
        """
        Search nodes by category.

        Args:
            category: Category to search for
            node_type: Type of nodes to search

        Returns:
            List of matching nodes
        """
        results = []

        for node, data in self.graph.nodes(data=True):
            if data.get('type') == node_type and data.get('category', '').lower() == category.lower():
                results.append({
                    'id': node,
                    'name': data.get('name'),
                    'category': data.get('category'),
                    'description': data.get('description')
                })

        return results

    def get_shortest_path(
        self,
        source_name: str,
        target_name: str
    ) -> List[Tuple[str, str]]:
        """
        Find shortest path between two nodes.

        Args:
            source_name: Source node name
            target_name: Target node name

        Returns:
            List of (node, relationship_type) tuples representing the path
        """
        # Find nodes
        source_nodes = self._find_nodes_by_name([source_name])
        target_nodes = self._find_nodes_by_name([target_name])

        if not source_nodes or not target_nodes:
            logger.warning(f"Could not find nodes: {source_name} or {target_name}")
            return []

        source = source_nodes[0]
        target = target_nodes[0]

        try:
            path = nx.shortest_path(self.graph, source, target)

            # Build path with relationship types
            path_with_relationships = []
            for i in range(len(path) - 1):
                node = path[i]
                next_node = path[i + 1]
                edge_data = self.graph.get_edge_data(node, next_node)
                rel_type = edge_data.get('type', 'connected_to')

                node_data = self.graph.nodes[node]
                path_with_relationships.append((
                    node_data.get('name', node),
                    rel_type
                ))

            # Add final node
            final_data = self.graph.nodes[target]
            path_with_relationships.append((final_data.get('name', target), None))

            return path_with_relationships

        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {source_name} and {target_name}")
            return []
