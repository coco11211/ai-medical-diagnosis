"""Medical knowledge graph builder."""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import json
from pathlib import Path
from loguru import logger


class KnowledgeGraphBuilder:
    """Build and manage medical knowledge graph."""

    def __init__(self):
        """Initialize knowledge graph builder."""
        self.graph = nx.DiGraph()
        logger.info("Initialized KnowledgeGraphBuilder")

    def add_disease(
        self,
        disease_id: str,
        name: str,
        category: str = None,
        description: str = None,
        **attributes
    ) -> None:
        """
        Add a disease node to the knowledge graph.

        Args:
            disease_id: Unique disease identifier
            name: Disease name
            category: Disease category
            description: Disease description
            **attributes: Additional attributes
        """
        self.graph.add_node(
            disease_id,
            type='disease',
            name=name,
            category=category,
            description=description,
            **attributes
        )

        logger.debug(f"Added disease: {name}")

    def add_symptom(
        self,
        symptom_id: str,
        name: str,
        description: str = None,
        **attributes
    ) -> None:
        """
        Add a symptom node to the knowledge graph.

        Args:
            symptom_id: Unique symptom identifier
            name: Symptom name
            description: Symptom description
            **attributes: Additional attributes
        """
        self.graph.add_node(
            symptom_id,
            type='symptom',
            name=name,
            description=description,
            **attributes
        )

        logger.debug(f"Added symptom: {name}")

    def add_treatment(
        self,
        treatment_id: str,
        name: str,
        treatment_type: str = None,
        description: str = None,
        **attributes
    ) -> None:
        """
        Add a treatment node to the knowledge graph.

        Args:
            treatment_id: Unique treatment identifier
            name: Treatment name
            treatment_type: Type of treatment
            description: Treatment description
            **attributes: Additional attributes
        """
        self.graph.add_node(
            treatment_id,
            type='treatment',
            name=name,
            treatment_type=treatment_type,
            description=description,
            **attributes
        )

        logger.debug(f"Added treatment: {name}")

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        **attributes
    ) -> None:
        """
        Add a relationship between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            weight: Relationship weight/strength
            **attributes: Additional attributes
        """
        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type,
            weight=weight,
            **attributes
        )

        logger.debug(f"Added relationship: {source_id} --{relationship_type}--> {target_id}")

    def add_disease_symptom_relation(
        self,
        disease_id: str,
        symptom_id: str,
        frequency: float = None,
        severity: str = None
    ) -> None:
        """
        Add relationship between disease and symptom.

        Args:
            disease_id: Disease ID
            symptom_id: Symptom ID
            frequency: How often this symptom occurs (0-1)
            severity: Severity level
        """
        self.add_relationship(
            disease_id,
            symptom_id,
            'has_symptom',
            weight=frequency if frequency else 1.0,
            severity=severity
        )

    def add_disease_treatment_relation(
        self,
        disease_id: str,
        treatment_id: str,
        effectiveness: float = None,
        first_line: bool = False
    ) -> None:
        """
        Add relationship between disease and treatment.

        Args:
            disease_id: Disease ID
            treatment_id: Treatment ID
            effectiveness: Treatment effectiveness (0-1)
            first_line: Whether this is first-line treatment
        """
        self.add_relationship(
            disease_id,
            treatment_id,
            'treated_by',
            weight=effectiveness if effectiveness else 1.0,
            first_line=first_line
        )

    def build_from_json(self, json_path: str) -> None:
        """
        Build knowledge graph from JSON file.

        Args:
            json_path: Path to JSON file

        Expected JSON format:
        {
            "diseases": [...],
            "symptoms": [...],
            "treatments": [...],
            "relationships": [...]
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Add diseases
        for disease in data.get('diseases', []):
            self.add_disease(**disease)

        # Add symptoms
        for symptom in data.get('symptoms', []):
            self.add_symptom(**symptom)

        # Add treatments
        for treatment in data.get('treatments', []):
            self.add_treatment(**treatment)

        # Add relationships
        for rel in data.get('relationships', []):
            self.add_relationship(**rel)

        logger.info(f"Built knowledge graph from {json_path}: "
                   f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def save_graph(self, output_path: str, format: str = 'gexf') -> None:
        """
        Save knowledge graph to file.

        Args:
            output_path: Output file path
            format: Output format ('gexf', 'graphml', 'json')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == 'gexf':
            nx.write_gexf(self.graph, output_path)
        elif format == 'graphml':
            nx.write_graphml(self.graph, output_path)
        elif format == 'json':
            data = nx.node_link_data(self.graph)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Graph saved to {output_path}")

    def load_graph(self, input_path: str, format: str = 'gexf') -> None:
        """
        Load knowledge graph from file.

        Args:
            input_path: Input file path
            format: Input format ('gexf', 'graphml', 'json')
        """
        if format == 'gexf':
            self.graph = nx.read_gexf(input_path)
        elif format == 'graphml':
            self.graph = nx.read_graphml(input_path)
        elif format == 'json':
            with open(input_path, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Graph loaded from {input_path}: "
                   f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get knowledge graph statistics.

        Returns:
            Dictionary of statistics
        """
        # Count nodes by type
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Count edges by type
        edge_types = {}
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'is_connected': nx.is_weakly_connected(self.graph),
            'density': nx.density(self.graph)
        }

        return stats

    def create_sample_knowledge_graph(self) -> None:
        """Create a sample medical knowledge graph for testing."""
        # Diseases
        self.add_disease('d1', 'Pneumonia', category='Respiratory',
                        description='Lung infection')
        self.add_disease('d2', 'COVID-19', category='Respiratory',
                        description='Coronavirus disease')
        self.add_disease('d3', 'Tuberculosis', category='Respiratory',
                        description='Bacterial lung infection')
        self.add_disease('d4', 'Diabetes', category='Metabolic',
                        description='Blood sugar disorder')
        self.add_disease('d5', 'Hypertension', category='Cardiovascular',
                        description='High blood pressure')

        # Symptoms
        self.add_symptom('s1', 'Fever', description='Elevated body temperature')
        self.add_symptom('s2', 'Cough', description='Persistent coughing')
        self.add_symptom('s3', 'Shortness of breath', description='Difficulty breathing')
        self.add_symptom('s4', 'Fatigue', description='Extreme tiredness')
        self.add_symptom('s5', 'Chest pain', description='Pain in chest area')
        self.add_symptom('s6', 'Frequent urination', description='Increased urination')
        self.add_symptom('s7', 'Headache', description='Head pain')

        # Treatments
        self.add_treatment('t1', 'Antibiotics', treatment_type='Medication')
        self.add_treatment('t2', 'Antiviral drugs', treatment_type='Medication')
        self.add_treatment('t3', 'Oxygen therapy', treatment_type='Supportive')
        self.add_treatment('t4', 'Insulin', treatment_type='Medication')
        self.add_treatment('t5', 'Blood pressure medication', treatment_type='Medication')

        # Disease-Symptom relationships
        self.add_disease_symptom_relation('d1', 's1', frequency=0.9, severity='high')
        self.add_disease_symptom_relation('d1', 's2', frequency=0.95, severity='high')
        self.add_disease_symptom_relation('d1', 's3', frequency=0.7, severity='medium')
        self.add_disease_symptom_relation('d1', 's5', frequency=0.6, severity='medium')

        self.add_disease_symptom_relation('d2', 's1', frequency=0.88, severity='high')
        self.add_disease_symptom_relation('d2', 's2', frequency=0.67, severity='medium')
        self.add_disease_symptom_relation('d2', 's3', frequency=0.55, severity='medium')
        self.add_disease_symptom_relation('d2', 's4', frequency=0.8, severity='high')

        self.add_disease_symptom_relation('d3', 's1', frequency=0.75, severity='medium')
        self.add_disease_symptom_relation('d3', 's2', frequency=0.9, severity='high')
        self.add_disease_symptom_relation('d3', 's4', frequency=0.85, severity='high')

        self.add_disease_symptom_relation('d4', 's6', frequency=0.9, severity='medium')
        self.add_disease_symptom_relation('d4', 's4', frequency=0.7, severity='medium')

        self.add_disease_symptom_relation('d5', 's7', frequency=0.6, severity='medium')

        # Disease-Treatment relationships
        self.add_disease_treatment_relation('d1', 't1', effectiveness=0.9, first_line=True)
        self.add_disease_treatment_relation('d1', 't3', effectiveness=0.7, first_line=False)

        self.add_disease_treatment_relation('d2', 't2', effectiveness=0.6, first_line=True)
        self.add_disease_treatment_relation('d2', 't3', effectiveness=0.8, first_line=True)

        self.add_disease_treatment_relation('d3', 't1', effectiveness=0.95, first_line=True)

        self.add_disease_treatment_relation('d4', 't4', effectiveness=0.9, first_line=True)

        self.add_disease_treatment_relation('d5', 't5', effectiveness=0.85, first_line=True)

        logger.info("Created sample knowledge graph")
