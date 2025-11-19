"""Patient history tracking and analysis."""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter
import json
from loguru import logger


class HistoryTracker:
    """Track and analyze patient medical history."""

    def __init__(self, patient_manager):
        """
        Initialize history tracker.

        Args:
            patient_manager: PatientManager instance
        """
        self.patient_manager = patient_manager
        logger.info("Initialized HistoryTracker")

    def track_diagnosis(
        self,
        patient_id: str,
        disease_name: str,
        confidence: float,
        symptoms: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = 'suspected'
    ) -> bool:
        """
        Track a new diagnosis.

        Args:
            patient_id: Patient identifier
            disease_name: Name of disease
            confidence: Confidence score
            symptoms: List of symptoms
            image_path: Path to diagnostic image
            notes: Additional notes
            status: Diagnosis status

        Returns:
            True if successful, False otherwise
        """
        diagnosis = self.patient_manager.add_diagnosis(
            patient_id=patient_id,
            disease_name=disease_name,
            confidence=confidence,
            symptoms=symptoms,
            image_path=image_path,
            notes=notes,
            status=status
        )

        return diagnosis is not None

    def track_visit(
        self,
        patient_id: str,
        chief_complaint: str,
        symptoms: Optional[List[str]] = None,
        vital_signs: Optional[Dict[str, float]] = None,
        diagnosis: Optional[str] = None,
        treatment: Optional[str] = None,
        notes: Optional[str] = None,
        follow_up_days: Optional[int] = None
    ) -> bool:
        """
        Track a patient visit.

        Args:
            patient_id: Patient identifier
            chief_complaint: Main complaint
            symptoms: List of symptoms
            vital_signs: Dictionary of vital signs
            diagnosis: Diagnosis given
            treatment: Treatment prescribed
            notes: Additional notes
            follow_up_days: Days until follow-up

        Returns:
            True if successful, False otherwise
        """
        follow_up_date = None
        if follow_up_days:
            follow_up_date = datetime.utcnow() + timedelta(days=follow_up_days)

        visit = self.patient_manager.add_visit(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            vital_signs=vital_signs,
            diagnosis=diagnosis,
            treatment=treatment,
            notes=notes,
            follow_up_date=follow_up_date
        )

        return visit is not None

    def get_diagnosis_history(
        self,
        patient_id: str,
        time_window_days: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Get patient's diagnosis history.

        Args:
            patient_id: Patient identifier
            time_window_days: Only get diagnoses within this many days

        Returns:
            List of diagnoses
        """
        diagnoses = self.patient_manager.get_patient_diagnoses(patient_id)

        # Filter by time window if specified
        if time_window_days:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            diagnoses = [d for d in diagnoses if d.diagnosis_date >= cutoff_date]

        # Format results
        history = []
        for d in diagnoses:
            history.append({
                'disease': d.disease_name,
                'confidence': d.confidence,
                'date': d.diagnosis_date.isoformat() if d.diagnosis_date else None,
                'symptoms': d.symptoms,
                'status': d.status,
                'notes': d.notes
            })

        return history

    def get_visit_history(
        self,
        patient_id: str,
        time_window_days: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Get patient's visit history.

        Args:
            patient_id: Patient identifier
            time_window_days: Only get visits within this many days

        Returns:
            List of visits
        """
        visits = self.patient_manager.get_patient_visits(patient_id)

        # Filter by time window if specified
        if time_window_days:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            visits = [v for v in visits if v.visit_date >= cutoff_date]

        # Format results
        history = []
        for v in visits:
            history.append({
                'date': v.visit_date.isoformat() if v.visit_date else None,
                'complaint': v.chief_complaint,
                'symptoms': v.symptoms,
                'vital_signs': v.vital_signs,
                'diagnosis': v.diagnosis,
                'treatment': v.treatment,
                'notes': v.notes,
                'follow_up': v.follow_up_date.isoformat() if v.follow_up_date else None
            })

        return history

    def analyze_disease_patterns(
        self,
        patient_id: str
    ) -> Dict[str, any]:
        """
        Analyze disease patterns in patient history.

        Args:
            patient_id: Patient identifier

        Returns:
            Dictionary with pattern analysis
        """
        diagnoses = self.patient_manager.get_patient_diagnoses(patient_id)

        if not diagnoses:
            return {
                'total_diagnoses': 0,
                'unique_diseases': 0,
                'most_common_diseases': [],
                'recurring_diseases': []
            }

        # Extract disease names
        disease_names = [d.disease_name for d in diagnoses]

        # Count occurrences
        disease_counts = Counter(disease_names)

        # Find recurring diseases (diagnosed more than once)
        recurring = {
            disease: count
            for disease, count in disease_counts.items()
            if count > 1
        }

        # Get most common diseases
        most_common = disease_counts.most_common(5)

        # Calculate average confidence
        avg_confidence = sum(d.confidence for d in diagnoses if d.confidence) / len(diagnoses)

        analysis = {
            'total_diagnoses': len(diagnoses),
            'unique_diseases': len(disease_counts),
            'most_common_diseases': [
                {'disease': disease, 'count': count}
                for disease, count in most_common
            ],
            'recurring_diseases': [
                {'disease': disease, 'count': count}
                for disease, count in recurring.items()
            ],
            'average_confidence': avg_confidence,
            'first_diagnosis_date': diagnoses[-1].diagnosis_date.isoformat() if diagnoses else None,
            'latest_diagnosis_date': diagnoses[0].diagnosis_date.isoformat() if diagnoses else None
        }

        return analysis

    def analyze_symptom_patterns(
        self,
        patient_id: str
    ) -> Dict[str, any]:
        """
        Analyze symptom patterns in patient history.

        Args:
            patient_id: Patient identifier

        Returns:
            Dictionary with symptom analysis
        """
        # Get all visits and diagnoses
        visits = self.patient_manager.get_patient_visits(patient_id)
        diagnoses = self.patient_manager.get_patient_diagnoses(patient_id)

        # Collect all symptoms
        all_symptoms = []

        for visit in visits:
            if visit.symptoms:
                all_symptoms.extend(visit.symptoms)

        for diagnosis in diagnoses:
            if diagnosis.symptoms:
                all_symptoms.extend(diagnosis.symptoms)

        if not all_symptoms:
            return {
                'total_symptom_reports': 0,
                'unique_symptoms': 0,
                'most_common_symptoms': []
            }

        # Count symptom occurrences
        symptom_counts = Counter(all_symptoms)

        # Get most common symptoms
        most_common = symptom_counts.most_common(10)

        analysis = {
            'total_symptom_reports': len(all_symptoms),
            'unique_symptoms': len(symptom_counts),
            'most_common_symptoms': [
                {'symptom': symptom, 'count': count}
                for symptom, count in most_common
            ]
        }

        return analysis

    def get_risk_factors(
        self,
        patient_id: str
    ) -> Dict[str, any]:
        """
        Identify risk factors based on patient history.

        Args:
            patient_id: Patient identifier

        Returns:
            Dictionary with risk factors
        """
        patient = self.patient_manager.get_patient(patient_id)

        if not patient:
            return {}

        # Analyze diagnosis history
        disease_analysis = self.analyze_disease_patterns(patient_id)

        # Analyze symptoms
        symptom_analysis = self.analyze_symptom_patterns(patient_id)

        # Identify risk factors
        risk_factors = {
            'age_related': patient.age >= 65 if patient.age else False,
            'chronic_conditions': len(disease_analysis['recurring_diseases']) > 0,
            'frequent_symptoms': len(symptom_analysis['most_common_symptoms']) > 5,
            'multiple_diagnoses': disease_analysis['total_diagnoses'] > 3,
            'recent_activity': False  # Will be set below
        }

        # Check for recent activity
        recent_diagnoses = self.get_diagnosis_history(patient_id, time_window_days=30)
        risk_factors['recent_activity'] = len(recent_diagnoses) > 0

        # Calculate overall risk score (0-1)
        risk_score = sum(risk_factors.values()) / len(risk_factors)

        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'risk_level': 'High' if risk_score > 0.6 else 'Medium' if risk_score > 0.3 else 'Low'
        }

    def generate_timeline(
        self,
        patient_id: str,
        time_window_days: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Generate timeline of patient's medical events.

        Args:
            patient_id: Patient identifier
            time_window_days: Time window in days

        Returns:
            List of events in chronological order
        """
        timeline = []

        # Get diagnoses
        diagnoses = self.get_diagnosis_history(patient_id, time_window_days)
        for d in diagnoses:
            timeline.append({
                'type': 'diagnosis',
                'date': d['date'],
                'description': f"Diagnosed with {d['disease']} (confidence: {d['confidence']:.2f})",
                'details': d
            })

        # Get visits
        visits = self.get_visit_history(patient_id, time_window_days)
        for v in visits:
            timeline.append({
                'type': 'visit',
                'date': v['date'],
                'description': f"Visit: {v['complaint']}",
                'details': v
            })

        # Sort by date (most recent first)
        timeline.sort(key=lambda x: x['date'] if x['date'] else '', reverse=True)

        return timeline

    def export_patient_history(
        self,
        patient_id: str,
        output_path: str
    ) -> bool:
        """
        Export patient history to JSON file.

        Args:
            patient_id: Patient identifier
            output_path: Path to save export

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get comprehensive patient summary
            summary = self.patient_manager.get_patient_summary(patient_id)

            if not summary:
                logger.warning(f"Patient not found: {patient_id}")
                return False

            # Add additional analysis
            summary['disease_patterns'] = self.analyze_disease_patterns(patient_id)
            summary['symptom_patterns'] = self.analyze_symptom_patterns(patient_id)
            summary['risk_factors'] = self.get_risk_factors(patient_id)
            summary['timeline'] = self.generate_timeline(patient_id)

            # Save to file
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Exported patient history to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting patient history: {e}")
            return False
