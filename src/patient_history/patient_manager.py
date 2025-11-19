"""Patient management and database operations."""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import List, Dict, Optional
import json
from pathlib import Path
from loguru import logger

Base = declarative_base()


class Patient(Base):
    """Patient model."""

    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(200))
    age = Column(Integer)
    sex = Column(String(10))
    date_of_birth = Column(DateTime)
    contact = Column(String(100))
    address = Column(Text)
    medical_history = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    visits = relationship("Visit", back_populates="patient", cascade="all, delete-orphan")


class Diagnosis(Base):
    """Diagnosis model."""

    __tablename__ = 'diagnoses'

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    disease_name = Column(String(200), nullable=False)
    disease_id = Column(String(50))
    confidence = Column(Float)
    diagnosis_date = Column(DateTime, default=datetime.utcnow)
    symptoms = Column(JSON)
    image_path = Column(String(500))
    notes = Column(Text)
    status = Column(String(50))  # confirmed, suspected, ruled_out
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")


class Visit(Base):
    """Patient visit model."""

    __tablename__ = 'visits'

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    visit_date = Column(DateTime, default=datetime.utcnow)
    chief_complaint = Column(Text)
    symptoms = Column(JSON)
    vital_signs = Column(JSON)
    diagnosis = Column(String(200))
    treatment = Column(Text)
    notes = Column(Text)
    follow_up_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="visits")


class PatientManager:
    """Manage patient records and database operations."""

    def __init__(self, database_url: str = "sqlite:///medical_diagnosis.db"):
        """
        Initialize patient manager.

        Args:
            database_url: Database connection URL
        """
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        logger.info(f"Initialized PatientManager with database: {database_url}")

    def create_patient(
        self,
        patient_id: str,
        name: str,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        **kwargs
    ) -> Patient:
        """
        Create a new patient record.

        Args:
            patient_id: Unique patient identifier
            name: Patient name
            age: Patient age
            sex: Patient sex
            **kwargs: Additional patient attributes

        Returns:
            Patient object
        """
        patient = Patient(
            patient_id=patient_id,
            name=name,
            age=age,
            sex=sex,
            **kwargs
        )

        self.session.add(patient)
        self.session.commit()

        logger.info(f"Created patient: {patient_id}")
        return patient

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """
        Get patient by ID.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient object or None
        """
        patient = self.session.query(Patient).filter_by(patient_id=patient_id).first()
        return patient

    def update_patient(
        self,
        patient_id: str,
        **updates
    ) -> Optional[Patient]:
        """
        Update patient information.

        Args:
            patient_id: Patient identifier
            **updates: Fields to update

        Returns:
            Updated patient object or None
        """
        patient = self.get_patient(patient_id)

        if patient:
            for key, value in updates.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)

            patient.updated_at = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated patient: {patient_id}")

        return patient

    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete patient record.

        Args:
            patient_id: Patient identifier

        Returns:
            True if deleted, False otherwise
        """
        patient = self.get_patient(patient_id)

        if patient:
            self.session.delete(patient)
            self.session.commit()
            logger.info(f"Deleted patient: {patient_id}")
            return True

        return False

    def add_diagnosis(
        self,
        patient_id: str,
        disease_name: str,
        confidence: float,
        symptoms: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[Diagnosis]:
        """
        Add diagnosis to patient.

        Args:
            patient_id: Patient identifier
            disease_name: Name of diagnosed disease
            confidence: Confidence score
            symptoms: List of symptoms
            **kwargs: Additional diagnosis attributes

        Returns:
            Diagnosis object or None
        """
        patient = self.get_patient(patient_id)

        if not patient:
            logger.warning(f"Patient not found: {patient_id}")
            return None

        diagnosis = Diagnosis(
            patient_id=patient.id,
            disease_name=disease_name,
            confidence=confidence,
            symptoms=symptoms,
            **kwargs
        )

        self.session.add(diagnosis)
        self.session.commit()

        logger.info(f"Added diagnosis for patient {patient_id}: {disease_name}")
        return diagnosis

    def get_patient_diagnoses(
        self,
        patient_id: str,
        limit: Optional[int] = None
    ) -> List[Diagnosis]:
        """
        Get patient's diagnosis history.

        Args:
            patient_id: Patient identifier
            limit: Maximum number of results

        Returns:
            List of diagnoses
        """
        patient = self.get_patient(patient_id)

        if not patient:
            return []

        query = self.session.query(Diagnosis).filter_by(patient_id=patient.id)\
                    .order_by(Diagnosis.diagnosis_date.desc())

        if limit:
            query = query.limit(limit)

        return query.all()

    def add_visit(
        self,
        patient_id: str,
        chief_complaint: str,
        symptoms: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[Visit]:
        """
        Add patient visit record.

        Args:
            patient_id: Patient identifier
            chief_complaint: Main reason for visit
            symptoms: List of symptoms
            **kwargs: Additional visit attributes

        Returns:
            Visit object or None
        """
        patient = self.get_patient(patient_id)

        if not patient:
            logger.warning(f"Patient not found: {patient_id}")
            return None

        visit = Visit(
            patient_id=patient.id,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            **kwargs
        )

        self.session.add(visit)
        self.session.commit()

        logger.info(f"Added visit for patient {patient_id}")
        return visit

    def get_patient_visits(
        self,
        patient_id: str,
        limit: Optional[int] = None
    ) -> List[Visit]:
        """
        Get patient's visit history.

        Args:
            patient_id: Patient identifier
            limit: Maximum number of results

        Returns:
            List of visits
        """
        patient = self.get_patient(patient_id)

        if not patient:
            return []

        query = self.session.query(Visit).filter_by(patient_id=patient.id)\
                    .order_by(Visit.visit_date.desc())

        if limit:
            query = query.limit(limit)

        return query.all()

    def search_patients(
        self,
        name: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        sex: Optional[str] = None
    ) -> List[Patient]:
        """
        Search patients by criteria.

        Args:
            name: Patient name (partial match)
            age_min: Minimum age
            age_max: Maximum age
            sex: Patient sex

        Returns:
            List of matching patients
        """
        query = self.session.query(Patient)

        if name:
            query = query.filter(Patient.name.ilike(f'%{name}%'))

        if age_min:
            query = query.filter(Patient.age >= age_min)

        if age_max:
            query = query.filter(Patient.age <= age_max)

        if sex:
            query = query.filter(Patient.sex == sex)

        return query.all()

    def get_patient_summary(self, patient_id: str) -> Optional[Dict[str, any]]:
        """
        Get comprehensive patient summary.

        Args:
            patient_id: Patient identifier

        Returns:
            Dictionary with patient summary
        """
        patient = self.get_patient(patient_id)

        if not patient:
            return None

        # Get recent diagnoses and visits
        recent_diagnoses = self.get_patient_diagnoses(patient_id, limit=5)
        recent_visits = self.get_patient_visits(patient_id, limit=5)

        summary = {
            'patient_id': patient.patient_id,
            'name': patient.name,
            'age': patient.age,
            'sex': patient.sex,
            'contact': patient.contact,
            'total_diagnoses': len(patient.diagnoses),
            'total_visits': len(patient.visits),
            'recent_diagnoses': [
                {
                    'disease': d.disease_name,
                    'confidence': d.confidence,
                    'date': d.diagnosis_date.isoformat() if d.diagnosis_date else None,
                    'status': d.status
                }
                for d in recent_diagnoses
            ],
            'recent_visits': [
                {
                    'date': v.visit_date.isoformat() if v.visit_date else None,
                    'complaint': v.chief_complaint,
                    'diagnosis': v.diagnosis
                }
                for v in recent_visits
            ],
            'medical_history': patient.medical_history
        }

        return summary

    def close(self):
        """Close database session."""
        self.session.close()
        logger.info("Closed database session")
