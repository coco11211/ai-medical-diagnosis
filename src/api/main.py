"""FastAPI REST API for medical diagnosis system."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from loguru import logger

# Import system modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.imaging.dicom_processor import DICOMProcessor
from src.imaging.image_preprocessor import ImagePreprocessor
from src.nlp.symptom_analyzer import SymptomAnalyzer
from src.patient_history.patient_manager import PatientManager
from src.patient_history.history_tracker import HistoryTracker
from src.utils.config import load_config
from src.utils.logger import setup_logger

# Initialize FastAPI app
app = FastAPI(
    title="AI Medical Diagnosis System",
    description="Advanced AI-powered medical diagnosis system with multi-modal analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and processors
config = None
dicom_processor = None
image_preprocessor = None
symptom_analyzer = None
patient_manager = None
history_tracker = None
disease_names = None


# Pydantic models for API requests/responses
class SymptomRequest(BaseModel):
    """Request model for symptom analysis."""
    symptoms: str = Field(..., description="Description of symptoms")
    patient_id: Optional[str] = Field(None, description="Patient identifier")


class DiagnosisResponse(BaseModel):
    """Response model for diagnosis."""
    predictions: List[Dict[str, any]] = Field(..., description="List of predictions")
    primary_diagnosis: str = Field(..., description="Primary diagnosis")
    confidence: float = Field(..., description="Confidence score")
    needs_review: bool = Field(..., description="Whether diagnosis needs review")


class PatientCreateRequest(BaseModel):
    """Request model for creating patient."""
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Patient name")
    age: Optional[int] = Field(None, description="Patient age")
    sex: Optional[str] = Field(None, description="Patient sex")
    contact: Optional[str] = Field(None, description="Contact information")


class VisitRequest(BaseModel):
    """Request model for patient visit."""
    patient_id: str = Field(..., description="Patient identifier")
    chief_complaint: str = Field(..., description="Main complaint")
    symptoms: Optional[List[str]] = Field(None, description="List of symptoms")
    vital_signs: Optional[Dict[str, float]] = Field(None, description="Vital signs")
    notes: Optional[str] = Field(None, description="Additional notes")


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global config, dicom_processor, image_preprocessor, symptom_analyzer
    global patient_manager, history_tracker, disease_names

    # Setup logger
    setup_logger()
    logger.info("Starting AI Medical Diagnosis System API")

    # Load configuration
    try:
        config = load_config()
    except:
        from src.utils.config import AppConfig
        config = AppConfig()

    # Initialize processors
    dicom_processor = DICOMProcessor(
        window_center=config.dicom.window_center,
        window_width=config.dicom.window_width,
        target_size=tuple(config.dicom.target_size)
    )

    image_preprocessor = ImagePreprocessor(
        target_size=tuple(config.model.input_size),
        normalize=True,
        augment=False
    )

    # Initialize NLP analyzer
    symptom_analyzer = SymptomAnalyzer(
        model_name=config.nlp.model_name,
        max_length=config.nlp.max_length,
        device=config.device
    )

    # Initialize patient management
    patient_manager = PatientManager(
        database_url=f"{config.database.type}:///{config.database.database}"
    )
    history_tracker = HistoryTracker(patient_manager)

    # Load disease names
    disease_names = [
        "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer",
        "Pulmonary Edema", "Pleural Effusion", "Cardiomegaly",
        "Nodule", "Mass", "Atelectasis"
    ]

    logger.info("API initialization complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if patient_manager:
        patient_manager.close()
    logger.info("API shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Medical Diagnosis System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "dicom_processor": dicom_processor is not None,
            "image_preprocessor": image_preprocessor is not None,
            "symptom_analyzer": symptom_analyzer is not None,
            "patient_manager": patient_manager is not None
        }
    }


@app.post("/analyze/symptoms", response_model=Dict[str, any])
async def analyze_symptoms(request: SymptomRequest):
    """
    Analyze symptoms using NLP.

    Args:
        request: Symptom analysis request

    Returns:
        Analysis results
    """
    try:
        logger.info(f"Analyzing symptoms: {request.symptoms[:50]}...")

        # Analyze symptoms
        analysis = symptom_analyzer.analyze_symptoms(request.symptoms)

        # Remove numpy arrays from response (not JSON serializable)
        response = {
            'text': analysis['text'],
            'symptoms': analysis['symptoms'],
            'body_parts': analysis['body_parts'],
            'num_symptoms': analysis['num_symptoms'],
            'num_body_parts': analysis['num_body_parts'],
            'embedding_dim': analysis['embedding_dim']
        }

        # Track visit if patient ID provided
        if request.patient_id:
            history_tracker.track_visit(
                patient_id=request.patient_id,
                chief_complaint=request.symptoms[:200],
                symptoms=analysis['symptoms']
            )

        return response

    except Exception as e:
        logger.error(f"Error analyzing symptoms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None
):
    """
    Analyze medical image (DICOM or standard image).

    Args:
        file: Uploaded image file
        patient_id: Optional patient identifier

    Returns:
        Image analysis results
    """
    try:
        logger.info(f"Analyzing image: {file.filename}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Check if DICOM
            if file.filename.lower().endswith('.dcm'):
                # Process DICOM
                processed_image, dicom_data = dicom_processor.process_dicom_file(tmp_path)
                metadata = dicom_processor.extract_metadata(dicom_data)

                response = {
                    'type': 'dicom',
                    'shape': list(processed_image.shape),
                    'metadata': metadata,
                    'processing_status': 'success'
                }
            else:
                # Process standard image
                import cv2
                image = cv2.imread(tmp_path)
                if image is None:
                    raise ValueError("Failed to read image")

                # Preprocess
                preprocessed = image_preprocessor.preprocess(image)

                response = {
                    'type': 'standard',
                    'shape': list(preprocessed.shape),
                    'processing_status': 'success'
                }

            return response

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/patients", response_model=Dict[str, any])
async def create_patient(request: PatientCreateRequest):
    """
    Create a new patient record.

    Args:
        request: Patient creation request

    Returns:
        Created patient information
    """
    try:
        patient = patient_manager.create_patient(
            patient_id=request.patient_id,
            name=request.name,
            age=request.age,
            sex=request.sex,
            contact=request.contact
        )

        return {
            'patient_id': patient.patient_id,
            'name': patient.name,
            'age': patient.age,
            'sex': patient.sex,
            'created_at': patient.created_at.isoformat()
        }

    except Exception as e:
        logger.error(f"Error creating patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    """
    Get patient information.

    Args:
        patient_id: Patient identifier

    Returns:
        Patient information
    """
    try:
        summary = patient_manager.get_patient_summary(patient_id)

        if not summary:
            raise HTTPException(status_code=404, detail="Patient not found")

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visits")
async def create_visit(request: VisitRequest):
    """
    Create a new patient visit record.

    Args:
        request: Visit request

    Returns:
        Visit confirmation
    """
    try:
        success = history_tracker.track_visit(
            patient_id=request.patient_id,
            chief_complaint=request.chief_complaint,
            symptoms=request.symptoms,
            vital_signs=request.vital_signs,
            notes=request.notes
        )

        if not success:
            raise HTTPException(status_code=404, detail="Patient not found")

        return {
            'status': 'success',
            'message': 'Visit recorded successfully'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating visit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    days: Optional[int] = None
):
    """
    Get patient medical history.

    Args:
        patient_id: Patient identifier
        days: Optional time window in days

    Returns:
        Patient history
    """
    try:
        diagnosis_history = history_tracker.get_diagnosis_history(
            patient_id, time_window_days=days
        )
        visit_history = history_tracker.get_visit_history(
            patient_id, time_window_days=days
        )

        return {
            'patient_id': patient_id,
            'diagnoses': diagnosis_history,
            'visits': visit_history,
            'time_window_days': days
        }

    except Exception as e:
        logger.error(f"Error getting patient history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}/risk-factors")
async def get_risk_factors(patient_id: str):
    """
    Get patient risk factors.

    Args:
        patient_id: Patient identifier

    Returns:
        Risk factor analysis
    """
    try:
        risk_factors = history_tracker.get_risk_factors(patient_id)

        if not risk_factors:
            raise HTTPException(status_code=404, detail="Patient not found")

        return risk_factors

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting risk factors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diseases")
async def list_diseases():
    """
    List all supported diseases.

    Returns:
        List of diseases
    """
    return {
        'diseases': disease_names,
        'count': len(disease_names)
    }


def start_server():
    """Start the API server (for CLI entry point)."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    start_server()
