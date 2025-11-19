# API Documentation

## Overview

The AI Medical Diagnosis System provides a comprehensive REST API for medical image analysis, symptom interpretation, and patient management.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production use, implement JWT or OAuth2 authentication.

## Endpoints

### System Status

#### GET /

Get API information and status.

**Response:**
```json
{
  "message": "AI Medical Diagnosis System API",
  "version": "1.0.0",
  "status": "running"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "dicom_processor": true,
    "image_preprocessor": true,
    "symptom_analyzer": true,
    "patient_manager": true
  }
}
```

### Symptom Analysis

#### POST /analyze/symptoms

Analyze patient symptoms using NLP.

**Request Body:**
```json
{
  "symptoms": "Patient presents with fever, cough, and shortness of breath",
  "patient_id": "P001"  // optional
}
```

**Response:**
```json
{
  "text": "Patient presents with fever, cough, and shortness of breath",
  "symptoms": ["fever", "cough", "breath"],
  "body_parts": ["chest"],
  "num_symptoms": 3,
  "num_body_parts": 1,
  "embedding_dim": 768
}
```

### Image Analysis

#### POST /analyze/image

Analyze medical image (DICOM or standard format).

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (DICOM or image file)
- Query Parameters:
  - `patient_id` (optional): Patient identifier

**Response for DICOM:**
```json
{
  "type": "dicom",
  "shape": [512, 512],
  "metadata": {
    "patient_id": "12345",
    "patient_name": "DOE^JOHN",
    "modality": "CT",
    "study_date": "20250101"
  },
  "processing_status": "success"
}
```

### Patient Management

#### POST /patients

Create a new patient record.

**Request Body:**
```json
{
  "patient_id": "P001",
  "name": "John Doe",
  "age": 45,
  "sex": "M",
  "contact": "+1234567890"
}
```

**Response:**
```json
{
  "patient_id": "P001",
  "name": "John Doe",
  "age": 45,
  "sex": "M",
  "created_at": "2025-01-01T12:00:00"
}
```

#### GET /patients/{patient_id}

Get patient information and summary.

**Response:**
```json
{
  "patient_id": "P001",
  "name": "John Doe",
  "age": 45,
  "sex": "M",
  "total_diagnoses": 5,
  "total_visits": 10,
  "recent_diagnoses": [
    {
      "disease": "Pneumonia",
      "confidence": 0.92,
      "date": "2025-01-15T10:30:00",
      "status": "confirmed"
    }
  ],
  "recent_visits": [
    {
      "date": "2025-01-15T10:00:00",
      "complaint": "Chest pain",
      "diagnosis": "Pneumonia"
    }
  ]
}
```

### Patient Visits

#### POST /visits

Record a patient visit.

**Request Body:**
```json
{
  "patient_id": "P001",
  "chief_complaint": "Chest pain and difficulty breathing",
  "symptoms": ["chest pain", "dyspnea"],
  "vital_signs": {
    "temperature": 38.5,
    "heart_rate": 95,
    "blood_pressure": "130/85"
  },
  "notes": "Patient appears distressed"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Visit recorded successfully"
}
```

### Patient History

#### GET /patients/{patient_id}/history

Get patient medical history.

**Query Parameters:**
- `days` (optional): Time window in days

**Response:**
```json
{
  "patient_id": "P001",
  "diagnoses": [
    {
      "disease": "Pneumonia",
      "confidence": 0.92,
      "date": "2025-01-15T10:30:00",
      "symptoms": ["fever", "cough"],
      "status": "confirmed"
    }
  ],
  "visits": [
    {
      "date": "2025-01-15T10:00:00",
      "complaint": "Chest pain",
      "diagnosis": "Pneumonia"
    }
  ],
  "time_window_days": 30
}
```

#### GET /patients/{patient_id}/risk-factors

Get patient risk factor analysis.

**Response:**
```json
{
  "risk_factors": {
    "age_related": true,
    "chronic_conditions": true,
    "frequent_symptoms": false,
    "multiple_diagnoses": true,
    "recent_activity": true
  },
  "risk_score": 0.6,
  "risk_level": "Medium"
}
```

### Disease Information

#### GET /diseases

List all supported diseases.

**Response:**
```json
{
  "diseases": [
    "Pneumonia",
    "COVID-19",
    "Tuberculosis",
    "Lung Cancer",
    "Pulmonary Edema"
  ],
  "count": 5
}
```

## Error Responses

### 404 Not Found

```json
{
  "detail": "Patient not found"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Error message describing the issue"
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production, implement rate limiting based on your requirements.

## Best Practices

1. **Always validate inputs** before sending to API
2. **Handle errors gracefully** on the client side
3. **Use appropriate content types** (JSON for data, multipart/form-data for files)
4. **Include patient IDs** when available for better tracking
5. **Monitor API health** using the `/health` endpoint

## Examples

### Python Example

```python
import requests

# Analyze symptoms
response = requests.post(
    "http://localhost:8000/analyze/symptoms",
    json={
        "symptoms": "Patient has fever and persistent cough",
        "patient_id": "P001"
    }
)
print(response.json())

# Upload and analyze image
with open("xray.dcm", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/image",
        files={"file": f}
    )
print(response.json())
```

### cURL Example

```bash
# Create patient
curl -X POST "http://localhost:8000/patients" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "name": "John Doe",
    "age": 45,
    "sex": "M"
  }'

# Analyze symptoms
curl -X POST "http://localhost:8000/analyze/symptoms" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "Fever and cough for 3 days"
  }'
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.
