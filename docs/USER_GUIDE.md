# User Guide

## Getting Started

This guide will help you get started with the AI Medical Diagnosis System.

## Installation

### Step 1: System Requirements

Before installing, ensure your system meets these requirements:

- **Operating System**: Windows 11
- **Python**: 3.9 or higher
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 10GB free space
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ support

### Step 2: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH"
4. Verify installation:
```bash
python --version
```

### Step 3: Install the System

1. Open Command Prompt or PowerShell
2. Navigate to installation directory
3. Create virtual environment:
```bash
python -m venv venv
```

4. Activate virtual environment:
```bash
venv\Scripts\activate
```

5. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Download Required Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Basic Usage

### Starting the System

1. Activate virtual environment:
```bash
venv\Scripts\activate
```

2. Start the API server:
```bash
python -m src.api.main
```

3. Open browser and go to:
```
http://localhost:8000/docs
```

### Using the Web Interface

The Swagger UI at `/docs` provides an interactive interface:

1. Click on an endpoint to expand it
2. Click "Try it out"
3. Fill in parameters
4. Click "Execute"
5. View the response

## Common Tasks

### Task 1: Analyze Patient Symptoms

1. Navigate to `/analyze/symptoms` endpoint
2. Enter symptom description:
```json
{
  "symptoms": "Patient reports persistent headache and fever for 2 days"
}
```
3. Click Execute
4. Review detected symptoms and body parts

### Task 2: Process Medical Images

1. Navigate to `/analyze/image` endpoint
2. Click "Choose File"
3. Select DICOM or image file
4. Click Execute
5. Review processing results and metadata

### Task 3: Create Patient Record

1. Navigate to `/patients` POST endpoint
2. Enter patient information:
```json
{
  "patient_id": "P12345",
  "name": "John Smith",
  "age": 52,
  "sex": "M",
  "contact": "555-1234"
}
```
3. Click Execute

### Task 4: View Patient History

1. Navigate to `/patients/{patient_id}/history` endpoint
2. Enter patient ID (e.g., "P12345")
3. Optionally set time window (days)
4. Click Execute
5. Review diagnosis and visit history

## Advanced Features

### Multi-Modal Analysis

Combine image and text analysis:

```python
from src.fusion.multimodal_fusion import MultiModalFusion
from src.models.resnet_model import MedicalResNet
from src.nlp.symptom_analyzer import SymptomAnalyzer

# Initialize components
image_model = MedicalResNet(num_classes=10)
text_analyzer = SymptomAnalyzer()

# Extract features
image_features = image_model.get_embedding(image_tensor)
text_features = text_analyzer.encode_text(symptoms)

# Fuse and predict
fusion_model = MultiModalFusion(
    image_feature_dim=512,
    text_feature_dim=768,
    num_classes=10
)

predictions = fusion_model.predict(image_features, text_features)
```

### Explainability

Generate explanations for predictions:

```python
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer

# SHAP explanation
shap_explainer = SHAPExplainer(model, background_data)
shap_values = shap_explainer.explain(input_data)
shap_explainer.visualize_image_explanation(
    image,
    shap_values,
    output_path="shap_explanation.png"
)

# LIME explanation
lime_explainer = LIMEExplainer(model, class_names)
explanation, mask = lime_explainer.explain_image(image)
lime_explainer.visualize_image_explanation(
    image,
    explanation,
    output_path="lime_explanation.png"
)
```

### Knowledge Graph Queries

Query the medical knowledge graph:

```python
from src.knowledge_graph.graph_query import KnowledgeGraphQuery

query = KnowledgeGraphQuery(knowledge_graph)

# Find diseases by symptoms
diseases = query.get_disease_by_symptoms(
    ["fever", "cough", "fatigue"],
    threshold=0.6
)

# Get recommended treatments
treatments = query.get_treatments_by_disease("Pneumonia")

# Get differential diagnosis
differential = query.get_differential_diagnosis(
    ["chest pain", "shortness of breath"],
    top_k=5
)
```

## Configuration

### Basic Configuration

Edit `config/config.yaml`:

```yaml
# Model settings
model:
  num_classes: 10
  batch_size: 32
  learning_rate: 0.001

# Device selection
device: "cuda"  # Use "cpu" if no GPU

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

### Environment Variables

Create `.env` file:

```env
DEVICE=cuda
LOG_LEVEL=INFO
DB_TYPE=sqlite
DB_NAME=medical_diagnosis.db
API_PORT=8000
```

## Troubleshooting

### Issue: API won't start

**Solution:**
1. Check if port 8000 is available
2. Try different port in config
3. Check Python version (must be 3.9+)

### Issue: GPU not detected

**Solution:**
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Set `device: "cpu"` in config if no GPU

### Issue: Import errors

**Solution:**
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt --upgrade`
3. Check Python path

### Issue: Out of memory

**Solution:**
1. Reduce batch size in config
2. Use smaller model variant (ResNet18 instead of ResNet50)
3. Set `device: "cpu"` if GPU memory is limited

## Best Practices

1. **Always activate virtual environment** before running
2. **Keep patient data secure** - use encryption
3. **Regular backups** of patient database
4. **Monitor system resources** during operation
5. **Update dependencies** regularly for security
6. **Use GPU** for better performance when available
7. **Test on sample data** before production use

## Support

For issues and questions:
- Check documentation in `docs/` folder
- Review API docs at `/docs` endpoint
- Submit issues on GitHub
- Contact support team

## Next Steps

- Read API Documentation for detailed endpoint information
- Explore example notebooks in `examples/` directory
- Train custom models with your data
- Integrate with existing medical systems
- Set up automated processing pipelines
