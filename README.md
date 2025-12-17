# SleepScope 🌙

**Explainable AI for Insomnia and Mental Health**

SleepScope is an explainable machine learning framework designed to analyze insomnia severity, classify insomnia subtypes, and explore the correlation between insomnia and depression. The application provides an interactive dashboard for both users and clinicians to assess sleep disorders using validated assessment tools and polysomnography (PSG) data.

## 🌟 Features

### Core Functionality

- **Insomnia Severity Assessment**: Calculate and interpret Insomnia Severity Index (ISI) scores (0-28 scale)
- **Depression Screening**: Evaluate depression levels using the PHQ-9 questionnaire (0-27 scale)
- **Subtype Classification**: ML-powered insomnia subtype prediction based on sleep patterns and lifestyle factors
- **PSG Analysis**: Process polysomnography and hypnogram data for clinical insights
- **Correlation Analysis**: Real-time visualization of ISI-PHQ9 correlation using Firestore data
- **Explainable AI**:  SHAP (SHapley Additive exPlanations) visualizations for model interpretability

### User Interfaces

1. **Overview**:  Introduction to the framework and its capabilities
2. **User Dashboard**: Interactive ISI and PHQ-9 questionnaires with automated scoring
3. **Clinician PSG Upload**: Upload and analyze polysomnography files (. edf format)
4. **Correlation Explorer**: View population-level insights on insomnia-depression relationships

## 🏗️ Architecture

### Backend (FastAPI)

The backend provides RESTful API endpoints for: 

- `/isi` - ISI questionnaire processing and scoring
- `/phq9` - PHQ-9 questionnaire processing and scoring
- `/subtype` - Insomnia subtype classification
- `/psg` - PSG file analysis
- `/correlation` - Global ISI-PHQ9 correlation computation

### Frontend (Streamlit)

Interactive web application with: 
- Multi-tab interface for different user roles
- Real-time form validation and scoring
- Data visualization (scatter plots, SHAP plots)
- Session data persistence via Firestore

### Data Storage

- **Google Cloud Firestore**: Stores user assessments, scores, and session data
- **Model Persistence**: Serialized ML models stored locally using joblib

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Cloud Platform account (for Firestore)
- Required Python packages (see Installation)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/manojnac/SleepScope.git
cd SleepScope/SS_backend-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `streamlit` - Web application framework
- `fastapi` - Backend API framework
- `shap` - Explainable AI library
- `scikit-learn` - Machine learning models
- `google-cloud-firestore` - Database integration
- `mne` - PSG/EEG data processing
- `pandas`, `numpy` - Data manipulation
- `matplotlib` - Visualization

3. Set up Google Cloud Firestore:
   - Create a Firestore database
   - Download service account credentials JSON
   - Configure credentials in the application

### Running the Application

**Streamlit Dashboard:**
```bash
streamlit run streamlit_app.py
```

**FastAPI Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📊 Model Details

### Insomnia Subtype Classification

The ML model uses the following features for subtype prediction: 
- Total Sleep Time (TST)
- Sleep Efficiency (SE)
- Sleep stage ratios (N1, N2, N3, REM)
- Wake After Sleep Onset (WASO) rate
- Sleep Onset Latency (SOL) rate
- Sleep Fragmentation Index
- Age, Gender, BMI

### Feature Engineering

PSG data is processed to extract:
- Sleep stage durations and ratios
- Sleep efficiency metrics
- Fragmentation indices
- Normalized lifestyle factors

### Explainability

SHAP waterfall plots provide:
- Local feature contribution for individual predictions
- Transparent decision-making process
- Clinical interpretability

## 📝 Assessment Tools

### ISI (Insomnia Severity Index)
7 questions rated 0-4, measuring:
- Difficulty falling/staying asleep
- Early morning awakening
- Sleep satisfaction
- Impact on daily functioning

**Severity Categories:**
- 0-7: No clinically significant insomnia
- 8-14: Subthreshold insomnia
- 15-21: Moderate insomnia
- 22-28: Severe insomnia

### PHQ-9 (Patient Health Questionnaire)
9 questions rated 0-3, assessing depression symptoms over the past 2 weeks. 

**Severity Categories:**
- 0-4:  Minimal depression
- 5-9: Mild depression
- 10-14: Moderate depression
- 15-19: Moderately severe depression
- 20-27: Severe depression

## 🗂️ Project Structure

```
SleepScope/
└── SS_backend-main/
    ├── streamlit_app.py          # Main Streamlit application
    ├── Procfile                  # Deployment configuration
    ├── app/
    │   ├── main.py              # FastAPI application entry point
    │   ├── routers/             # API route handlers
    │   │   ├── isi. py
    │   │   ├── phq9.py
    │   │   ├── subtype.py
    │   │   ├── psg.py
    │   │   └── correlation. py
    │   ├── services/            # Business logic
    │   │   ├── isi_service.py
    │   │   ├── phq9_service.py
    │   │   └── subtype_service.py
    │   ├── models/              # Request/Response models
    │   └── utils/               # Utility functions
    │       ├── preprocess.py    # PSG feature extraction
    │       ├── scoring.py       # Assessment scoring
    │       └── firestore. py     # Database operations
    └── models/                  # Trained ML models
```

## 🔒 Privacy & Security

- User responses are stored securely in Google Cloud Firestore
- Session IDs are UUID-generated for anonymization
- CORS middleware configured for secure API access
- No personally identifiable information (PII) is required for assessments

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is available for research and educational purposes.

## 👨‍💻 Author

**manojnac**

## 🙏 Acknowledgments

- Insomnia Severity Index (ISI) - Morin, C.  M.  (1993)
- PHQ-9 Depression Scale - Kroenke, K., et al. (2001)
- SHAP library for explainable AI
- MNE-Python for PSG processing

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub. 

---

**Note**: This application is intended for research and educational purposes. It should not replace professional medical diagnosis or treatment.  Always consult healthcare professionals for medical advice.
