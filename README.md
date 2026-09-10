# Multiple Disease Prediction System

A machine learning web application that predicts the likelihood of **Parkinson's Disease**, **Chronic Kidney Disease**, and **Liver Disease** based on user-input medical parameters.

## Features
- Multi-disease prediction in one unified app
- Interactive, styled Streamlit interface
- Real-time predictions using trained ML models

## Tech Stack
- **Frontend/App**: Streamlit
- **Backend**: Python
- **ML Algorithms**: Random Forest Classifier
- **Libraries**: Scikit-learn, Pandas, NumPy, Joblib

## Datasets Used
1. Parkinson's Disease dataset
2. Chronic Kidney Disease dataset
3. Indian Liver Patient dataset

## Model Accuracy
| Disease | Accuracy |
|---|---|
| Parkinson's | 94.87% |
| Kidney Disease | 100% |
| Liver Disease | 72.65% |

## How to Run Locally
```bash
git clone https://github.com/Kali-py59/multiple-disease-prediction.git
cd multiple-disease-prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
multiple-disease-prediction/
├── datasets/ # Raw CSV datasets
├── models/ # Training scripts
├── saved_models/ # Trained model + scaler files
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md
## Author
Kalieswari
