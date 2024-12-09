# Health Risk Assessment Application 🏥

A comprehensive machine learning-powered web application for medical risk assessment, built with Streamlit and scikit-learn. This application helps healthcare professionals and researchers evaluate potential health risks using advanced machine learning models.

## Features 🌟

- **Multi-Disease Risk Assessment**
  - 🎗️ Breast Cancer Prediction
  - 💝 Heart Disease Analysis
  - 🫁 Lung Cancer Evaluation
  - 🩺 Diabetes Risk Assessment

- **Advanced ML Model Integration**
  - Logistic Regression (Base Model)
  - Random Forest Classifier (Ensemble Learning)
  - XGBoost (Gradient Boosting)

- **Interactive Features**
  - Real-time Predictions
  - Interactive Data Visualization
  - Model Performance Comparisons
  - Confidence Scores
  - Feature Importance Analysis

## Quick Start 🚀

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/health-risk-assessment.git
   cd health-risk-assessment
   ```

2. Set up virtual environment:
   ```bash
   python -m venv .venv
   # For Windows
   .venv\Scripts\activate
   # For Unix/MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   streamlit run frontend.py
   ```

## Project Structure 📁

```
health-risk-assessment/
├── frontend.py              # Main application file
├── requirements.txt         # Python dependencies
├── CSV/                     # Data directory
│ └── breast-cancer.csv
├── breast_cancer_image/    # Model performance visualizations
├── heart_disease/
├── lung_cancer/
└── diabetes/

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

[Add your license information here]

## Contact 📧

[Add your contact information here]