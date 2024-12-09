import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# Page config
st.set_page_config(
    page_title="Health Risk Assessment App",
    page_icon="👨‍⚕️",
    layout="wide"
)

# Model training function
@st.cache_resource
def train_models(X, y, model_type='classification'):
    """Cache the trained models using st.cache_resource as they can't be stored in a database"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                trained_models[name] = model
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                continue
        
        return trained_models, scaler
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None

# Cache data loading
@st.cache_data
def load_datasets():
    try:
        # Load only breast cancer dataset
        breast_cancer = pd.read_csv("CSV/breast-cancer.csv")
        
        # Breast Cancer preprocessing
        breast_cancer['diagnosis'] = (breast_cancer['diagnosis'] == 'M').astype(int)
        
        return breast_cancer
    except FileNotFoundError as e:
        st.error(f"""
            Error loading dataset. Please ensure file exists:
            - CSV/breast-cancer.csv
            
            Detailed error: {str(e)}
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        print(f"Error details: {str(e)}")
        st.stop()

# Title and description
st.title("👨‍⚕️ Comprehensive Health Risk Assessment")
st.markdown("""
This application helps assess health risks using machine learning models. 
Upload your medical parameters and get instant risk assessments for:
- 🎗️ Breast Cancer

You can also explore detailed model performance analysis for:
- 💝 Heart Disease
- 🫁 Lung Cancer
- 🩺 Diabetes
""")

# Load only breast cancer dataset
breast_cancer = load_datasets()

# Sidebar navigation
st.sidebar.header("Navigation")
condition = st.sidebar.selectbox(
    "Select Health Condition", 
    ["Data - Visualization Overview", "Breast Cancer"]
)

# Overview page
if condition == "Data - Visualization Overview":
    st.header("Health Risk Statistics")
    
    condition2 = st.selectbox(
        "Select Disease Type",
        ["Breast Cancer", "Heart Disease", "Lung Cancer", "Diabetes"]
    )
    
    condition3 = st.selectbox(
        "Select Classification Model", 
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    st.subheader("Analysis")
    
    # Create two columns for side-by-side images
    col_left, col_right = st.columns(2)
    
    if condition2 == "Breast Cancer":
        if condition3 == "Logistic Regression":
            with col_left:
                st.image("breast_cancer_image/BreastCancer - LogisticRegression.jpeg", 
                        caption="Logistic Regression Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("breast_cancer_image/BreastCancer - LR_Output.jpeg",
                        caption="Logistic Regression Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "Random Forest":
            with col_left:
                st.image("breast_cancer_image/BreastCancer - RandomForest.jpeg",
                        caption="Random Forest Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("breast_cancer_image/BreastCancer - RandomForest_OUTPUT.jpeg",
                        caption="Random Forest Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "XGBoost":
            with col_left:
                st.image("breast_cancer_image/BreastCancer - XGBoost.jpeg",
                        caption="XGBoost Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("breast_cancer_image/BreastCancer - XGBoost_OUTPUT.jpeg",
                        caption="XGBoost Detailed Output",
                        use_container_width=True,
                        width=400)
    
    elif condition2 == "Heart Disease":
        if condition3 == "Logistic Regression":
            with col_left:
                st.image("heart disease/Logistic Regression.jpeg",
                        caption="Logistic Regression Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("heart disease/LogReg.png",
                        caption="Logistic Regression Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "Random Forest":
            with col_left:
                st.image("heart disease/Random Forest.jpeg",
                        caption="Random Forest Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("heart disease/RandomForest.png",
                        caption="Random Forest Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "XGBoost":
            with col_left:
                st.image("heart disease/XGBoost.jpeg",
                        caption="XGBoost Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("heart disease/XGBoost.png",
                        caption="XGBoost Detailed Output",
                        use_container_width=True,
                        width=400)
    
    elif condition2 == "Lung Cancer":
        if condition3 == "Logistic Regression":
            with col_left:
                st.image("lung_cancer/Lung Cancer - Logistic Regression.jpeg",
                        caption="Logistic Regression Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("lung_cancer/Lung Cancer - LogisticRegression.jpeg",
                        caption="Logistic Regression Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "Random Forest":
            with col_left:
                st.image("lung_cancer/Lung Cancer - Random Forest.jpeg",
                        caption="Random Forest Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("lung_cancer/Lung Cancer - Random Forest (1).jpeg",
                        caption="Random Forest Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "XGBoost":
            with col_left:
                st.image("lung_cancer/Lung Cancer - XGBoost.jpeg",
                        caption="XGBoost Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("lung_cancer/Lung Cancer - XGBoost_OUTPUT.jpeg",
                        caption="XGBoost Detailed Output",
                        use_container_width=True,
                        width=400)
    
    elif condition2 == "Diabetes":
        if condition3 == "Logistic Regression":
            with col_left:
                st.image("diabetes/Logistic Regression_Confusion Matrix.jpeg",
                        caption="Logistic Regression Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("diabetes/Logistic Regression - OUTPUT.jpeg",
                        caption="Logistic Regression Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "Random Forest":
            with col_left:
                st.image("diabetes/Random Forest.jpeg",
                        caption="Random Forest Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("diabetes/RandomForest.jpeg",
                        caption="Random Forest Detailed Output",
                        use_container_width=True,
                        width=400)
        elif condition3 == "XGBoost":
            with col_left:
                st.image("diabetes/XGBoost.jpeg",
                        caption="XGBoost Results",
                        use_container_width=True,
                        width=400)
            with col_right:
                st.image("diabetes/XGBoost (1).jpeg",
                        caption="XGBoost Detailed Output",
                        use_container_width=True,
                        width=400)

elif condition == "Breast Cancer":
    st.header("🎗️ Breast Cancer Risk Assessment")
    
    # Model Performance Metrics
    st.subheader("Model Performance - Confusion Matrices")
    
    # Display pre-generated confusion matrix images with reduced size
    matrix_cols = st.columns(3)
    
    with matrix_cols[0]:
        st.write("**Logistic Regression**")
        st.image("breast_cancer_image/BreastCancer - LogisticRegression.jpeg", 
                caption="Logistic Regression Results",
                use_container_width=True,
                width=250)  # Reduced width
        
    with matrix_cols[1]:
        st.write("**Random Forest**")
        st.image("breast_cancer_image/BreastCancer - RandomForest.jpeg",
                caption="Random Forest Results",
                use_container_width=True,
                width=250)  # Reduced width
        
    with matrix_cols[2]:
        st.write("**XGBoost**")
        st.image("breast_cancer_image/BreastCancer - XGBoost.jpeg",
                caption="XGBoost Results",
                use_container_width=True,
                width=250)  # Reduced width
    
    # Input form for breast cancer parameters
    st.subheader("Risk Assessment Input")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius = st.number_input("Mean Radius", min_value=0.0, max_value=40.0, value=15.0)
        texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, value=20.0)
        perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, value=90.0)
        
    with col2:
        area = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, value=600.0)
        smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.2, value=0.1)
        compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=0.3, value=0.1)
        
    with col3:
        concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=0.4, value=0.1)
        symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=0.3, value=0.2)
        fractal_dim = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.1, value=0.06)

    if st.button("Predict Breast Cancer Risk"):
        try:
            # Prepare input data
            input_data = np.array([[
                radius, texture, perimeter, area, smoothness,
                compactness, concavity, symmetry, fractal_dim
            ]])
            
            # Get features and target from breast cancer dataset
            X = breast_cancer[['radius_mean', 'texture_mean', 'perimeter_mean', 
                             'area_mean', 'smoothness_mean', 'compactness_mean',
                             'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean']]
            y = breast_cancer['diagnosis']
            
            # Train models and get predictions
            models, scaler = train_models(X, y)
            if models is None or scaler is None:
                st.error("Error in model training. Please try again.")
                st.stop()
                
            input_scaled = scaler.transform(input_data)
            
            st.subheader("Prediction Results")
            result_cols = st.columns(3)
            
            # Display results for each model
            for idx, (name, model) in enumerate(models.items()):
                prediction = model.predict_proba(input_scaled)[0]
                probability_malignant = prediction[1]
                
                with result_cols[idx]:
                    st.write(f"**{name}**")
                    
                    # Show only the final prediction
                    if probability_malignant >= 0.8:
                        st.write("**Prediction: Malignant**")
                    else:
                        st.write("**Prediction: Benign**")
                    
                    st.write("---")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.warning("""
                Troubleshooting steps:
                1. Verify all input values are within expected ranges
                2. Check if the model features match the input parameters
                3. Ensure numerical inputs are valid
            """)