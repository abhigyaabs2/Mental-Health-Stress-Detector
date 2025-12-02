# 🧠 Mental Health Stress Level Detector

A machine learning project that predicts stress levels from survey data or text input using Logistic Regression.

## 📋 Project Overview

This project includes:
- **Data Generation**: Synthetic mental health survey dataset
- **ML Model**: Logistic Regression for multi-class classification
- **Dual Input Methods**: Survey-based and text-based prediction
- **Interactive Dashboard**: Streamlit web application
- **Visualizations**: Feature importance, confusion matrices, and predictions

## 🎯 Features

- **Survey Assessment**: Predict stress levels based on 7 lifestyle factors
- **Text Analysis**: Analyze written descriptions using NLP
- **Real-time Predictions**: Get instant stress level classifications
- **Personalized Recommendations**: Receive tailored mental health advice
- **Interactive Visualizations**: View probability distributions and insights

## 📊 Stress Level Classes

- **0 - Low Stress**: Well-balanced, manageable life
- **1 - Medium Stress**: Some concerns, manageable with effort
- **2 - High Stress**: Significant stress, may need professional support

## 🛠️ Installation

### 1. Clone or Download Project Files

Create a project folder and save all the provided files:
```
stress-detector/
├── stress_detector.ipynb    # Jupyter notebook for training
├── app.py                     # Streamlit application
└── README.md                  # This file
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## 🚀 Usage

### Step 1: Train the Models

Open and run the Jupyter notebook:

```bash
jupyter notebook stress_detector.ipynb
```

Run all cells to:
- Generate synthetic dataset (mental_health_data.csv)
- Train survey-based model (stress_model.pkl, scaler.pkl)
- Train text-based model (text_model.pkl, vectorizer.pkl)
- Generate visualizations and evaluation metrics

**Expected Output Files:**
- `mental_health_data.csv` - Training dataset
- `stress_model.pkl` - Survey prediction model
- `scaler.pkl` - Feature scaler
- `text_model.pkl` - Text prediction model
- `vectorizer.pkl` - TF-IDF vectorizer
- Various PNG visualization files

### Step 2: Run Streamlit App

After training, launch the web application:

```bash
streamlit run stress.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Using the Application

### Survey Assessment Tab
1. Adjust sliders for your lifestyle factors:
   - Sleep hours (3-10 hours)
   - Work hours (4-14 hours)
   - Social support level (1-10 scale)
   - Physical activity (0-7 hours/week)
   - Anxiety level (1-10 scale)
   - Mood swings frequency (1-10 scale)
   - Age
2. Click "Analyze Survey Results"
3. View your predicted stress level and personalized recommendations

### Text Analysis Tab
1. Write about your current mental state or feelings
2. Click "Analyze Text"
3. See the detected stress level based on your text

### Dataset Info Tab
- View dataset statistics
- Explore stress level distribution
- Review sample data and feature summaries

## 🧪 Model Performance

- **Survey Model Accuracy**: ~85%
- **Text Model Accuracy**: ~80%
- **Algorithm**: Multinomial Logistic Regression
- **Features**: 7 survey features + text vectorization (TF-IDF)

## 📈 Technical Details

### Survey Model Features
1. Sleep hours
2. Work hours
3. Social support level
4. Physical activity
5. Anxiety level
6. Mood swings frequency
7. Age

### Text Model
- **Vectorization**: TF-IDF (100 features)
- **Preprocessing**: Lowercase, stop word removal
- **Input**: Free-form text descriptions

### Machine Learning Pipeline
```
Data Collection → Feature Engineering → Standardization → 
Logistic Regression → Evaluation → Deployment
```

## 🔧 Customization

### Modify Dataset Size
In the notebook, change `n_samples`:
```python
df = generate_stress_dataset(n_samples=2000)  # Default is 1000
```

### Adjust Model Parameters
```python
lr_model = LogisticRegression(
    max_iter=1000,
    C=1.0,  # Regularization strength
    random_state=42
)
```

### Change Stress Categories
Modify the binning logic:
```python
df['stress_level'] = pd.cut(stress_score, bins=3, labels=[0, 1, 2])
# bins=4 for 4 categories, bins=2 for 2 categories
```

## 📊 Visualizations Generated

- Correlation matrix of features
- Stress level distribution
- Confusion matrix
- Feature importance chart
- Probability distributions (in Streamlit)

## ⚠️ Important Notes

- This is an **educational project** for learning ML concepts
- **Not a substitute** for professional mental health diagnosis
- The synthetic dataset is for demonstration purposes
- Model predictions should not be used for medical decisions

## 🆘 Mental Health Resources

If you're experiencing mental health concerns:
- **National Suicide Prevention Lifeline (US)**: 988
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

## 🤝 Contributing

Ideas for improvement:
- Add more features (diet, work type, relationship status)
- Implement deep learning models (LSTM, BERT)
- Create mobile application
- Add user authentication and history tracking
- Integrate real mental health datasets (with proper permissions)

## 📝 License

This project is for educational purposes. Feel free to modify and use for learning.

## 📧 Support

For questions or issues:
1. Check that all model files were generated from the notebook
2. Verify all dependencies are installed
3. Ensure you're using Python 3.8 or higher

## 🎓 Learning Outcomes

By completing this project, you'll learn:
- Binary and multi-class classification
- Feature engineering and scaling
- Text processing with TF-IDF
- Model evaluation metrics
- Streamlit web app development
- ML model deployment and serialization

---

**Happy Learning! 🚀**
