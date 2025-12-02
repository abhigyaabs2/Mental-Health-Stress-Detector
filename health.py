import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Mental Health Stress Detector",
    page_icon="🧠",
    layout="wide"
)


st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .low-stress {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .medium-stress {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .high-stress {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        with open('stress_model.pkl', 'rb') as f:
            survey_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('text_model.pkl', 'rb') as f:
            text_model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return survey_model, scaler, text_model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run the Jupyter notebook first to train the models.")
        return None, None, None, None

survey_model, scaler, text_model, vectorizer = load_models()


st.title("🧠 Mental Health Stress Level Detector")
st.markdown("""
This application uses machine learning to predict stress levels based on:
- **Survey responses** about your lifestyle and mental state
- **Text input** describing how you're feeling

**Note:** This is for educational purposes only and not a substitute for professional mental health advice.
""")


st.sidebar.title(" About")
st.sidebar.info("""
This ML project uses **Logistic Regression** to predict stress levels:
- **Low Stress** (0): Well-balanced, manageable
- **Medium Stress** (1): Some concerns, manageable with effort
- **High Stress** (2): Significant stress, may need support

**Developed with:**
- Scikit-learn
- Streamlit
- Plotly
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Survey Model Accuracy", "~85%")
st.sidebar.metric("Text Model Accuracy", "~80%")


if survey_model and scaler and text_model and vectorizer:
    
    tab1, tab2, tab3 = st.tabs(["📋 Survey Assessment", "💭 Text Analysis", "📊 Dataset Info"])
    
    
    with tab1:
        st.header("Complete the Mental Health Survey")
        st.markdown("Answer the following questions about your current state:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sleep_hours = st.slider("💤 Hours of sleep per night", 3.0, 10.0, 7.0, 0.5)
            work_hours = st.slider("💼 Work hours per day", 4.0, 14.0, 8.0, 0.5)
            social_support = st.slider("👥 Social support level (1-10)", 1, 10, 5)
            physical_activity = st.slider("🏃 Physical activity (hours/week)", 0, 7, 3)
        
        with col2:
            anxiety_level = st.slider("😰 Anxiety level (1-10)", 1, 10, 5)
            mood_swings = st.slider("🎭 Mood swings frequency (1-10)", 1, 10, 5)
            age = st.number_input("🎂 Age", 18, 100, 30)
        
        if st.button(" Analyze Survey Results", key="survey_btn"):
           
            input_data = np.array([[sleep_hours, work_hours, social_support, 
                                   physical_activity, anxiety_level, mood_swings, age]])
            input_scaled = scaler.transform(input_data)
            
            
            prediction = survey_model.predict(input_scaled)[0]
            probabilities = survey_model.predict_proba(input_scaled)[0]
            
            stress_labels = ['Low', 'Medium', 'High']
            stress_colors = ['green', 'orange', 'red']
            stress_classes = ['low-stress', 'medium-stress', 'high-stress']
            
           
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            
            st.markdown(f"""
                <div class="result-box {stress_classes[prediction]}">
                    <h2>Your Stress Level: {stress_labels[prediction].upper()}</h2>
                    <h3>Confidence: {probabilities[prediction]:.1%}</h3>
                </div>
            """, unsafe_allow_html=True)
            
           
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=stress_labels,
                        y=probabilities * 100,
                        marker_color=stress_colors,
                        text=[f'{p:.1%}' for p in probabilities],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Stress Level Probabilities",
                    xaxis_title="Stress Level",
                    yaxis_title="Probability (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Your Scores")
                st.metric("Sleep Quality", f"{sleep_hours}h", 
                         delta="Good" if sleep_hours >= 7 else "Low")
                st.metric("Work Balance", f"{work_hours}h",
                         delta="Balanced" if work_hours <= 8 else "High")
                st.metric("Anxiety Level", f"{anxiety_level}/10",
                         delta="Low" if anxiety_level <= 5 else "High")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Personalized Recommendations")
            
            recommendations = []
            if sleep_hours < 7:
                recommendations.append("😴 **Sleep:** Aim for 7-9 hours of sleep. Maintain a consistent sleep schedule.")
            if work_hours > 9:
                recommendations.append("⏰ **Work-Life Balance:** Consider reducing work hours or taking regular breaks.")
            if social_support < 5:
                recommendations.append("👥 **Social Connection:** Reach out to friends or family. Social support is crucial.")
            if physical_activity < 3:
                recommendations.append("🏃 **Exercise:** Aim for at least 150 minutes of moderate activity per week.")
            if anxiety_level > 6:
                recommendations.append("🧘 **Stress Management:** Try meditation, deep breathing, or yoga.")
            if mood_swings > 6:
                recommendations.append("😌 **Emotional Regulation:** Consider journaling or talking to a counselor.")
            
            if prediction == 2:
                recommendations.append("⚠️ **Important:** Your stress level appears high. Consider consulting a mental health professional.")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.success("You're doing great! Keep maintaining your healthy habits.")
    
    # Tab 2: Text Analysis
    with tab2:
        st.header("Describe How You're Feeling")
        st.markdown("Write a few sentences about your current mental state, work situation, or stress levels.")
        
        text_input = st.text_area(
            "Your thoughts:",
            placeholder="e.g., I've been feeling overwhelmed with work lately. Sleep has been difficult and I'm constantly worried about deadlines...",
            height=150
        )
        
        if st.button(" Analyze Text", key="text_btn"):
            if text_input.strip():
                # Vectorize and predict
                text_vec = vectorizer.transform([text_input])
                prediction = text_model.predict(text_vec)[0]
                probabilities = text_model.predict_proba(text_vec)[0]
                
                stress_labels = ['Low', 'Medium', 'High']
                stress_colors = ['green', 'orange', 'red']
                stress_classes = ['low-stress', 'medium-stress', 'high-stress']
                
                # Display results
                st.markdown("---")
                st.subheader("Text Analysis Results")
                
                st.markdown(f"""
                    <div class="result-box {stress_classes[prediction]}">
                        <h2>Detected Stress Level: {stress_labels[prediction].upper()}</h2>
                        <h3>Confidence: {probabilities[prediction]:.1%}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                fig = px.pie(
                    values=probabilities,
                    names=stress_labels,
                    title="Stress Level Distribution",
                    color_discrete_sequence=stress_colors
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Key phrases detected
                st.markdown("###Analysis Insights")
                words = text_input.lower().split()
                stress_keywords = {
                    'high': ['overwhelmed', 'anxious', 'stressed', 'worried', 'exhausted', 'can\'t sleep', 'panic'],
                    'low': ['calm', 'relaxed', 'peaceful', 'balanced', 'good', 'well', 'happy']
                }
                
                detected_high = [w for w in stress_keywords['high'] if w in text_input.lower()]
                detected_low = [w for w in stress_keywords['low'] if w in text_input.lower()]
                
                if detected_high:
                    st.warning(f" Stress indicators detected: {', '.join(detected_high)}")
                if detected_low:
                    st.success(f" Positive indicators: {', '.join(detected_low)}")
                
                if prediction == 2:
                    st.error("Your text suggests high stress levels. Please consider reaching out to a mental health professional.")
            else:
                st.warning("Please enter some text to analyze.")
    
    # Tab 3: Dataset Info
    with tab3:
        st.header(" Dataset Information")
        
        try:
            df = pd.read_csv('mental_health_data.csv')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", len(df))
            col2.metric("Features", len(df.columns) - 1)
            col3.metric("Classes", df['stress_level'].nunique())
            
            st.markdown("### Stress Level Distribution")
            stress_dist = df['stress_level'].value_counts().sort_index()
            fig = px.bar(
                x=['Low', 'Medium', 'High'],
                y=stress_dist.values,
                color=['Low', 'Medium', 'High'],
                color_discrete_sequence=['green', 'orange', 'red'],
                title="Number of Samples per Stress Level"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### Feature Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Dataset file not found. Please run the Jupyter notebook first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Disclaimer: This tool is for educational purposes only. 
        If you're experiencing mental health concerns, please consult a qualified healthcare professional.</p>
    </div>
""", unsafe_allow_html=True)