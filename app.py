import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# ATHLETE AI
# Intelligent Athlete Readiness & Performance Dashboard
# ============================================================


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AthleteAI",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    package = joblib.load(
        "athlete_ai_model.pkl"
    )

    return package


model_package = load_model()

model = model_package["model"]

features = model_package["features"]

model_accuracy = model_package["accuracy"]

feature_importance = model_package["feature_importance"]


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(
        "athlete_data.csv"
    )


data = load_data()


# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827,
            #1f2937
        );
        border: 1px solid #374151;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #cbd5e1;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #151b26;
        border: 1px solid #2d3748;
        text-align: center;
    }

    .recommendation {
        padding: 18px;
        border-radius: 15px;
        background-color: #151b26;
        border: 1px solid #2d3748;
        margin-bottom: 10px;
    }

    .small-text {
        color: #94a3b8;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.markdown(
        "# 🏆 AthleteAI"
    )

    st.caption(
        "AI-Powered Athlete Readiness & Performance Assistant"
    )

    st.divider()

    st.markdown(
        "### 🧭 Navigation"
    )

    page = st.radio(
        "Choose a section",
        [
            "🏠 Dashboard",
            "📊 Athlete Analysis",
            "💪 Training Load",
            "💧 Hydration",
            "😴 Recovery & Sleep",
            "🤖 AI Insights",
            "📈 Model Information"
        ]
    )

    st.divider()

    st.markdown(
        "### About"
    )

    st.caption(
        "AthleteAI uses machine learning "
        "to estimate athlete readiness "
        "from training and recovery inputs."
    )

    st.caption(
        "College Project Prototype"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🏆 ATHLETE AI
        </div>

        <div class="hero-subtitle">
        Intelligent Athlete Readiness & Performance Dashboard
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "## Welcome to AthleteAI 👋"
    )

    st.write(
        """
        AthleteAI combines training, recovery, sleep,
        hydration and performance information to estimate
        an athlete's current readiness level.
        """
    )

    st.divider()

    # Dataset statistics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Athletes in Dataset",
            f"{len(data):,}"
        )

    with col2:

        st.metric(
            "Sports Covered",
            data["Sport"].nunique()
        )

    with col3:

        st.metric(
            "ML Model",
            "Random Forest"
        )

    with col4:

        st.metric(
            "Model Accuracy",
            f"{model_accuracy * 100:.1f}%"
        )

    st.divider()

    st.markdown(
        "## 📊 Dataset Overview"
    )

    col1, col2 = st.columns(2)

    with col1:

        sport_counts = (
            data["Sport"]
            .value_counts()
            .reset_index()
        )

        sport_counts.columns = [
            "Sport",
            "Athletes"
        ]

        fig = px.bar(
            sport_counts,
            x="Sport",
            y="Athletes",
            title="Athletes by Sport"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Sport",
            yaxis_title="Number of Athletes"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        readiness_counts = (
            data["Readiness_Category"]
            .value_counts()
            .reset_index()
        )

        readiness_counts.columns = [
            "Readiness",
            "Athletes"
        ]

        fig = px.pie(
            readiness_counts,
            names="Readiness",
            values="Athletes",
            title="Readiness Distribution"
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "Use the Athlete Analysis section to enter an athlete's current information."
    )


# ============================================================
# ATHLETE ANALYSIS
# ============================================================

elif page == "📊 Athlete Analysis":

    st.markdown(
        "## 📊 Athlete Readiness Analysis"
    )

    st.write(
        """
        Enter the athlete's current training and recovery
        information below. AthleteAI will estimate readiness
        using the trained Random Forest model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # ATHLETE PROFILE
    # --------------------------------------------------------

    st.markdown(
        "### 👤 Athlete Profile"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        athlete_name = st.text_input(
            "Athlete Name",
            "Demo Athlete"
        )

    with col2:

        sport = st.selectbox(
            "Sport",
            [
                "Basketball",
                "Football",
                "Cricket",
                "Athletics",
                "Swimming",
                "Tennis",
                "Volleyball",
                "Cycling"
            ]
        )

    with col3:

        age = st.slider(
            "Age",
            min_value=16,
            max_value=40,
            value=21
        )

    st.divider()

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### 🏋️ Training Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        training_sessions = st.slider(
            "Training Sessions / Week",
            1,
            10,
            5
        )

    with col2:

        training_duration = st.slider(
            "Average Session Duration (minutes)",
            20,
            240,
            90
        )

    with col3:

        training_intensity = st.slider(
            "Training Intensity",
            1,
            10,
            6
        )

    st.divider()

    # --------------------------------------------------------
    # RECOVERY INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### 😴 Recovery Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        sleep_hours = st.slider(
            "Average Sleep (hours)",
            4.0,
            10.0,
            7.5,
            0.5
        )

    with col2:

        rest_days = st.slider(
            "Rest Days / Week",
            0,
            4,
            2
        )

    with col3:

        recovery_score = st.slider(
            "Recovery Score",
            0,
            100,
            75
        )

    st.divider()

    # --------------------------------------------------------
    # HYDRATION & PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        "### 💧 Hydration & Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        water_intake = st.slider(
            "Daily Water Intake (litres)",
            1.0,
            6.0,
            2.5,
            0.1
        )

    with col2:

        previous_performance = st.slider(
            "Previous Performance Score",
            0,
            100,
            75
        )

    with col3:

        training_load = (
            training_sessions
            * training_duration
            * training_intensity
            / 10
        )

        training_load = round(
            training_load,
            1
        )

        st.metric(
            "Calculated Training Load",
            training_load
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict = st.button(
        "🚀 ANALYZE ATHLETE",
        use_container_width=True
    )

    if predict:

        input_data = pd.DataFrame({

            "Age": [age],

            "Training_Sessions": [
                training_sessions
            ],

            "Training_Duration": [
                training_duration
            ],

            "Training_Intensity": [
                training_intensity
            ],

            "Sleep_Hours": [
                sleep_hours
            ],

            "Rest_Days": [
                rest_days
            ],

            "Water_Intake": [
                water_intake
            ],

            "Previous_Performance": [
                previous_performance
            ],

            "Recovery_Score": [
                recovery_score
            ],

            "Training_Load": [
                training_load
            ]

        })

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        class_names = model.classes_

        probability_dict = dict(
            zip(
                class_names,
                probabilities
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success(
            f"Analysis completed for {athlete_name}"
        )

        st.markdown(
            "## 🎯 AI Readiness Result"
        )

        if prediction == "High":

            score = 85

        elif prediction == "Moderate":

            score = 65

        else:

            score = 40

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Readiness Level",
                prediction
            )

        with col2:

            st.metric(
                "Estimated Readiness",
                f"{score}/100"
            )

        with col3:

            st.metric(
                "Training Load",
                training_load
            )

        # ----------------------------------------------------
        # GAUGE
        # ----------------------------------------------------

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={
                    "text": "Athlete Readiness"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 AI Prediction Confidence"
        )

        probability_df = pd.DataFrame({

            "Readiness": list(
                probability_dict.keys()
            ),

            "Probability": list(
                probability_dict.values()
            )

        })

        probability_df["Probability"] *= 100

        fig = px.bar(
            probability_df,
            x="Readiness",
            y="Probability",
            title="Model Prediction Probabilities"
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Probability (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # PERSONALIZED INSIGHTS
        # ----------------------------------------------------

        st.markdown(
            "## 💡 Athlete Insights"
        )

        recommendations = []

        if sleep_hours < 7:

            recommendations.append(
                "😴 Sleep is relatively low. Prioritize adequate recovery sleep."
            )

        elif sleep_hours >= 8:

            recommendations.append(
                "😴 Sleep duration is strong and supports recovery."
            )

        else:

            recommendations.append(
                "😴 Sleep duration is moderate. Consistency is important."
            )

        if recovery_score < 60:

            recommendations.append(
                "🧘 Recovery score is low. Consider additional recovery time."
            )

        elif recovery_score >= 80:

            recommendations.append(
                "🧘 Recovery indicators look strong."
            )

        else:

            recommendations.append(
                "🧘 Recovery is moderate. Monitor fatigue."
            )

        if training_load > 600:

            recommendations.append(
                "🏋️ Training load is high. Monitor fatigue and recovery."
            )

        elif training_load > 400:

            recommendations.append(
                "🏋️ Training load is moderate-to-high."
            )

        else:

            recommendations.append(
                "🏋️ Training load is relatively manageable."
            )

        if water_intake < 2:

            recommendations.append(
                "💧 Hydration intake appears low. Maintain regular fluid intake."
            )

        else:

            recommendations.append(
                "💧 Hydration level appears reasonable for this prototype."
            )

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">
                {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "These are general educational recommendations and not medical advice."
        )


# ============================================================
# TRAINING LOAD
# ============================================================

elif page == "💪 Training Load":

    st.markdown(
        "## 💪 Training Load Analysis"
    )

    st.write(
        """
        Training load provides an estimate of the overall
        training demand based on session frequency,
        duration and intensity.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        sessions = st.slider(
            "Sessions per Week",
            1,
            10,
            5
        )

    with col2:

        duration = st.slider(
            "Duration per Session",
            20,
            240,
            90
        )

    with col3:

        intensity = st.slider(
            "Intensity",
            1,
            10,
            6
        )

    load = round(
        sessions * duration * intensity / 10,
        1
    )

    st.divider()

    st.metric(
        "Training Load",
        load
    )

    if load < 300:

        status = "Low Training Load"

    elif load < 600:

        status = "Moderate Training Load"

    else:

        status = "High Training Load"

    st.info(
        status
    )

    # --------------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------------

    components = pd.DataFrame({

        "Component": [
            "Sessions",
            "Duration",
            "Intensity"
        ],

        "Value": [
            sessions,
            duration,
            intensity
        ]

    })

    fig = px.bar(
        components,
        x="Component",
        y="Value",
        title="Training Load Components"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### 📌 Training Load Formula"
    )

    st.code(
        """
Training Load =
Sessions × Duration × Intensity ÷ 10
        """
    )


# ============================================================
# HYDRATION
# ============================================================

elif page == "💧 Hydration":

    st.markdown(
        "## 💧 Hydration Dashboard"
    )

    st.write(
        """
        Hydration is an important part of maintaining
        performance and recovery. AthleteAI provides
        a simple educational hydration tracker.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        body_weight = st.number_input(
            "Body Weight (kg)",
            min_value=30.0,
            max_value=150.0,
            value=60.0,
            step=0.5
        )

    with col2:

        current_water = st.number_input(
            "Current Daily Water Intake (L)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        )

    # General educational estimate

    estimated_water = round(
        body_weight * 0.035,
        2
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Estimated Baseline",
            f"{estimated_water} L"
        )

    with col2:

        st.metric(
            "Current Intake",
            f"{current_water} L"
        )

    with col3:

        difference = round(
            current_water - estimated_water,
            2
        )

        st.metric(
            "Difference",
            f"{difference} L"
        )

    if current_water < estimated_water:

        st.warning(
            "Current intake is below the general baseline estimate."
        )

    else:

        st.success(
            "Current intake is at or above the general baseline estimate."
        )

    st.caption(
        "Hydration needs vary with climate, exercise duration, "
        "sweat rate and individual factors. This is only an "
        "educational estimate."
    )


# ============================================================
# RECOVERY & SLEEP
# ============================================================

elif page == "😴 Recovery & Sleep":

    st.markdown(
        "## 😴 Recovery & Sleep Analysis"
    )

    st.write(
        """
        Recovery is essential for adapting to training.
        Use the dashboard to review sleep, rest and recovery indicators.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        sleep = st.slider(
            "Sleep Hours",
            4.0,
            10.0,
            7.5,
            0.5
        )

    with col2:

        rest = st.slider(
            "Rest Days",
            0,
            5,
            2
        )

    with col3:

        recovery = st.slider(
            "Recovery Score",
            0,
            100,
            75
        )

    # Sleep score

    sleep_score = 100 - abs(
        sleep - 8
    ) * 18

    sleep_score = np.clip(
        sleep_score,
        0,
        100
    )

    # Recovery score

    recovery_score_display = recovery

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Sleep Score",
            f"{sleep_score:.0f}/100"
        )

    with col2:

        st.metric(
            "Recovery Score",
            f"{recovery_score_display}/100"
        )

    # Radar chart

    categories = [
        "Sleep",
        "Recovery",
        "Rest"
    ]

    values = [
        sleep_score,
        recovery,
        min(rest * 20, 100)
    ]

    categories += [categories[0]]

    values += [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Recovery Profile"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        template="plotly_dark",
        title="Recovery Profile"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if sleep < 6:

        st.warning(
            "Sleep is currently low. Recovery should be prioritized."
        )

    elif sleep < 8:

        st.info(
            "Sleep duration is moderate."
        )

    else:

        st.success(
            "Sleep duration is strong."
        )


# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "🤖 AI Insights":

    st.markdown(
        "## 🤖 AI Insights"
    )

    st.write(
        """
        This section shows which features the machine-learning
        model considers most useful when making readiness predictions.
        """
    )

    st.divider()

    importance_display = feature_importance.copy()

    importance_display["Feature"] = (
        importance_display["Feature"]
        .str.replace("_", " ")
    )

    importance_display["Importance"] *= 100

    fig = px.bar(
        importance_display,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Importance (%)",
        yaxis_title="Feature"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### 🧠 How the AI Works"
    )

    st.write(
        """
        AthleteAI uses a Random Forest classification model.
        The model learns patterns from training examples and
        predicts whether an athlete is likely to have Low,
        Moderate or High readiness.
        """
    )

    st.markdown(
        """
        **Input Features**

        • Training frequency  
        • Training duration  
        • Training intensity  
        • Sleep  
        • Rest days  
        • Hydration  
        • Previous performance  
        • Recovery score  
        • Training load  

        **Output**

        → Low Readiness  
        → Moderate Readiness  
        → High Readiness
        """
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📈 Model Information":

    st.markdown(
        "## 📈 Machine Learning Model"
    )

    st.write(
        """
        Technical information about the machine-learning
        system powering AthleteAI.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

    with col2:

        st.metric(
            "Training Samples",
            f"{int(len(data) * 0.8):,}"
        )

    with col3:

        st.metric(
            "Test Samples",
            f"{int(len(data) * 0.2):,}"
        )

    st.divider()

    st.markdown(
        "### 🎯 Model Accuracy"
    )

    st.progress(
        float(model_accuracy)
    )

    st.write(
        f"Model accuracy: **{model_accuracy * 100:.2f}%**"
    )

    st.divider()

    st.markdown(
        "### 🔢 Input Features"
    )

    feature_table = pd.DataFrame({

        "Feature": features

    })

    feature_table["Feature"] = (
        feature_table["Feature"]
        .str.replace("_", " ")
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "### 📚 Project Architecture"
    )

    st.code(
        """
Athlete Input
      ↓
Data Processing
      ↓
Feature Engineering
      ↓
Random Forest Model
      ↓
Readiness Prediction
      ↓
Dashboard Visualization
      ↓
Athlete Insights
        """
    )

    st.info(
        "AthleteAI is a college project prototype designed "
        "to demonstrate practical applications of machine learning."
    )