import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# ATHLETE AI - MACHINE LEARNING MODEL
# ============================================================

print("=" * 60)
print("ATHLETE AI - MODEL TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. CREATE SYNTHETIC ATHLETE DATA
# ------------------------------------------------------------

np.random.seed(42)

NUMBER_OF_ATHLETES = 2500

sports = [
    "Basketball",
    "Football",
    "Cricket",
    "Athletics",
    "Swimming",
    "Tennis",
    "Volleyball",
    "Cycling"
]

sport = np.random.choice(
    sports,
    NUMBER_OF_ATHLETES
)

age = np.random.randint(
    16,
    36,
    NUMBER_OF_ATHLETES
)

training_sessions = np.random.randint(
    1,
    8,
    NUMBER_OF_ATHLETES
)

training_duration = np.random.randint(
    30,
    181,
    NUMBER_OF_ATHLETES
)

training_intensity = np.random.randint(
    1,
    11,
    NUMBER_OF_ATHLETES
)

sleep_hours = np.round(
    np.random.normal(
        7.0,
        1.5,
        NUMBER_OF_ATHLETES
    ),
    1
)

sleep_hours = np.clip(
    sleep_hours,
    4,
    10
)

rest_days = np.random.randint(
    0,
    4,
    NUMBER_OF_ATHLETES
)

water_intake = np.round(
    np.random.normal(
        2.4,
        0.7,
        NUMBER_OF_ATHLETES
    ),
    1
)

water_intake = np.clip(
    water_intake,
    1,
    5
)

previous_performance = np.round(
    np.random.uniform(
        50,
        100,
        NUMBER_OF_ATHLETES
    ),
    1
)

recovery_score = np.round(
    np.random.normal(
        72,
        15,
        NUMBER_OF_ATHLETES
    ),
    1
)

recovery_score = np.clip(
    recovery_score,
    25,
    100
)

# ------------------------------------------------------------
# 2. CALCULATE TRAINING LOAD
# ------------------------------------------------------------

training_load = (
    training_sessions
    * training_duration
    * training_intensity
)

training_load = np.round(
    training_load / 10,
    1
)


# ------------------------------------------------------------
# 3. CALCULATE READINESS COMPONENTS
# ------------------------------------------------------------

sleep_component = (
    100
    - abs(sleep_hours - 8) * 18
)

sleep_component = np.clip(
    sleep_component,
    20,
    100
)


training_balance = (
    100
    - np.maximum(
        training_load - 450,
        0
    ) * 0.08
)

training_balance = np.clip(
    training_balance,
    30,
    100
)


hydration_component = (
    100
    - abs(water_intake - 2.8) * 22
)

hydration_component = np.clip(
    hydration_component,
    30,
    100
)


rest_component = (
    65
    + rest_days * 10
)

rest_component = np.clip(
    rest_component,
    40,
    100
)


# ------------------------------------------------------------
# 4. CALCULATE OVERALL READINESS SCORE
# ------------------------------------------------------------

readiness_score = (

    sleep_component * 0.25

    + recovery_score * 0.25

    + hydration_component * 0.10

    + training_balance * 0.20

    + rest_component * 0.10

    + previous_performance * 0.10

)


noise = np.random.normal(
    0,
    3,
    NUMBER_OF_ATHLETES
)

readiness_score += noise


readiness_score = np.clip(
    readiness_score,
    0,
    100
)


readiness_score = np.round(
    readiness_score,
    1
)


# ------------------------------------------------------------
# 5. CONVERT SCORE INTO CATEGORY
# ------------------------------------------------------------

def categorize_readiness(score):

    if score < 65:

        return "Low"

    elif score < 82:

        return "Moderate"

    else:

        return "High"

readiness_category = [

    categorize_readiness(score)

    for score in readiness_score

]


# ------------------------------------------------------------
# 6. CREATE DATAFRAME
# ------------------------------------------------------------

data = pd.DataFrame({

    "Sport": sport,

    "Age": age,

    "Training_Sessions": training_sessions,

    "Training_Duration": training_duration,

    "Training_Intensity": training_intensity,

    "Sleep_Hours": sleep_hours,

    "Rest_Days": rest_days,

    "Water_Intake": water_intake,

    "Previous_Performance": previous_performance,

    "Recovery_Score": recovery_score,

    "Training_Load": training_load,

    "Readiness_Score": readiness_score,

    "Readiness_Category": readiness_category

})


# ------------------------------------------------------------
# 7. SAVE DATASET
# ------------------------------------------------------------

data.to_csv(
    "athlete_data.csv",
    index=False
)


print()
print("Dataset created successfully.")

print()
print("Dataset shape:")
print(data.shape)

print()
print("First five rows:")
print(data.head())

print()
print("Readiness distribution:")
print(
    data["Readiness_Category"].value_counts()
)


# ------------------------------------------------------------
# 8. SELECT FEATURES
# ------------------------------------------------------------

features = [

    "Age",

    "Training_Sessions",

    "Training_Duration",

    "Training_Intensity",

    "Sleep_Hours",

    "Rest_Days",

    "Water_Intake",

    "Previous_Performance",

    "Recovery_Score",

    "Training_Load"

]


X = data[features]

y = data["Readiness_Category"]


# ------------------------------------------------------------
# 9. SPLIT DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ------------------------------------------------------------
# 10. CREATE RANDOM FOREST MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    random_state=42,

    class_weight="balanced"

)


# ------------------------------------------------------------
# 11. TRAIN MODEL
# ------------------------------------------------------------

print()
print("Training Random Forest model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ------------------------------------------------------------
# 12. MAKE PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(
    X_test
)


# ------------------------------------------------------------
# 13. EVALUATE MODEL
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print()
print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ------------------------------------------------------------
# 14. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


importance = importance.sort_values(

    "Importance",

    ascending=False

)


print()
print("Feature Importance:")

print(importance)


# ------------------------------------------------------------
# 15. SAVE MODEL
# ------------------------------------------------------------

model_package = {

    "model": model,

    "features": features,

    "accuracy": accuracy,

    "feature_importance": importance

}


joblib.dump(

    model_package,

    "athlete_ai_model.pkl"

)


# ------------------------------------------------------------
# 16. FINAL MESSAGE
# ------------------------------------------------------------

print()
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print()
print("Created files:")

print("✓ athlete_data.csv")

print("✓ athlete_ai_model.pkl")

print()
print("ATHLETE AI model is ready!")