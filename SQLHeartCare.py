import sqlite3
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import (train_test_split,cross_val_score,StratifiedKFold)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score,classification_report)
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,ConfusionMatrixDisplay,roc_curve)

# STEP 1: LOAD DATASET
print("Loading dataset from UCI...")
diabetes_data = fetch_ucirepo(id=296)
features = diabetes_data.data.features
target = diabetes_data.data.targets
df_raw = pd.concat([features, target], axis=1)
print("Dataset loaded successfully.")
print("Rows:", df_raw.shape[0])
print("Columns:", df_raw.shape[1])

# STEP 2: DATA CLEANING
print("\nCleaning data...")
df = df_raw.copy()
df.replace("?", np.nan, inplace=True)
df["encounter_id"] = range(1, len(df) + 1)
df["readmitted_30"] = 0
df.loc[df["readmitted"] == "<30", "readmitted_30"] = 1
print("Cleaning complete.")

# STEP 3: CREATE SQLITE DATABASE
print("\nCreating SQLite database...")
conn = sqlite3.connect(":memory:")
print("Database created.")

# TABLE 1 : ENCOUNTERS
print("\nCreating encounters table...")
encounters_columns = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "readmitted_30"]

encounters_columns_existing = []
for column in encounters_columns:
    if column in df.columns:
        encounters_columns_existing.append(column)
encounters_df = df[encounters_columns_existing]
encounters_df.to_sql("encounters",conn,index=False,if_exists="replace")
print("encounters table created.")

# TABLE 2 : DIAGNOSES
print("\nCreating diagnoses table...")
diagnosis_columns = ["encounter_id"]
if "diag_1" in df.columns:
    diagnosis_columns.append("diag_1")
if "diag_2" in df.columns:
    diagnosis_columns.append("diag_2")
if "diag_3" in df.columns:
    diagnosis_columns.append("diag_3")
diagnosis_df = df[diagnosis_columns]
diagnosis_long = diagnosis_df.melt(id_vars="encounter_id",var_name="diagnosis_position",value_name="diagnosis_code")
diagnosis_long = diagnosis_long.dropna(subset=["diagnosis_code"])
diagnosis_long.to_sql("diagnoses",conn,index=False,if_exists="replace")
print("diagnoses table created.")

# TABLE 3 : MEDICATIONS
print("\nCreating medications table...")
medication_columns = [
    "encounter_id",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "insulin",
    "change",
    "diabetesMed"]

existing_medication_columns = []
for column in medication_columns:
    if column in df.columns:
        existing_medication_columns.append(column)
medications_df = df[existing_medication_columns]
medications_df.to_sql("medications",conn,index=False,if_exists="replace")
print("medications table created.")

# TABLE 4 : LAB RESULTS
print("\nCreating lab_results table...")
lab_columns = ["encounter_id"]
if "A1Cresult" in df.columns:
    lab_columns.append("A1Cresult")
if "max_glu_serum" in df.columns:
    lab_columns.append("max_glu_serum")
lab_df = df[lab_columns]
lab_df.to_sql("lab_results",conn,index=False,if_exists="replace")
print("lab_results table created.")

# CHECK TABLES
print("\nChecking database tables...")
tables_query = """
SELECT name
FROM sqlite_master
WHERE type='table'
"""
tables = pd.read_sql_query(tables_query, conn)
print(tables)
print("\nDatabase setup complete.")

# STEP 4: SQL ANALYSIS
print("\nRunning SQL Analysis...")
# QUERY 1
# Readmission rate by age
print("QUERY 1: READMISSION RATE BY AGE")
query1 = """
SELECT
    age,
    COUNT(*) AS total_encounters,
    SUM(readmitted_30) AS readmitted_count,
    ROUND(
        100.0 * SUM(readmitted_30) / COUNT(*),
        2
    ) AS readmission_rate_pct
FROM encounters
GROUP BY age
ORDER BY readmission_rate_pct DESC
"""
q1 = pd.read_sql_query(query1, conn)
print(q1)
# QUERY 2
# Most common diagnoses
print("QUERY 2: TOP DIAGNOSES")
query2 = """
SELECT
    diagnosis_code,
    COUNT(*) AS frequency
FROM diagnoses
WHERE diagnosis_position='diag_1'
AND diagnosis_code NOT LIKE 'V%'
AND diagnosis_code NOT LIKE 'E%'
GROUP BY diagnosis_code
ORDER BY frequency DESC
LIMIT 10
"""
q2 = pd.read_sql_query(query2, conn)
print(q2)
# QUERY 3
# Insulin and hospital stay
print("QUERY 3: INSULIN ANALYSIS")
query3 = """
SELECT
    m.insulin,
    COUNT(*) AS patients,
    ROUND(
        AVG(e.time_in_hospital),
        2
    ) AS avg_days,
    ROUND(
        AVG(e.num_medications),
        2
    ) AS avg_medications,
    ROUND(
        100.0 * SUM(e.readmitted_30) /
        COUNT(*),
        2
    ) AS readmit_rate_pct
FROM encounters e
JOIN medications m
ON e.encounter_id = m.encounter_id
WHERE m.insulin IS NOT NULL
GROUP BY m.insulin
ORDER BY avg_days DESC
"""
q3 = pd.read_sql_query(query3, conn)
print(q3)
# QUERY 4
# A1C and readmission
print("QUERY 4: A1C ANALYSIS")
query4 = """
SELECT
    l.A1Cresult,
    COUNT(*) AS encounters,
    ROUND(
        AVG(e.time_in_hospital),
        2
    ) AS avg_stay_days,
    ROUND(
        100.0 * SUM(e.readmitted_30)
        / COUNT(*),
        2
    ) AS readmit_rate_pct
FROM encounters e
JOIN lab_results l
ON e.encounter_id = l.encounter_id
WHERE l.A1Cresult IS NOT NULL
GROUP BY l.A1Cresult
ORDER BY readmit_rate_pct DESC
"""
q4 = pd.read_sql_query(query4, conn)
print(q4)
# QUERY 5
# High risk patients
print("QUERY 5: HIGH RISK PATIENTS")
query5 = """
SELECT
    e.age,
    e.number_diagnoses,
    e.number_emergency,
    m.insulin,

    COUNT(*) AS patient_count,

    ROUND(
        100.0 * SUM(e.readmitted_30)
        / COUNT(*),
        2
    ) AS readmit_rate_pct

FROM encounters e

JOIN medications m
ON e.encounter_id = m.encounter_id

WHERE e.number_diagnoses >= 7
AND e.number_emergency >= 1
AND m.insulin IN ('Up','Steady','Down')

GROUP BY e.age, m.insulin

HAVING patient_count >= 20

ORDER BY readmit_rate_pct DESC

LIMIT 10
"""
q5 = pd.read_sql_query(query5, conn)
print(q5)
# STEP 5
# FEATURE ENGINEERING IN SQL
print("\nCreating ML features using SQL...")
feature_query = """
SELECT

    e.encounter_id,

    e.time_in_hospital,
    e.num_lab_procedures,
    e.num_procedures,
    e.num_medications,

    e.number_outpatient,
    e.number_emergency,
    e.number_inpatient,

    e.number_diagnoses,

    (
        e.number_outpatient +
        e.number_emergency +
        e.number_inpatient
    ) AS total_prior_visits,

    ROUND(
        CAST(e.num_procedures AS FLOAT)
        /
        (e.time_in_hospital + 1),
        4
    ) AS procedures_per_day,

    ROUND(
        CAST(e.num_medications AS FLOAT)
        /
        (e.number_diagnoses + 1),
        4
    ) AS meds_per_diagnosis,

    CASE
        WHEN e.age IN
        (
            '[70-80)',
            '[80-90)',
            '[90-100)'
        )
        THEN 1
        ELSE 0
    END AS elderly_flag,

    CASE
        WHEN m.insulin IN
        (
            'Up',
            'Down'
        )
        THEN 1
        ELSE 0
    END AS insulin_adjusted,

    CASE
        WHEN l.A1Cresult IS NOT NULL
        AND l.A1Cresult != 'None'
        THEN 1
        ELSE 0
    END AS a1c_tested,

    CASE
        WHEN l.A1Cresult IN
        (
            '>8',
            '>7'
        )
        THEN 1
        ELSE 0
    END AS a1c_abnormal,

    CASE
        WHEN e.number_emergency >= 2
        THEN 1
        ELSE 0
    END AS frequent_emergency,

    CASE
        WHEN m.diabetesMed = 'Yes'
        THEN 1
        ELSE 0
    END AS on_diabetes_med,

    e.readmitted_30

FROM encounters e

LEFT JOIN medications m
ON e.encounter_id = m.encounter_id

LEFT JOIN lab_results l
ON e.encounter_id = l.encounter_id
"""

df_ml = pd.read_sql_query(feature_query,conn)
print("\nML Dataset Created")
print("Rows:", df_ml.shape[0])
print("Columns:", df_ml.shape[1])
print("Readmission Rate:", round(df_ml["readmitted_30"].mean() * 100,2),"%")

# CHECK DATASET
print("\nFirst 5 Rows")
print(df_ml.head())
print("\nColumn Names")
for column in df_ml.columns:
    print(column)

# STEP 6: MACHINE LEARNING
print("\nStarting Machine Learning...")
# PREPARE X AND y
feature_columns = []
for column in df_ml.columns:
    if column != "encounter_id" and column != "readmitted_30":
        feature_columns.append(column)

X = df_ml[feature_columns]
y = df_ml["readmitted_30"]
print("\nFeatures used:")
for column in feature_columns:
    print(column)
print("\nTarget:")
print("readmitted_30")

# TRAIN TEST SPLIT
print("\nCreating Train/Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))
# CROSS VALIDATION SETTINGS
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# RESULTS DICTIONARY
results = {}
# MODEL 1
# LOGISTIC REGRESSION
print("LOGISTIC REGRESSION")
logistic_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
print("Running Cross Validation...")
logistic_cv_scores = cross_val_score(logistic_pipeline,X_train,y_train,cv=cv,scoring="roc_auc")
print("CV AUC:",logistic_cv_scores.mean(),"+/-",logistic_cv_scores.std())
print("Training model...")
logistic_pipeline.fit(X_train,y_train)
logistic_predictions = logistic_pipeline.predict(X_test)
logistic_probabilities = (logistic_pipeline.predict_proba(X_test)[:, 1])
logistic_auc = roc_auc_score(y_test,logistic_probabilities)
print("Test AUC:", logistic_auc)
print(classification_report(y_test,logistic_predictions,target_names=["Not Readmitted","Readmitted <30d"],zero_division=0))
results["Logistic Regression"] = {"auc": logistic_auc,"predictions": logistic_predictions,"probabilities": logistic_probabilities}

# MODEL 2
# RANDOM FOREST
print("RANDOM FOREST")
rf_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("model",RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1))])

print("Running Cross Validation...")
rf_cv_scores = cross_val_score(rf_pipeline,X_train,y_train,cv=cv,scoring="roc_auc")
print("CV AUC:",rf_cv_scores.mean(),"+/-",rf_cv_scores.std())
print("Training model...")
rf_pipeline.fit(X_train,y_train)
rf_predictions = rf_pipeline.predict(X_test)
rf_probabilities = (rf_pipeline.predict_proba(X_test)[:, 1])
rf_auc = roc_auc_score(y_test,rf_probabilities)
print("Test AUC:", rf_auc)
print(classification_report(y_test,rf_predictions,target_names=["Not Readmitted","Readmitted <30d"],zero_division=0))
results["Random Forest"] = {"auc": rf_auc,"predictions": rf_predictions,"probabilities": rf_probabilities}

# MODEL 3
# XGBOOST
print("XGBOOST")
negative_class_count = (y == 0).sum()
positive_class_count = (y == 1).sum()
scale_weight = (negative_class_count / positive_class_count)
xgb_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("model",XGBClassifier(n_estimators=200,scale_pos_weight=scale_weight,eval_metric="logloss",random_state=42,verbosity=0))])

print("Running Cross Validation...")
xgb_cv_scores = cross_val_score(xgb_pipeline,X_train,y_train,cv=cv,scoring="roc_auc")
print("CV AUC:",xgb_cv_scores.mean(),"+/-",xgb_cv_scores.std())
print("Training model...")
xgb_pipeline.fit(X_train,y_train)
xgb_predictions = xgb_pipeline.predict(X_test)
xgb_probabilities = (xgb_pipeline.predict_proba(X_test)[:, 1])
xgb_auc = roc_auc_score(y_test,xgb_probabilities)
print("Test AUC:", xgb_auc)
print(classification_report(y_test,xgb_predictions,target_names=["Not Readmitted","Readmitted <30d"],zero_division=0))
results["XGBoost"] = {"auc": xgb_auc,"predictions": xgb_predictions,"probabilities": xgb_probabilities}

# COMPARE MODELS
print("MODEL COMPARISON")
print("Logistic Regression AUC:",logistic_auc)
print("Random Forest AUC:",rf_auc)
print("XGBoost AUC:",xgb_auc)

# FIND BEST MODEL
best_model_name = ""
best_auc = 0
for model_name in results:
    current_auc = results[model_name]["auc"]
    if current_auc > best_auc:
        best_auc = current_auc
        best_model_name = model_name
print("\nBest Model:")
print(best_model_name)
print("Best AUC:")
print(best_auc)
id="plot_part"

# STEP 7: VISUALIZATION
print("\nGenerating plots...")
# CREATE FIGURE
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Hospital Readmission Analysis",fontsize=14,fontweight="bold")

# PLOT 1: READMISSION BY AGE
ax1 = fig.add_subplot(3, 3, 1)
q1_sorted = q1.sort_values("age")
bars = ax1.barh(q1_sorted["age"],q1_sorted["readmission_rate_pct"])
ax1.set_title("Readmission by Age")
ax1.set_xlabel("Rate (%)")
ax1.grid(True, axis="x", alpha=0.3)

# PLOT 2: INSULIN VS HOSPITAL STAY
ax2 = fig.add_subplot(3, 3, 2)
q3_clean = q3.dropna()
ax2.bar(q3_clean["insulin"],q3_clean["avg_days"])
ax2.set_title("Insulin vs Hospital Stay")
ax2.set_ylabel("Days in hospital")
ax2.grid(True, axis="y", alpha=0.3)

# PLOT 3: A1C ANALYSIS
ax3 = fig.add_subplot(3, 3, 3)
q4_clean = q4[q4["A1Cresult"] != "None"]
ax3.bar(q4_clean["A1Cresult"],q4_clean["readmit_rate_pct"])
ax3.set_title("A1C vs Readmission")
ax3.set_ylabel("Readmission Rate (%)")
ax3.grid(True, axis="y", alpha=0.3)

# PLOT 4: MODEL COMPARISON
ax4 = fig.add_subplot(3, 3, 4)
model_names = list(results.keys())
auc_values = []
for name in model_names:
    auc_values.append(results[name]["auc"])

ax4.barh(model_names, auc_values)
ax4.set_title("Model AUC Comparison")
ax4.set_xlabel("AUC Score")
ax4.grid(True, axis="x", alpha=0.3)

# PLOT 5: ROC CURVES
ax5 = fig.add_subplot(3, 3, 5)
# Logistic ROC
fpr, tpr, _ = roc_curve(y_test,logistic_probabilities)
ax5.plot(fpr, tpr, label="Logistic")
# Random Forest ROC
fpr, tpr, _ = roc_curve(y_test,rf_probabilities)
ax5.plot(fpr, tpr, label="Random Forest")
# XGBoost ROC
fpr, tpr, _ = roc_curve(y_test,xgb_probabilities)
ax5.plot(fpr, tpr, label="XGBoost")
# diagonal line
ax5.plot([0, 1], [0, 1], "--")
ax5.set_title("ROC Curves")
ax5.legend()
ax5.grid(True)

# PLOT 6: CONFUSION MATRIX
ax6 = fig.add_subplot(3, 3, 6)
cm = confusion_matrix(y_test,results[best_model_name]["predictions"])
disp = ConfusionMatrixDisplay(cm,display_labels=["No Readmit", "<30 Days"])
disp.plot(ax=ax6, colorbar=False)
ax6.set_title("Confusion Matrix")

# PLOT 7: FEATURE IMPORTANCE
ax7 = fig.add_subplot(3, 3, 7)
rf_model = rf_pipeline.named_steps["model"]
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({"feature": feature_columns,"importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="importance")
ax7.barh(feature_importance_df["feature"],feature_importance_df["importance"])
ax7.set_title("Feature Importance (Random Forest)")

# PLOT 8: CLASS BALANCE
ax8 = fig.add_subplot(3, 3, 8)
counts = y.value_counts()
ax8.bar(["Not Readmit", "Readmit <30"],counts.values)
ax8.set_title("Class Distribution")

# PLOT 9: SQL FEATURE IMPACT (SIMPLE VERSION)
ax9 = fig.add_subplot(3, 3, 9)
sql_features = [
    "total_prior_visits",
    "procedures_per_day",
    "meds_per_diagnosis",
    "elderly_flag",
    "insulin_adjusted",
    "a1c_tested",
    "a1c_abnormal",
    "frequent_emergency",
    "on_diabetes_med"]

sql_importance = 0
other_importance = 0
for i in range(len(feature_columns)):
    feature_name = feature_columns[i]
    importance_value = importances[i]
    if feature_name in sql_features:
        sql_importance += importance_value
    else:
        other_importance += importance_value
ax9.bar(["SQL Features", "Original Features"],[sql_importance, other_importance])
ax9.set_title("SQL vs Original Features Impact")

# FINALIZE
plt.tight_layout()
plt.savefig("project_results.png",dpi=150,bbox_inches="tight")
plt.show()
print("\nPlot saved: project_results.png")
print("\nDONE!")



