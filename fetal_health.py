import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

st.title("Fetal Health Classification: A Machine Learning App")
st.image('fetal_health_image.gif', width = 700)
st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

# Load the trained models
dt_pickle = open('dt_fetal.pickle', 'rb')
dt_clf = pickle.load(dt_pickle) 
dt_pickle.close()

rf_pickle = open('rf_fetal.pickle', 'rb')
rf_clf = pickle.load(rf_pickle)
rf_pickle.close()

ada_pickle = open('ada_fetal.pickle', 'rb')
ada_clf = pickle.load(ada_pickle)
ada_pickle.close()

sv_pickle = open('sv_fetal.pickle', 'rb')
sv_clf = pickle.load(sv_pickle)
sv_pickle.close()

# Use default dataset for automation
default_df = pd.read_csv('fetal_health.csv')
default_df = default_df.dropna().reset_index(drop=True)
default_df = default_df.drop(columns = ['fetal_health'])

st.sidebar.header('Fetal Health Features Input')
user_data = st.sidebar.file_uploader('Upload your data')
st.sidebar.warning("⚠️ Ensure your data strictly follows the format outlined below.")
st.sidebar.dataframe(default_df.head())

ml_model = st.sidebar.radio('Choose Model for Prediction',
                       ["Decision Tree", "Random Forest", "AdaBoost", "Soft Voting"])

if ml_model == "Decision Tree":
    clf = dt_clf
elif ml_model == "Random Forest":
    clf = rf_clf
elif ml_model == "AdaBoost":
    clf = ada_clf
else:
    clf = sv_clf


st.sidebar.info(f'Your selection: **{ml_model}**', icon = ":material/check_circle:")


# Main Page

if user_data is None:
    st.info("*Please upload data to proceed.*", icon = "ℹ️")
else:
    st.success("*CSV like uploaded successfully.*", icon = "✅")

    st.subheader(f"Predicting Fetal Health Class Using {ml_model} Model")
    user_df = pd.read_csv(user_data)

    # Dropping null values
    user_df = user_df.dropna().reset_index(drop = True)
    default_df = default_df.dropna().reset_index(drop = True)
    
    # Remove output (fetal_health) from user data
    if 'fetal_health' in user_df.columns:
        user_df = user_df.drop(columns = ['fetal_health'])
   
   # Ensure user data has the same feature order as the original training features
    feature_cols = [c for c in default_df.columns if c != 'fetal_health']
    user_df = user_df[feature_cols]

   # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([default_df, user_df], axis = 0)

   # Number of rows in original dataframe
    original_rows = default_df.shape[0]

   # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

   # Split data into original and user dataframes using row index
    default_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

   # Predict
    predictions = clf.predict(user_df_encoded)

    # Probabilities (shape: n_rows x n_classes)
    proba = clf.predict_proba(user_df_encoded) 

    # Also compute the top-class probability per row
    top_prob = proba.max(axis=1)                
    top_prob_pct = (top_prob * 100).round(2) 

    # Final table
    result_df = user_df.copy()
    result_df["Predicted Fetal Health"] = predictions 
    result_df["Top Probability (%)"] = top_prob_pct

    # Used chatgpt to color the cells
    def color_pred(val):
        color_map = {
            "Normal": "lime",
            "Suspect": "yellow",
            "Pathological": "orange"
        }
        color = color_map.get(val, "white")
        return f"background-color: {color}; color: black; font-weight: bold;"

    # Apply styling only to the predicted column
    styled_df = result_df.style.applymap(
        color_pred, subset=["Predicted Fetal Health"]
    )

    st.dataframe(styled_df)

# used chatGPT to debug

    # Model Visualizations
    st.subheader("Model Performance and Insights")
    model_prefix = {'Decision Tree': 'dt_', 'Random Forest': 'rf_', 'AdaBoost': 'ada_', 'Soft Voting': 'sv_'}[ml_model]

    assets = {
        "feature_imp_img": f"{model_prefix}feature_imp.svg",
        "confusion_mat_img": f"{model_prefix}confusion_matrix.svg",
        "class_report_csv": f"{model_prefix}class_report.csv",
    }

    def show_image(path_str: str, title: str, caption: str):
        p = Path(path_str)
        st.write(f"### {title}")
        if p.exists():
            st.image(str(p))
            st.caption(caption)

    if ml_model == "Decision Tree":
        class_report_color = 'PuRd'
    elif ml_model == 'Random Forest':
        class_report_color = 'Oranges'
    elif ml_model == 'AdaBoost':
        class_report_color = 'YlGn'
    else:
        class_report_color = 'Blues'

    def show_class_report(csv_path: str):
        p = Path(csv_path)
        st.write("### Classification Report")
        if p.exists():
            report_df = pd.read_csv(p, index_col=0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap=class_report_color).format(precision=2),
                        use_container_width=True)
            st.caption("Precision, Recall, F1-Score, and Support for each species.")

    # -----------------------
    # Prediction Performance
    # -----------------------
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    with tab1:
        show_image(
            assets["feature_imp_img"],
            title="Feature Importance",
            caption="Features ranked by relative importance."
        )

    with tab2:
        show_image(
            assets["confusion_mat_img"],
            title="Confusion Matrix",
            caption=f"Confusion matrix for the {ml_model} model."
        )

    with tab3:
        show_class_report(assets["class_report_csv"])


