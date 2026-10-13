import datetime
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="centered")


# Load the default feature column list from training (cached)
@st.cache_resource
def load_columns():
  return joblib.load("model_columns.pkl")


model_columns = load_columns()

st.title("Content Monetization Modeler")

# Model selection and display of evaluation metrics
st.subheader("Model Selection")
results_df = joblib.load("evaluation_metrics.pkl")
active_model = st.selectbox("Choose Regression model", results_df["Model"].tolist())
met = results_df[results_df["Model"] == active_model].iloc[0]
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("R2 Score", f"{met['R2_Score']:.5f}")
col_m2.metric("RMSE", f"{met['RMSE']:,.5f}")
col_m3.metric("MAE", f"{met['MAE']:,.5f}")

# Load the specific model based on user selection
if active_model == "Lasso Regression":
  model = joblib.load("youtube_lasso.pkl")
elif active_model == "Ridge Regression":
  model = joblib.load("youtube_ridge.pkl")
elif active_model == "Random Forest Regression":
  model = joblib.load("youtube_rf.pkl")
elif active_model == "Gradient Boosting Regression":
  model = joblib.load("youtube_gb.pkl")
else:
  model = joblib.load("youtube_lr.pkl")

col1, col2 = st.columns(2)

with col1:
  views = st.number_input("Views", min_value=0, value=50000, step=100)
  likes = st.number_input("Likes", min_value=0, value=1500, step=10)
  comments = st.number_input("Comments", min_value=0, value=150, step=1)

with col2:
  watch_time_minutes = st.number_input(
      "Watch Time (Minutes)", min_value=0.0, value=12000.0, step=50.0
  )
  video_length_minutes = st.number_input(
      "Video Length (Minutes)", min_value=0.0, value=12.5, step=0.5
  )
  subscribers = st.number_input(
      "Subscribers", min_value=0, value=25000, step=100
  )

col3, col4, col5 = st.columns(3)

with col3:
  category = st.selectbox(
      "Category",
      ["Gaming", "Tech", "Entertainment", "Education", "Lifestyle", "Music"],
  )

with col4:
  device = st.selectbox("Primary Device", ["Mobile", "Desktop", "Tablet", "TV"])

with col5:
  country = st.selectbox("Target Country", ["US", "IN", "UK", "CA", "DE", "AU"])

# Add a calendar date picker (defaults to today's date)
upload_date = st.date_input("Video Upload Date", value=datetime.date.today())

upload_year = upload_date.year
upload_month = upload_date.month
upload_day = upload_date.day
upload_day_name = upload_date.strftime("%A")  # e.g., 'Friday', 'Monday'

# --- 2. Prediction Logic & Feature Alignment ---
if st.button("Predict Revenue", use_container_width=True):
  # Create a base DataFrame initialized to 0 with all exact training columns
  input_encoded = pd.DataFrame(0, index=[0], columns=model_columns)

  # Assign numeric features if they exist in the model columns
  numeric_inputs = {
      "views": views,
      "likes": likes,
      "comments": comments,
      "watch_time_minutes": watch_time_minutes,
      "video_length_minutes": video_length_minutes,
      "subscribers": subscribers,
      "year": upload_year,
      "month": upload_month,
      "day": upload_day,
  }

  for col, val in numeric_inputs.items():
    if col in input_encoded.columns:
      input_encoded[col] = val

  if "engagement_rate" in input_encoded.columns and views > 0:
    input_encoded["engagement_rate"] = (likes + comments) / views

  # FIX: Added missing commas and included alternative casing/name variations
  cat_cols_to_check = [
      f"category_{category}",
      f"Category_{category}",
      f"device_{device}",
      f"Device_{device}",
      f"country_{country}",
      f"Country_{country}",
      f"day_name_{upload_day_name}",
  ]

  # Activate the matching one-hot columns
  for col in cat_cols_to_check:
    if col in input_encoded.columns:
      input_encoded[col] = 1

  st.subheader("🔍 Debug: Input Data Sent to Model")
  st.dataframe(input_encoded)

  # FIX: Pass .to_numpy() to avoid strict scikit-learn feature name mismatch errors
  prediction = model.predict(input_encoded.to_numpy())

  # Display the final predicted revenue nicely
  st.write(f"### Estimated Ad Revenue: ${prediction[0]:,.2f}")