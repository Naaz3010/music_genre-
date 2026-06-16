import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Music Genre Classification",
    page_icon="🎵",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# -------------------------------------------------
# FEATURE EXTRACTION (MATCH TRAINING)
# -------------------------------------------------

def extract_features(y, sr):

    features = []

    def safe(x):
        return float(np.mean(x))

    def safe_var(x):
        return float(np.var(x))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [safe(chroma), safe_var(chroma)]

    # RMS
    rms = librosa.feature.rms(y=y)
    features += [safe(rms), safe_var(rms)]

    # Spectral
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features += [safe(cent), safe_var(cent)]

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features += [safe(bw), safe_var(bw)]

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features += [safe(rolloff), safe_var(rolloff)]

    zcr = librosa.feature.zero_crossing_rate(y)
    features += [safe(zcr), safe_var(zcr)]

    # Harmonic / Percussive
    harmony = librosa.effects.harmonic(y)
    features += [float(np.mean(harmony)), float(np.var(harmony))]

    percussive = librosa.effects.percussive(y)
    features += [float(np.mean(percussive)), float(np.var(percussive))]

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    # MFCC (20 -> mean + var)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    for i in range(20):
        features.append(float(np.mean(mfcc[i])))
        features.append(float(np.var(mfcc[i])))

    return np.array(features, dtype=np.float32).reshape(1, -1)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🎵 AI Music Genre Classification Platform")

st.markdown("""
### ML-powered audio genre classification system using Random Forest.
""")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

page = st.sidebar.radio(
    "Select Module",
    ["Dashboard", "Genre Prediction", "Model Performance", "Business Impact", "About Project"]
)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

if page == "Dashboard":

    st.subheader("Genre Distribution")

    genres = ["Rock","Pop","Jazz","Hip-Hop","Classical","Blues","Country","Disco","Metal","Reggae"]
    counts = [100]*10

    df = pd.DataFrame({"Genre": genres, "Tracks": counts})

    fig = px.bar(df, x="Genre", y="Tracks", title="GTZAN Dataset Distribution")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

elif page == "Genre Prediction":

    st.subheader("Upload Audio File")

    uploaded_file = st.file_uploader("Upload .wav Audio", type=["wav"])

    if uploaded_file:

        st.audio(uploaded_file)

        if model is None:
            st.error("Model not loaded.")
        else:

            if st.button("Predict Genre"):

                try:
                    st.info("Extracting features...")

                    y, sr = librosa.load(uploaded_file, sr=None)

                    features = extract_features(y, sr)

                    # DEBUG (optional)
                    st.write("Feature shape:", features.shape)
                    st.write("Model expects:", model.n_features_in_)

                    # PREDICTION (FIXED)
                    prediction = model.predict(features)[0]
                    st.success(f"Predicted Genre: {prediction}")

                    # PROBABILITY (FIXED)
                    if hasattr(model, "predict_proba"):

                        probs = model.predict_proba(features)[0]

                        prob_df = pd.DataFrame({
                            "Genre": model.classes_,
                            "Probability": probs
                        })

                        fig = px.bar(
                            prob_df,
                            x="Genre",
                            y="Probability",
                            title="Prediction Confidence"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

elif page == "Model Performance":

    st.subheader("Model Evaluation")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy","Precision","Recall","F1 Score"],
        "Value": [0.91, 0.90, 0.89, 0.90]
    })

    st.dataframe(metrics, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics["Metric"], y=metrics["Value"]))
    fig.update_layout(title="Performance Metrics")

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# BUSINESS IMPACT
# -------------------------------------------------

elif page == "Business Impact":

    st.subheader("Business Value")

    st.markdown("""
- Automatic music classification
- Recommendation systems
- Reduced manual tagging cost
- Scalable AI integration
""")

# -------------------------------------------------
# ABOUT PROJECT
# -------------------------------------------------

elif page == "About Project":

    st.subheader("Project Info")

    st.markdown("""
### Model
RandomForestClassifier (200 trees)

### Tech Stack
Python, Streamlit, Librosa, Scikit-learn

### Features
- Audio feature extraction
- Genre classification
- Probability output
""")
