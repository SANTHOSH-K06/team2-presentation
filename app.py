import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Trend AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS (DARK UI) ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background: #161b22;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}
.metric {
    font-size: 36px;
    font-weight: bold;
    color: #4cc9f0;
}
.subtitle {
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
test_size = st.sidebar.slider("Test Data Size", 0.1, 0.4, 0.2)
random_seed = st.sidebar.number_input("Random Seed", value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.write("Using **NSE-TATAGLOBAL** Historical Data")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Info")
st.sidebar.write("Type: Logistic Regression")
st.sidebar.write("Target: Price Movement (Up / Down)")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>📈 Stock Trend AI</h1>
<p style='text-align:center; color:#9ca3af;'>
Predicting Market Direction with Logistic Regression
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
url = "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"
df = pd.read_csv(url)

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# ---------------- TRAIN TEST SPLIT ----------------
X = df[['Close']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_seed
)

# ---------------- MODEL ----------------
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------- DASHBOARD ROW 1 ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Market Overview")

    plt.figure(figsize=(6,4))
    plt.plot(df['Close'], color='#4cc9f0')
    plt.title("Price History: NSE-TATAGLOBAL")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.grid(alpha=0.3)
    st.pyplot(plt)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Model Performance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric'>{accuracy*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Accuracy</div>", unsafe_allow_html=True)

    with c2:
        up_bias = y_pred.mean() * 100
        st.markdown(f"<div class='metric'>{up_bias:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Up Bias</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DASHBOARD ROW 2 ----------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📌 Predict Market Direction")

price = st.number_input("Enter Current Stock Price", value=150.0)

test_df = pd.DataFrame([[price]], columns=['Close'])
pred = model.predict(test_df)[0]
prob = model.predict_proba(test_df)[0][1]

st.subheader("📊 Prediction Visualization (Your Input)")

# Create safe price range from dataset
min_price = float(df['Close'].min())
max_price = float(df['Close'].max())

price_range = np.linspace(min_price, max_price, 500).reshape(-1, 1)

# Convert to DataFrame (VERY IMPORTANT)
price_range_df = pd.DataFrame(price_range, columns=['Close'])

# Predict probabilities
curve_probs = model.predict_proba(price_range_df)[:, 1]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(price_range, curve_probs, label="Model Probability Curve")
ax.axvline(price, linestyle="--", label="Your Input Price")
ax.scatter(price, prob, s=120, zorder=5, label="Your Prediction")

ax.set_xlabel("Stock Price")
ax.set_ylabel("Probability of Going Up")
ax.set_title("Predicted Probability for Given Stock Price")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown(
    f"""
    <h2 style='color:#4cc9f0;'>
    {"UP 📈" if pred == 1 else "DOWN 📉"}
    </h2>
    <p class='subtitle'>Probability of going up: {prob:.2%}</p>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
import joblib

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "logistic_stock_model.pkl")

# Load model
loaded_model = joblib.load("logistic_stock_model.pkl")

# Test prediction
test_val = pd.DataFrame([[150]], columns=['Close'])
loaded_model.predict(test_val)