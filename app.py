import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Chatbot Comparison", layout="centered")

st.title("🤖 Chatbot Model Comparison")
st.write("Compare the performance of three different chatbot models: GPT, BERT, and NLTK")

# Sample input message
user_input = st.text_input("Enter your message:", "Hello, how can I help you?")

# Fake responses
responses = {
    "GPT": "Hello! How can I assist you today?",
    "BERT": "Hi! What can I help you with?",
    "NLTK": "Hello! How may I help you?",
}

# Show responses
if user_input:
    st.subheader("💬 Responses from Models")
    for model, reply in responses.items():
        st.markdown(f"**{model}:** {reply}")

    # Fake metrics
    st.subheader("📊 Performance Comparison")

    data = {
        "Model": ["GPT", "BERT", "NLTK"],
        "Accuracy": [0.95, 0.88, 0.75],
        "Response Time (ms)": [150, 200, 50],
        "Fluency Score": [9.5, 8.2, 6.0],
    }

    df = pd.DataFrame(data)

    st.dataframe(df.set_index("Model"))

    # Define light red shades
    red_shades = ["#f28e8e", "#f2b6b6", "#f2dcdc"]

    # Plot Accuracy
    st.markdown("### Accuracy")
    fig1, ax1 = plt.subplots()
    ax1.bar(df["Model"], df["Accuracy"], color=red_shades)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("Accuracy")
    st.pyplot(fig1)

    # Plot Response Time
    st.markdown("### Response Time (Lower is better)")
    fig2, ax2 = plt.subplots()
    ax2.bar(df["Model"], df["Response Time (ms)"], color=red_shades)
    ax2.set_ylabel("Milliseconds")
    st.pyplot(fig2)

    # Plot Fluency
    st.markdown("### Fluency Score (Out of 10)")
    fig3, ax3 = plt.subplots()
    ax3.bar(df["Model"], df["Fluency Score"], color=red_shades)
    ax3.set_ylim([0, 10])
    ax3.set_ylabel("Score")
    st.pyplot(fig3)

st.markdown("---")
st.caption("🔬 Metrics are illustrative and not based on actual model inference.")
