import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
    font-size: 90px;
    font-weight: 800;
    color: white;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 10px;
}

    .subtitle {
        font-size: 28px;
        font-weight: bold;
        color: #4CAF50;
        margin-top: 30px;
    }

    .content {
        font-size: 18px;
        line-height: 1.8;
        color: #D3D3D3;
        text-align: justify;
        background-color: #161B22;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
    }

    .highlight {
        color: #4CAF50;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<p class="title">🛡️ Fake Job Detection Using Machine Learning</p>',
    unsafe_allow_html=True
)

# Subtitle
st.markdown(
    '<p class="subtitle">📌 Problem Statement</p>',
    unsafe_allow_html=True
)

# Content Box
st.markdown("""
<div class="content">

Online job portals have become one of the primary sources for employment opportunities. However, the rapid growth of digital recruitment platforms has also led to an increase in <span class="highlight">fraudulent job postings</span> that mislead job seekers through fake offers, scams, and misleading company information.

These fraudulent postings can result in financial loss, identity theft, and reduced trust in online hiring platforms.

This project aims to develop a <span class="highlight">Machine Learning–based system</span> capable of detecting whether a job posting is genuine or fraudulent. The system analyzes multiple features of job advertisements, including job descriptions, company profiles, employment type, required experience, education, industry, and other relevant attributes to perform accurate classification.

Using techniques such as <span class="highlight">data preprocessing</span>, <span class="highlight">feature engineering</span>, <span class="highlight">natural language processing</span>, and <span class="highlight">machine learning algorithms</span>, the proposed system helps improve the reliability of online recruitment platforms and assists users in identifying suspicious job postings effectively.

</div>
""", unsafe_allow_html=True)