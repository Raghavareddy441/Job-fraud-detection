import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fake Job Detection EDA",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("Fake Job Posting Detection - Exploratory Data Analysis")

st.markdown("""
This dashboard provides detailed exploratory data analysis of the fake job posting dataset.
""")

st.markdown("---")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fake_job_postings.csv")

# ==========================================
# DATASET OVERVIEW
# ==========================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", df.shape[0])

with col2:
    st.metric("Total Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.markdown("---")

# ==========================================
# TARGET VARIABLE DISTRIBUTION
# ==========================================

st.header("Fraudulent vs Non-Fraudulent Jobs")

fraud_count = df["fraudulent"].value_counts()

fig = px.pie(
    names=["Real Jobs", "Fake Jobs"],
    values=fraud_count.values,
    title="Distribution of Job Postings",
    hole=0.4
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Majority of job postings are genuine.
- Fraudulent jobs form a smaller portion of the dataset.
- The dataset is imbalanced, which is common in fraud detection problems.
""")

st.markdown("---")

# ==========================================
# EMPLOYMENT TYPE ANALYSIS
# ==========================================

st.header("Employment Type Analysis")

employment_counts = df["employment_type"].fillna("Unknown").value_counts()

fig = px.bar(
    x=employment_counts.index,
    y=employment_counts.values,
    labels={"x":"Employment Type", "y":"Count"},
    title="Employment Type Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Full-time jobs dominate the dataset.
- Some employment types appear very rarely.
- Fraudulent jobs are commonly associated with vague employment information.
""")

st.markdown("---")

# ==========================================
# WORK FROM HOME ANALYSIS
# ==========================================

st.header("Work From Home Analysis")

tele_counts = df["telecommuting"].value_counts()

fig = px.bar(
    x=["Office Work", "Work From Home"],
    y=tele_counts.values,
    labels={"x":"Work Type", "y":"Count"},
    title="Work From Home vs Office Jobs"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Most jobs are office-based.
- Work-from-home opportunities are fewer.
- Many fake jobs misuse remote work opportunities to attract applicants.
""")

st.markdown("---")

# ==========================================
# REQUIRED EXPERIENCE
# ==========================================

st.header("Required Experience Analysis")

experience_counts = df["required_experience"].fillna("Unknown").value_counts().head(10)

fig = px.bar(
    x=experience_counts.index,
    y=experience_counts.values,
    labels={"x":"Experience Level", "y":"Count"},
    title="Top Required Experience Levels"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Entry-level and mid-level jobs are most common.
- Fraudulent jobs often avoid specifying clear experience requirements.
""")

st.markdown("---")

# ==========================================
# REQUIRED EDUCATION
# ==========================================

st.header("Required Education Analysis")

education_counts = df["required_education"].fillna("Unknown").value_counts().head(10)

fig = px.bar(
    x=education_counts.index,
    y=education_counts.values,
    labels={"x":"Education", "y":"Count"},
    title="Required Education Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Bachelor's degree appears most frequently.
- Some fake jobs use very generic education requirements.
""")

st.markdown("---")

# ==========================================
# COMPANY LOGO ANALYSIS
# ==========================================

st.header("Company Logo Analysis")

logo_counts = df["has_company_logo"].value_counts()

fig = px.pie(
    names=["No Logo", "Has Logo"],
    values=logo_counts.values,
    title="Company Logo Presence"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Genuine companies are more likely to include logos.
- Missing company logos can sometimes indicate suspicious postings.
""")

st.markdown("---")

# ==========================================
# QUESTIONS ANALYSIS
# ==========================================

st.header("Screening Questions Analysis")

question_counts = df["has_questions"].value_counts()

fig = px.bar(
    x=["No Questions", "Has Questions"],
    y=question_counts.values,
    labels={"x":"Questions", "y":"Count"},
    title="Presence of Screening Questions"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Many genuine companies include screening questions.
- Fake jobs often avoid detailed applicant filtering.
""")

st.markdown("---")

# ==========================================
# TOP INDUSTRIES
# ==========================================

st.header("Top Industries")

industry_counts = df["industry"].fillna("Unknown").value_counts().head(10)

fig = px.bar(
    x=industry_counts.values,
    y=industry_counts.index,
    orientation='h',
    labels={"x":"Count", "y":"Industry"},
    title="Top 10 Industries"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Insights

- Technology and marketing industries dominate.
- Some industries are more targeted by fraudulent postings.
""")

st.markdown("---")

# ==========================================
# WORD CLOUD COMPARISON
# ==========================================

st.header("Word Cloud Comparison")

fake_text = " ".join(
    df[df["fraudulent"] == 1]["description"].fillna("").astype(str)
)

real_text = " ".join(
    df[df["fraudulent"] == 0]["description"].fillna("").astype(str)
)

col1, col2 = st.columns(2)

# ------------------------------------------
# FAKE WORD CLOUD
# ------------------------------------------

with col1:

    st.subheader("Fraudulent Job Word Cloud")

    fake_wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Reds"
    ).generate(fake_text)

    fig1, ax1 = plt.subplots(figsize=(8,4))

    ax1.imshow(fake_wc, interpolation="bilinear")

    ax1.axis("off")

    st.pyplot(fig1)

    st.markdown("""
    ### Insights

    - Fraudulent jobs use highly attractive wording.
    - Terms related to money and easy work appear frequently.
    - Scam-oriented language is more common.
    """)

# ------------------------------------------
# REAL WORD CLOUD
# ------------------------------------------

with col2:

    st.subheader("Non-Fraudulent Job Word Cloud")

    real_wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate(real_text)

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.imshow(real_wc, interpolation="bilinear")

    ax2.axis("off")

    st.pyplot(fig2)

    st.markdown("""
    ### Insights

    - Genuine jobs focus on technical skills and qualifications.
    - Professional terminology appears more frequently.
    - Real companies emphasize responsibilities and teamwork.
    """)

st.markdown("---")



st.markdown("""
### Insights

- Fraudulent postings show relationships with telecommuting and missing company logos.
- Correlation analysis helps identify important predictive features.
""")

st.markdown("---")

# ==========================================
# FINAL SUMMARY
# ==========================================

st.header("EDA Summary")

st.success("""
Key Findings:
- Dataset is highly imbalanced.
- Fraudulent jobs commonly use attractive language.
- Work-from-home and missing company details are important indicators.
- Company logos and structured descriptions are more common in genuine jobs.
- Feature engineering from textual and categorical data is important for fraud detection.
""")