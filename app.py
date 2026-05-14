import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import hstack, csr_matrix

# ======================================
# Load Saved Files
# ======================================

with open("fraud_detection_model.pkl", "rb") as f:
    model_xgb = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("one_hot_encoding.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("ordinal_encoding.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

with open("frequency_encoding.pkl", "rb") as f:
    freq_encodings = pickle.load(f)

# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="💼",
    layout="centered"
)

# ======================================
# Title
# ======================================

st.title("💼 Fake Job Posting Detection System")

st.markdown("""
This application helps identify whether a job posting is **REAL** or **FAKE** using Machine Learning.

Please fill in the job details below.
""")

st.markdown("---")

# ======================================
# Job Information Section
# ======================================

st.header("📌 Job Information")

job_title = st.text_input(
    "What is the Job Title?",
    placeholder="Example: Software Engineer"
)

company_profile = st.text_area(
    "Tell us about the Company",
    placeholder="Example: Infosys is a global IT services company..."
)

job_description = st.text_area(
    "Describe the Job Role",
    placeholder="Explain responsibilities, daily tasks, technologies used, etc."
)

requirements = st.text_area(
    "What skills or qualifications are required?",
    placeholder="Example: Python, SQL, Communication Skills..."
)

benefits = st.text_area(
    "What benefits does the company provide?",
    placeholder="Example: Health Insurance, Paid Leave, Bonus..."
)

# ======================================
# Employment Details
# ======================================

st.header("🧾 Employment Details")

employment_type = st.selectbox(
    "What type of job is this?",
    [
        "Full-time",
        "Part-time",
        "Contract",
        "Temporary",
        "Other"
    ]
)

required_experience = st.selectbox(
    "What experience level is required?",
    [
        "Internship",
        "Entry level",
        "Associate",
        "Mid-Senior level",
        "Director",
        "Executive",
        "Not Applicable"
    ]
)

required_education = st.selectbox(
    "What education level is required?",
    [
        "High School or equivalent",
        "Bachelor's Degree",
        "Master's Degree",
        "Associate Degree",
        "Certification",
        "Some College Coursework Completed",
        "Unspecified"
    ]
)

# ======================================
# Company Details
# ======================================

st.header("🏢 Company & Location Details")

industry = st.text_input(
    "Which industry does the company belong to?",
    placeholder="Example: Information Technology"
)

function = st.text_input(
    "What is the job function?",
    placeholder="Example: Engineering, Marketing, HR"
)

department = st.text_input(
    "Which department is hiring?",
    placeholder="Example: IT Department"
)

location = st.text_input(
    "Where is the job located?",
    placeholder="Example: Hyderabad, India"
)

# ======================================
# Additional Information
# ======================================

st.header("⚙️ Additional Information")

# ======================================
# Work Mode
# ======================================

st.header("💻 Work Mode")

work_mode = st.selectbox(
    "What is the work mode?",
    [
        "Work From Office",
        "Work From Home"
    ]
)

# Convert for model

if work_mode == "Work From Home":
    telecommuting = 1

else:
    telecommuting = 0

has_company_logo = st.selectbox(
    "Does the company have a logo?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

has_questions = st.selectbox(
    "Does the application process include screening questions?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

st.markdown("---")

# ======================================
# Prediction Button
# ======================================

if st.button("🔍 Verify Job Posting"):

    sample_job = {

        'title': job_title,

        'company_profile': company_profile,

        'description': job_description,

        'requirements': requirements,

        'benefits': benefits,

        'employment_type': employment_type,

        'required_experience': required_experience,

        'required_education': required_education,

        'industry': industry,

        'function': function,

        'department': department,

        'location': location,

        'telecommuting': telecommuting,

        'has_company_logo': has_company_logo,

        'has_questions': has_questions
    }

    test_df = pd.DataFrame([sample_job])

    # ======================================
    # Combine Text Columns
    # ======================================

    text_cols = [
        'title',
        'company_profile',
        'description',
        'requirements',
        'benefits'
    ]

    test_df['combined_text'] = (
        test_df[text_cols]
        .fillna('')
        .agg(' '.join, axis=1)
    )

    # ======================================
    # TF-IDF Features
    # ======================================

    X_tfidf = tfidf.transform(
        test_df['combined_text']
    )

    # ======================================
    # One Hot Encoding
    # ======================================

    X_ohe = ohe.transform(
        test_df[['employment_type']]
    )

    # ======================================
    # Ordinal Encoding
    # ======================================

    ordinal_cols = [
        'required_experience',
        'required_education'
    ]

    test_df[ordinal_cols] = ordinal_encoder.transform(
        test_df[ordinal_cols]
    )

    # ======================================
    # Frequency Encoding
    # ======================================

    freq_cols = [
        'industry',
        'function',
        'department',
        'location'
    ]

    for col in freq_cols:

        test_df[col] = test_df[col].map(
            freq_encodings[col]
        ).fillna(0)

    # ======================================
    # Drop Unwanted Columns
    # ======================================

    drop_cols = [
        'title',
        'company_profile',
        'description',
        'requirements',
        'benefits',
        'combined_text',
        'employment_type'
    ]

    X_sparse_df = test_df.drop(columns=drop_cols)

    # ======================================
    # Convert to Numeric
    # ======================================

    X_sparse_df = X_sparse_df.apply(
        pd.to_numeric,
        errors='coerce'
    ).fillna(0)

    # ======================================
    # Sparse Matrix
    # ======================================

    X_sparse = csr_matrix(
        X_sparse_df.values
    )

    # ======================================
    # Final Features
    # ======================================

    X_test = hstack([
        X_sparse,
        X_ohe,
        X_tfidf
    ])

    # ======================================
    # Prediction
    # ======================================

    proba = model_xgb.predict_proba(X_test)

    fake_probability = proba[0][1]

    st.markdown("---")

    st.header("📊 Prediction Result")

    st.write(
        f"Probability of being Fake: {round(fake_probability * 100, 2)}%"
    )

    if fake_probability > 0.35:

        st.error("⚠️ This Job Posting appears to be FAKE")

    else:

        st.success("✅ This Job Posting appears to be REAL")

# ======================================
# Footer
# ======================================

st.markdown("---")

st.caption("Developed using Streamlit and Machine Learning")