import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from resume_parser import extract_text_from_pdf, extract_text_from_docx
from feedback_generator import analyze_resume_with_llm
from pdf_exporter import export_feedback_as_pdf
import cohere
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'cohere_client' not in st.session_state:
    st.session_state.cohere_client = None

# Match score between resume and job description
def match_score(resume_text, job_description):
    vect = CountVectorizer().fit_transform([resume_text, job_description])
    score = cosine_similarity(vect)[0][1]
    return round(score * 100, 2)

def configure_api_key():
    st.sidebar.title("🔑 API Setup")
    st.sidebar.info("To use the resume analysis features, get your free Cohere API key from [dashboard.cohere.com](https://dashboard.cohere.com) and paste it below.")
    
    api_key = st.sidebar.text_input(
        "Enter Cohere API Key:",
        type="password",
        placeholder="prod_...",
        help="Required to access AI-powered resume feedback"
    )
    if api_key:
        try:
            co = cohere.Client(api_key)
            _ = co.chat(message="Hello", model="command-r-mini")
            st.session_state.api_key_configured = True
            st.session_state.cohere_client = co
            st.sidebar.success("✅ API key validated!")
            return api_key
        except Exception as e:
            st.sidebar.error(f"❌ Error validating API key: {str(e)}")
            st.session_state.api_key_configured = False
            return None
    return None

def main():
    st.set_page_config(
        page_title="Resume Sudharak",
        page_icon="🧠",
        layout="wide"
    )

    api_key = configure_api_key()
    if not st.session_state.api_key_configured:
        st.warning("Please enter your Cohere API key in the sidebar to get started.")
        return

    # 🎨 Logo and Title Layout
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("logo.jpg", width=100)
    with col_title:
        st.markdown("<h1 style='margin-bottom:0;'> Resume Sudharak</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px; color:gray;'>Empowering job seekers with AI-driven resume reform.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Job Information")
        job_role = st.text_input("Target Job Role*", "", help="Required field")
        job_description = st.text_area("Job Description (Optional)", height=150)
        persona = st.selectbox("Choose Resume Tone", ["Confident", "Professional", "Friendly"])
        uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)*", type=["pdf", "docx"])

    with col2:
        st.subheader("Analysis Results")

        if uploaded_file and job_role:
            if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
                with st.spinner("Extracting text from resume..."):
                    try:
                        if uploaded_file.type == "application/pdf":
                            resume_text = extract_text_from_pdf(uploaded_file)
                        else:
                            resume_text = extract_text_from_docx(uploaded_file)
                        if not resume_text.strip():
                            st.error("No text could be extracted from the resume.")
                            return
                        st.session_state.resume_text = resume_text
                    except Exception as e:
                        st.error(f"Error extracting text: {str(e)}")
                        return

                with st.spinner("Analyzing resume with AI..."):
                    try:
                        feedback = analyze_resume_with_llm(
                            resume_text, job_role, job_description, st.session_state.cohere_client
                        )
                        st.session_state.feedback = feedback
                        st.session_state.analysis_done = True
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"Error analyzing resume: {str(e)}")
                        return

                # 📈 Match Score Visualization
                if job_description:
                    score = match_score(resume_text, job_description)
                    st.metric("Match Score", f"{score}%")
                    st.progress(score / 100)

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        title={'text': "Resume Match Score"},
                        gauge={'axis': {'range': [0, 100]}}
                    ))
                    st.plotly_chart(fig)

                    # 🧾 Sidebar Summary
                    st.sidebar.markdown("### 🔍 Quick Summary")
                    st.sidebar.markdown(f"**Job Role:** {job_role}")
                    st.sidebar.markdown(f"**Tone Preference:** {persona}")
                    st.sidebar.markdown(f"**Match Score:** {score}%")

                    # 🧠 Static Skill Coverage Pie Chart (can be made dynamic later)
                    st.subheader("🧠 Skill Coverage")
                    skills_found = 7
                    skills_missing = 3
                    fig2, ax2 = plt.subplots()
                    ax2.pie([skills_found, skills_missing],
                            labels=["Matched", "Missing"],
                            autopct="%1.1f%%",
                            colors=["#4CAF50", "#F44336"])
                    ax2.set_title("Skill Coverage")
                    st.pyplot(fig2)

        if st.session_state.feedback:
            st.subheader("📋 Feedback Report")
            with st.expander("View Detailed Feedback", expanded=True):
                st.markdown(st.session_state.feedback)
            if st.button("📥 Download Feedback as PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_path = export_feedback_as_pdf(st.session_state.feedback)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Download PDF Report",
                                data=f,
                                file_name="resume_feedback_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
        elif st.session_state.analysis_done:
            st.info("Upload a resume and enter a job role to get started.")

if __name__ == "__main__":
    main()


