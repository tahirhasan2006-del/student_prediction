
import os
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Optional chart library: Plotly
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student AI Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")

# ================= ADVANCED CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg,#eef2ff,#f8fafc,#e0f2fe);
    font-family: 'Poppins',sans-serif;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#172554,#312e81);
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(120deg,#172554,#2563eb,#7c3aed);
    box-shadow: 0 12px 30px rgba(37,99,235,.18);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 32px;
    font-weight: 800;
    margin: 0 0 10px;
}

.hero p {
    color: #dbeafe;
    margin: 0;
    font-size: 14px;
}

.section-title {
    color: #172554;
    font-size: 23px;
    font-weight: 800;
    margin: 15px 0;
}

.kpi {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 20px rgba(15,23,42,.05);
    min-height: 115px;
}

.kpi-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
}

.kpi-value {
    color: #1d4ed8;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.panel {
    background: white;
    padding: 22px;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(15,23,42,.04);
}

.section-banner {
    display: flex;
    align-items: center;
    gap: 15px;
    background: linear-gradient(120deg,#1d4ed8,#6366f1);
    color: white;
    border-radius: 16px;
    padding: 18px;
    margin: 10px 0 20px;
}

.section-banner .emoji {
    font-size: 30px;
}

.section-banner h3 {
    margin: 0;
    font-size: 19px;
}

.section-banner p {
    color: #dbeafe;
    margin: 5px 0 0;
    font-size: 12px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,.93);
    padding: 24px;
    border: 1px solid #dbeafe;
    border-radius: 20px;
}

div[data-testid="stFormSubmitButton"] button,
.stDownloadButton button {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 45px;
}

.result-panel {
    background: linear-gradient(135deg,#ffffff,#dbeafe);
    padding: 28px;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    margin: 18px 0;
}

.result-number {
    font-size: 42px;
    font-weight: 800;
    color: #1d4ed8;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 30px 0 5px;
}

@media(max-width:768px) {
    .hero h1 {font-size:24px;}
    .result-number {font-size:32px;}
    .block-container {padding:1rem;}
}
</style>
""", unsafe_allow_html=True)


# ================= MODEL LOADING =================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("student_model.pkl not found beside app.py.")
    st.stop()

try:
    artifact = load_model()
    model = artifact["model"]
    features = list(artifact["features"])
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info(
        "Check the scikit-learn version used to train "
        "and load the PKL file."
    )
    st.stop()


# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ================= SIDEBAR =================
st.sidebar.markdown("## 🎓 Student AI")
st.sidebar.caption("Machine Learning Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Predict Grade", "About Project"]
)

st.sidebar.divider()
st.sidebar.caption("Model: Student Performance Prediction")
st.sidebar.caption("Target: Final grade G3 / 20")


# ================= HERO =================
components.html("""
<div style="
font-family:Arial,sans-serif;
padding:25px;
border-radius:20px;
color:white;
background:linear-gradient(120deg,#172554,#2563eb,#7c3aed);
box-shadow:0 10px 30px rgba(37,99,235,.18);
">
    <div style="font-size:11px;letter-spacing:2px;color:#bfdbfe;">
        MACHINE LEARNING ANALYTICS PLATFORM
    </div>
    <h1 style="font-size:29px;margin:12px 0;">
        🎓 Student Performance Prediction
    </h1>
    <p style="font-size:14px;color:#dbeafe;">
        Explore student information, predict final grades,
        and review academic performance.
    </p>
    <div style="margin-top:20px;font-size:12px;color:#dcfce7;">
        ● Dashboard interface ready
    </div>
</div>
""", height=190, scrolling=False)


# ================= DASHBOARD =================
if page == "Dashboard":

    st.markdown(
        '<div class="section-title">📊 Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    total = len(st.session_state.history)
    latest = st.session_state.last_result

    if latest is not None:
        grade_text = f"{latest['prediction']:.2f}/20"
        percentage_text = f"{latest['percentage']:.1f}%"
        g1_text = str(latest["G1"])
        g2_text = str(latest["G2"])
    else:
        grade_text = "--"
        percentage_text = "--"
        g1_text = "--"
        g2_text = "--"

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "PREDICTIONS THIS SESSION", str(total), "🧠"),
        (c2, "LATEST PREDICTED GRADE", grade_text, "🎯"),
        (c3, "ESTIMATED PERCENTAGE", percentage_text, "📈"),
        (c4, "LATEST G1 / G2", f"{g1_text} / {g2_text}", "📚")
    ]

    for col, label, value, emoji in cards:
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{emoji} {label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown(
            '<div class="section-title">📈 Grade Overview</div>',
            unsafe_allow_html=True
        )

        if latest is not None:
            chart_df = pd.DataFrame({
                "Assessment": ["G1", "G2", "Predicted G3"],
                "Grade": [
                    latest["G1"],
                    latest["G2"],
                    latest["prediction"]
                ]
            })

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=chart_df["Assessment"],
                y=chart_df["Grade"],
                text=[
                    f"{v:.2f}" for v in chart_df["Grade"]
                ],
                textposition="outside",
                marker_color=["#60a5fa", "#818cf8", "#7c3aed"]
            ))

            fig.update_layout(
                yaxis=dict(title="Grade out of 20", range=[0, 22]),
                xaxis_title="Assessment",
                height=340,
                margin=dict(l=15, r=15, t=20, b=15),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No prediction yet. Open Predict Grade "
                "and enter student details to create your first chart."
            )

    with right:
        st.markdown(
            '<div class="section-title">🎯 Latest Result</div>',
            unsafe_allow_html=True
        )

        if latest is not None:
            st.markdown(f"""
            <div class="result-panel">
                <div style="color:#64748b;font-size:13px;">
                    PREDICTED FINAL GRADE
                </div>
                <div class="result-number">
                    {latest['prediction']:.2f}<span
                    style="font-size:18px;color:#64748b;"> / 20</span>
                </div>
                <p style="color:#475569;">
                    Estimated percentage: <b>{latest['percentage']:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(latest["prediction"] / 20)

            st.caption(
                "This result is an estimate from the trained model."
            )
        else:
            st.markdown("""
            <div class="panel">
                <h3>Welcome! 👋</h3>
                <p>Make your first prediction to see your analytics here.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🕘 Recent Predictions</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction History CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.caption("Your prediction history will appear here.")


# ================= PREDICT PAGE =================
elif page == "Predict Grade":

    st.markdown(
        '<div class="section-title">🧑‍🎓 Student Prediction Form</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter details using the same encodings used during model training."
    )

    with st.form("advanced_student_form"):

        st.markdown("""
        <div class="section-banner">
            <div class="emoji">👤</div>
            <div>
                <h3>Personal Details</h3>
                <p>Student profile and background</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)

        with p1:
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Sex (dataset code)", [0, 1])

        with p2:
            age = st.number_input(
                "Age", min_value=15, max_value=22, value=17
            )
            address = st.selectbox(
                "Address",
                ["U", "R"],
                format_func=lambda x:
                    "Urban (U)" if x == "U" else "Rural (R)"
            )

        with p3:
            absences = st.number_input(
                "Number of Absences",
                min_value=0, max_value=100, value=2
            )

        st.markdown("""
        <div class="section-banner"
        style="background:linear-gradient(120deg,#047857,#10b981);">
            <div class="emoji">📚</div>
            <div>
                <h3>Academic Details</h3>
                <p>Study habits and previous academic grades</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)

        with a1:
            studytime = st.selectbox(
                "Weekly Study Time (1–4)", [1, 2, 3, 4], index=1
            )
            failures = st.selectbox(
                "Past Class Failures", [0, 1, 2, 3]
            )

        with a2:
            G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
            G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

        with a3:
            Medu = st.selectbox(
                "Mother's Education (0–4)", [0, 1, 2, 3, 4],
                index=2
            )
            Fedu = st.selectbox(
                "Father's Education (0–4)", [0, 1, 2, 3, 4],
                index=2
            )

        st.markdown("""
        <div class="section-banner"
        style="background:linear-gradient(120deg,#7c3aed,#a855f7);">
            <div class="emoji">🏠</div>
            <div>
                <h3>Support & Activities</h3>
                <p>Family support, school support and interests</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)

        def yes_no(label, default=0):
            return st.selectbox(
                label,
                [0, 1],
                index=default,
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with s1:
            schoolsup = yes_no("Extra School Support")
            famsup = yes_no("Family Educational Support")

        with s2:
            paid = yes_no("Extra Paid Classes")
            activities = yes_no("Extra-curricular Activities")

        with s3:
            higher = yes_no("Wants Higher Education", 1)
            internet = yes_no("Internet Access", 1)

        submitted = st.form_submit_button(
            "🚀 Predict Final Grade",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": G1,
            "G2": G2,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "higher": higher,
            "internet": internet,
            "Medu": Medu,
            "Fedu": Fedu
        }

        missing = [f for f in features if f not in input_data]

        if missing:
            st.error(
                "The form does not contain all model features: "
                + ", ".join(missing)
            )
        else:
            try:
                input_df = pd.DataFrame(
                    [input_data],
                    columns=features
                )

                prediction = float(model.predict(input_df)[0])

                if not pd.notna(prediction):
                    raise ValueError("Model returned an invalid prediction.")

                prediction = max(0.0, min(20.0, prediction))
                percentage = prediction / 20 * 100

                result = {
                    "prediction": prediction,
                    "percentage": percentage,
                    "G1": G1,
                    "G2": G2
                }

                st.session_state.last_result = result

                history_row = {
                    "School": school,
                    "Age": age,
                    "G1": G1,
                    "G2": G2,
                    "Predicted G3": round(prediction, 2),
                    "Percentage": round(percentage, 1)
                }

                st.session_state.history.append(history_row)

                st.success("Prediction completed successfully!")

                st.markdown(f"""
                <div class="result-panel">
                    <div style="font-size:13px;color:#64748b;">
                        YOUR PREDICTED FINAL GRADE
                    </div>
                    <div class="result-number">
                        {prediction:.2f} / 20
                    </div>
                    <div style="font-size:16px;color:#334155;">
                        Estimated percentage: <b>{percentage:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(prediction / 20)

                r1, r2, r3 = st.columns(3)

                r1.metric("G1 Grade", f"{G1}/20")
                r2.metric("G2 Grade", f"{G2}/20")
                r3.metric("Predicted G3", f"{prediction:.2f}/20")

                report_df = pd.DataFrame([{
                    **input_data,
                    "Predicted_G3": round(prediction, 2),
                    "Estimated_Percentage": round(percentage, 1)
                }])

                st.download_button(
                    "📥 Download Student Report (CSV)",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name="student_prediction_report.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Educational estimate only. Actual results may differ."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ================= ABOUT PAGE =================
elif page == "About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h3>🎓 Student Performance Prediction System</h3>
        <p>
        This machine learning project estimates a student's final grade
        from academic history, personal information and support factors.
        </p>
        <hr>
        <b>Technologies Used</b>
        <ul>
            <li>Python — model execution and data processing</li>
            <li>Pandas — input data preparation</li>
            <li>Joblib — loading the trained model</li>
            <li>Streamlit — interactive web application</li>
            <li>HTML and CSS — user interface styling</li>
            <li>JavaScript — interactive header</li>
            <li>Plotly — academic charts</li>
        </ul>
        <b>Prediction Target:</b> G3, the final grade out of 20.
        <p style="color:#64748b;font-size:12px;">
        This application is an educational tool, not a guarantee
        of future academic performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Information")

    st.write("Number of model input features:", len(features))
    st.write("Expected feature names:", ", ".join(features))


# ================= FOOTER =================
st.markdown("""
<div class="footer">
    Student AI Analytics Dashboard<br>
    Built with Python · Streamlit · HTML · CSS · JavaScript
</div>
""", unsafe_allow_html=True)
