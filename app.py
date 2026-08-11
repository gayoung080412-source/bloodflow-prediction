
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="가상 혈류 상태 예측 프로그램",
    page_icon="🫀",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("final_bloodflow_model.pkl")

model = load_model()

st.title("🫀 가상 혈류 상태 예측 프로그램")
st.caption("총경동맥 관련 측정값 5개를 입력해 가상 혈류 상태를 분류합니다.")

st.info(
    "이 프로그램은 합성 데이터 기반 시뮬레이션 모델입니다. "
    "실제 의학적 진단이나 치료 판단에 사용할 수 없습니다."
)

st.subheader("환자 측정값 입력")

st.caption(
    "※ 아래 범위는 본 합성 데이터 기반 모델에서 사용하기 위한 입력 범위이며, "
    "실제 임상적 정상범위 또는 진단 기준을 의미하지 않습니다."
)

col1, col2 = st.columns(2)

with col1:
    d0_mm = st.slider(
        "총경동맥 직경 D₀ (mm)",
        min_value=4.0,
        max_value=8.0,
        value=6.2,
        step=0.1
    )

    distension_percent = st.slider(
        "혈관 팽창률 (%)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.1
    )

    sbp = st.slider(
        "수축기 혈압 SBP (mmHg)",
        min_value=90,
        max_value=180,
        value=120,
        step=1
    )

with col2:
    v_mean = st.slider(
        "평균 혈류속도 (m/s)",
        min_value=0.10,
        max_value=0.60,
        value=0.30,
        step=0.01
    )

    dbp = st.slider(
        "이완기 혈압 DBP (mmHg)",
        min_value=50,
        max_value=110,
        value=80,
        step=1
    )

if st.button("분석하기", type="primary", use_container_width=True):

    d0 = d0_mm / 1000
    distension_ratio = distension_percent / 100

    patient_input = pd.DataFrame({
        "D0": [d0],
        "Distension_ratio": [distension_ratio],
        "SBP": [sbp],
        "v_mean": [v_mean],
        "DBP": [dbp]
    })

    prediction = model.predict(patient_input)[0]
    probabilities = model.predict_proba(patient_input)[0]

    result_df = pd.DataFrame({
        "상태": model.classes_,
        "예측 확률(%)": probabilities * 100
    }).sort_values("예측 확률(%)", ascending=False)

    st.divider()
    st.subheader("분석 결과")

    state_korean = {
        "Normal": "정상형",
        "Hypertension": "고혈압형",
        "Arterial_Stiffness": "동맥 경직형",
        "Stenosis": "협착형"
    }

    st.metric(
        "가장 유사한 가상 혈류 상태",
        state_korean.get(prediction, prediction)
    )

    st.write("#### 상태별 예측 확률")

    chart_df = result_df.copy()
    chart_df["표시 상태"] = chart_df["상태"].map(
        lambda x: state_korean.get(x, x)
    )
    chart_df = chart_df.set_index("표시 상태")

    st.bar_chart(chart_df["예측 확률(%)"])

    display_df = result_df.copy()
    display_df["상태"] = display_df["상태"].map(
        lambda x: state_korean.get(x, x)
    )
    display_df["예측 확률(%)"] = display_df["예측 확률(%)"].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "해석 주의: 이 결과는 실제 환자 임상자료가 아니라 "
        "수학적 혈류 모델로 생성한 합성 데이터를 학습한 결과입니다."
    )

with st.expander("입력 변수 설명"):
    st.markdown(
        """
- **D₀**: 총경동맥의 기준 직경
- **혈관 팽창률**: 박동에 따른 직경 변화 비율
- **SBP**: 수축기 혈압
- **평균 혈류속도**: 총경동맥의 평균 Doppler 혈류속도
- **DBP**: 이완기 혈압
        """
    )
