
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

st.subheader("📌 입력 변수 설명")

with st.expander("5가지 입력값은 무엇을 의미하나요?"):
    st.markdown("""
    **① 총경동맥 직경 D₀ (mm)**  
    혈관의 기준 직경을 나타냅니다. 혈관의 단면적과 혈류 특성에 영향을 주는 변수입니다.

    **② 혈관 팽창률 (%)**  
    심장 박동에 따라 혈관 직경이 얼마나 변화하는지를 나타냅니다.
    혈관의 탄성 및 순응성과 관련된 지표로 활용됩니다.

    **③ 수축기 혈압 SBP (mmHg)**  
    심장이 수축하여 혈액을 내보낼 때의 최고 혈압입니다.

    **④ 평균 혈류속도 v_mean (m/s)**  
    혈액이 혈관을 통과하는 평균적인 속도를 나타냅니다.

    **⑤ 이완기 혈압 DBP (mmHg)**  
    심장이 이완되어 있을 때 유지되는 혈압입니다.
    """)

with st.expander("🔗 입력값들은 어떤 관계가 있나요?"):
    st.markdown(r"""
    이 프로그램의 입력 변수들은 서로 완전히 독립된 값이 아니라
    **혈류역학적 관계를 통해 서로 연결**되어 있습니다.

    **혈압과 혈류**  
    혈류량 \(Q\)는 기본적으로 혈관 양 끝의 압력 차이와 혈관 저항의 영향을 받습니다.

    \[
    Q = \frac{\Delta P}{R}
    \]

    **혈관 직경과 혈류**  
    혈관의 단면적 \(A\)와 평균 혈류속도 \(v\)를 이용하면 혈류량은 다음과 같이 표현할 수 있습니다.

    \[
    Q = Av
    \]

    원형 혈관으로 단순화하면

    \[
    A = \pi\left(\frac{D}{2}\right)^2
    \]

    이므로 혈관 직경과 혈류속도는 혈류량을 결정하는 데 함께 작용합니다.

    **혈압과 혈관 팽창**  
    심장 박동으로 혈압이 변화하면 혈관 직경도 변화합니다.
    혈관 팽창률은 이러한 혈관의 동적 변화를 표현하기 위한 변수입니다.

    따라서 본 모델에서는 **혈압, 혈관 직경, 혈관 팽창률, 혈류속도 사이의 관계를
    종합적으로 이용하여 가상 혈류 상태를 분류합니다.**
    """)

st.caption(
    "※ 본 프로그램은 혈류역학적 관계를 바탕으로 생성한 합성 데이터를 이용한 "
    "시뮬레이션 모델이며 실제 의료 진단을 목적으로 하지 않습니다."
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
    
st.divider()
st.subheader("🔍 결과 해석")

if prediction == "Normal":
    st.success(
        "입력된 혈관 직경, 팽창률, 혈압 및 평균 혈류속도의 조합이 "
        "합성 데이터의 정상형 패턴과 가장 유사하게 나타났습니다."
    )

elif prediction == "Hypertension":
    st.warning(
        "입력된 측정값의 조합이 합성 데이터의 고혈압형 패턴과 "
        "가장 유사하게 나타났습니다. 혈압 관련 입력값이 "
        "분류에 영향을 줄 수 있습니다."
    )

elif prediction == "Arterial_Stiffness":
    st.warning(
        "입력된 측정값의 조합이 합성 데이터의 동맥 경직형 패턴과 "
        "가장 유사하게 나타났습니다. 특히 혈관 팽창률은 "
        "혈관의 탄성 특성과 관련된 변수입니다."
    )

elif prediction == "Stenosis":
    st.error(
        "입력된 측정값의 조합이 합성 데이터의 협착형 패턴과 "
        "가장 유사하게 나타났습니다. 혈관 직경과 혈류속도의 "
        "관계가 분류에 영향을 줄 수 있습니다."
    )

st.caption(
    "※ 이 설명은 합성 데이터에서 학습된 패턴에 대한 해석이며, "
    "실제 질환의 원인이나 의학적 진단을 의미하지 않습니다."
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
