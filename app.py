
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
    
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "해석 주의: 이 결과는 실제 환자 임상자료가 아니라 "
        "수학적 혈류 모델로 생성한 합성 데이터를 학습한 결과입니다."
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
    # ==========================================
    # 입력값별 해석
    # 합성 데이터의 Normal 생성 조건과 비교
    # ==========================================

    st.divider()
    st.subheader("📊 입력값별 해석")

    st.caption(
        "※ 아래 기준은 실제 임상적 정상범위가 아니라, "
        "본 탐구에서 Normal 합성 데이터를 생성할 때 사용한 범위입니다."
    )


    # ------------------------------------------
    # 1. 총경동맥 직경 D0
    # Normal 생성 범위: 5.5 ~ 7.0 mm
    # ------------------------------------------

    if d0_mm < 5.5:
        st.write(
            f"**총경동맥 직경 D₀: {d0_mm:.1f} mm**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(5.5~7.0 mm)보다 작습니다.  \n"
            "→ 혈관 단면적이 상대적으로 작은 방향의 값이며, "
            "본 합성 데이터에서는 협착형과 관련된 패턴과 일부 유사할 수 있습니다."
        )

    elif d0_mm <= 7.0:
        st.write(
            f"**총경동맥 직경 D₀: {d0_mm:.1f} mm**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(5.5~7.0 mm)에 포함됩니다.  \n"
            "→ 혈관 직경만 보았을 때 Normal 생성 조건에서 크게 벗어나지 않습니다."
        )

    else:
        st.write(
            f"**총경동맥 직경 D₀: {d0_mm:.1f} mm**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(5.5~7.0 mm)보다 큽니다.  \n"
            "→ Normal 생성 조건보다 큰 혈관 직경을 나타내지만, "
            "이 값 하나만으로 특정 상태를 결정할 수는 없습니다."
        )


    # ------------------------------------------
    # 2. 혈관 팽창률
    # Normal 생성 범위: 4 ~ 7 %
    # ------------------------------------------

    if distension_percent < 4.0:
        st.write(
            f"**혈관 팽창률: {distension_percent:.1f}%**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(4~7%)보다 낮습니다.  \n"
            "→ 박동에 따른 혈관 직경 변화가 상대적으로 작다는 의미이며, "
            "본 합성 데이터에서는 동맥 경직형과 유사한 방향의 특징입니다."
        )

    elif distension_percent <= 7.0:
        st.write(
            f"**혈관 팽창률: {distension_percent:.1f}%**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(4~7%)에 포함됩니다.  \n"
            "→ 혈관의 박동성 직경 변화가 Normal 생성 조건에서 "
            "크게 벗어나지 않습니다."
        )

    else:
        st.write(
            f"**혈관 팽창률: {distension_percent:.1f}%**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(4~7%)보다 높습니다.  \n"
            "→ Normal 생성 조건보다 혈관의 직경 변화가 큰 방향입니다."
        )


    # ------------------------------------------
    # 3. 수축기 혈압 SBP
    # Normal 생성 범위: 105 ~ 130 mmHg
    # ------------------------------------------

    if sbp < 105:
        st.write(
            f"**수축기 혈압 SBP: {sbp:.0f} mmHg**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(105~130 mmHg)보다 낮습니다.  \n"
            "→ Normal 합성 데이터보다 낮은 수축기 압력 조건입니다."
        )

    elif sbp <= 130:
        st.write(
            f"**수축기 혈압 SBP: {sbp:.0f} mmHg**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(105~130 mmHg)에 포함됩니다.  \n"
            "→ 수축기 혈압만 보면 Normal 생성 조건에서 크게 벗어나지 않습니다."
        )

    else:
        st.write(
            f"**수축기 혈압 SBP: {sbp:.0f} mmHg**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(105~130 mmHg)보다 높습니다.  \n"
            "→ 본 합성 데이터에서는 고혈압형과의 유사도를 높일 수 있는 방향의 값입니다."
        )


    # ------------------------------------------
    # 4. 평균 혈류속도
    # v_mean은 직접 범위를 지정해 생성한 변수가 아님
    # ------------------------------------------

    st.write(
        f"**평균 혈류속도: {v_mean:.2f} m/s**  \n"
        "→ 평균 혈류속도는 합성 데이터 생성 과정에서 혈류속도 파형으로부터 "
        "계산된 값이므로, D₀·SBP·DBP처럼 별도의 Normal 기준 범위를 "
        "직접 설정하지 않았습니다.  \n"
        "→ 모델에서는 혈관 직경 등 다른 변수와 함께 혈류 상태를 구분하는 "
        "정보로 사용됩니다."
    )


    # ------------------------------------------
    # 5. 이완기 혈압 DBP
    # Normal 생성 범위: 65 ~ 85 mmHg
    # ------------------------------------------

    if dbp < 65:
        st.write(
            f"**이완기 혈압 DBP: {dbp:.0f} mmHg**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(65~85 mmHg)보다 낮습니다.  \n"
            "→ Normal 합성 데이터보다 낮은 이완기 압력 조건입니다."
        )

    elif dbp <= 85:
        st.write(
            f"**이완기 혈압 DBP: {dbp:.0f} mmHg**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(65~85 mmHg)에 포함됩니다.  \n"
            "→ 이완기 혈압만 보면 Normal 생성 조건에서 크게 벗어나지 않습니다."
        )

    else:
        st.write(
            f"**이완기 혈압 DBP: {dbp:.0f} mmHg**  \n"
            "→ 본 시뮬레이션의 Normal 생성 범위(65~85 mmHg)보다 높습니다.  \n"
            "→ 본 합성 데이터에서는 고혈압형과의 유사도를 높일 수 있는 방향의 값입니다."
        )


    # ------------------------------------------
    # 종합 안내
    # ------------------------------------------

    st.info(
        "💡 모델은 위의 값을 하나씩 독립적으로 판정하여 상태를 결정하는 것이 아니라, "
        "5개 입력값의 조합을 Random Forest가 학습한 합성 데이터 패턴과 비교하여 "
        "최종 상태를 분류합니다."
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

st.divider()
st.header("🧭 프로그램 개발 과정")

st.caption(
    "이 프로그램은 혈류를 수학적으로 모델링하는 탐구에서 시작해 "
    "가상 데이터 생성, 머신러닝 학습, 입력 변수 축소 과정을 거쳐 제작되었습니다."
)

with st.expander("1단계. 혈관을 전기회로로 모델링"):
    st.markdown("""
    혈관계와 전기회로의 유사성에서 탐구를 시작했습니다.

    - 혈압 P ↔ 전압 V
    - 혈류량 Q ↔ 전류 I
    - 혈관 저항 R ↔ 전기저항 R
    - 혈관 순응도 C ↔ 커패시턴스 C

    이를 통해 혈관의 저항성과 탄성적 특성을 RC 회로로 단순화하여 표현했습니다.
    """)

with st.expander("2단계. 시간에 따른 혈류 파형 구현"):
    st.markdown(r"""
    혈압, 혈관 직경, 혈류속도를 시간에 따른 함수로 표현했습니다.

    \[
    P(t), \quad D(t), \quad v(t)
    \]

    혈관 단면적은

    \[
    A(t)=\pi\left(\frac{D(t)}{2}\right)^2
    \]

    이고, 혈류량은

    \[
    Q(t)=A(t)v(t)
    \]

    로 계산했습니다.

    초기에는 단순한 대칭 파형을 사용했지만, 박동성 혈류의 특징을 더 잘 표현하기 위해
    비대칭 파형으로 수정했습니다.
    """)

with st.expander("3단계. RC 모델을 이용한 혈류 분석"):
    st.markdown(r"""
    생성된 혈압 파형과 혈류 파형을 이용하여 각 가상 환자의 혈류 특성을 분석했습니다.

    RC 모델을 통해 각 환자의

    - 추정 혈관 저항 \(R_{est}\)
    - 추정 혈관 순응도 \(C_{est}\)

    를 계산했습니다.

    즉 단순한 혈압과 혈류 그래프를
    '저항과 순응도'라는 수치적 특성으로 변환했습니다.
    """)

with st.expander("4단계. 가상 환자 2,000명 생성"):
    st.markdown("""
    실제 대규모 임상 데이터를 확보할 수 없었기 때문에
    혈류역학적 특징을 반영한 합성 데이터를 생성했습니다.

    - Normal: 500명
    - Hypertension: 500명
    - Arterial Stiffness: 500명
    - Stenosis: 500명

    총 2,000명의 가상 환자 데이터셋을 만들었습니다.

    ※ 이 데이터는 실제 환자 기록이 아니라 탐구를 위해 생성한 합성 데이터입니다.
    """)

with st.expander("5단계. Random Forest 학습"):
    st.markdown("""
    각 가상 환자의 혈압, 혈관 직경, 혈류속도, 혈류량,
    RC 모델에서 추정한 R과 C 등의 정보를 이용해
    Random Forest 분류 모델을 학습했습니다.

    초기 모델에서는 총 12개의 입력 변수를 사용했습니다.
    """)

with st.expander("6단계. 입력 변수 축소"):
    st.markdown("""
    실제 활용 가능성을 높이기 위해
    '모든 정보를 입력하지 않아도 상태를 구분할 수 있는가?'를 확인했습니다.

    변수 중요도와 분류 성능을 비교하면서 입력 수를 줄였고,
    최종적으로 다음 5개 변수를 선택했습니다.

    1. 총경동맥 기준 직경 D₀
    2. 혈관 팽창률
    3. 수축기 혈압 SBP
    4. 평균 혈류속도
    5. 이완기 혈압 DBP
    """)

with st.expander("7단계. 최종 예측 프로그램 구현"):
    st.markdown("""
    사용자가 5개의 측정값을 입력하면
    학습된 Random Forest 모델이 네 가지 가상 혈류 상태 중
    가장 유사한 상태를 분류합니다.

    또한 각 상태의 예측 확률을 함께 출력하여
    모델의 판단 결과를 비교할 수 있도록 구성했습니다.
    """)

with st.expander("8단계. 웹사이트로 구현"):
    st.markdown("""
    최종 모델을 Streamlit 웹사이트에 연결하여
    사용자가 직접 값을 입력하고 결과를 확인할 수 있도록 구현했습니다.

    현재 사이트에서는

    5개 값 입력
    → 상태 예측
    → 상태별 확률
    → 입력값별 해석

    순서로 결과를 제공합니다.
    """)
