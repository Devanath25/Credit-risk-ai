from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "model.pkl"
    return joblib.load(model_path)


def get_factor_contributions(model, feature_values: pd.DataFrame) -> list[tuple[str, float]]:
    feature_names = list(feature_values.columns)
    values = feature_values.iloc[0].to_numpy(dtype=float)
    coefficients = model.coef_[0]
    contributions = values * coefficients

    ranking = sorted(
        zip(feature_names, contributions),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return ranking


def show_business_recommendation(prediction: int, probability: float) -> None:
    st.subheader("Business Decision Recommendation")

    if prediction == 1:
        st.warning(
            "Decision: Route this customer to proactive risk management."
        )
        st.write(
            "Based on the current risk signal, prioritize this account for preventive action instead of standard servicing."
        )
        st.markdown(
            "- Trigger payment reminders (SMS/email) before due dates.\n"
            "- Offer credit counseling or financial wellness support.\n"
            "- Apply closer monthly monitoring and early intervention checks."
        )
    else:
        st.success(
            "Decision: Keep this customer in the standard monitoring segment."
        )
        st.write(
            "Current profile indicates lower near-term risk, so continue routine portfolio management while tracking changes over time."
        )
        st.markdown(
            "- Continue normal monitoring cadence.\n"
            "- Keep standard engagement and support policies.\n"
            "- Reassess if utilization, missed payments, or debt ratio worsens."
        )

    st.caption(
        f"Decision support note: recommendation combines model output ({probability:.2%} delinquency probability) with a business action policy."
    )


def show_input_bar_chart(feature_values: pd.DataFrame) -> None:
    st.subheader("Input Overview (Bar Chart)")

    display_df = pd.DataFrame(
        {
            "Feature": ["Income", "Credit Utilization", "Missed Payments", "Debt to Income"],
            "Value": [
                feature_values.at[0, "income"],
                feature_values.at[0, "credit_utilization"],
                feature_values.at[0, "missed_payments"],
                feature_values.at[0, "debt_to_income"],
            ],
        }
    ).set_index("Feature")

    st.bar_chart(display_df)
    st.caption("This chart shows the raw input values used for the current prediction.")


def main() -> None:
    st.set_page_config(page_title="Credit Risk AI", page_icon="💳", layout="centered")
    st.title("💳 Credit Risk AI")
    st.write("Predict whether a customer is likely to become delinquent.")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Model file not found. Please run `python src/train.py` first.")
        return

    st.subheader("Customer Inputs")
    income = st.number_input("Income", min_value=30000, max_value=150000, value=65000, step=1000)
    credit_utilization = st.slider("Credit Utilization", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    missed_payments = st.slider("Missed Payments", min_value=0, max_value=10, value=1, step=1)
    debt_to_income = st.slider("Debt to Income", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    features = pd.DataFrame(
        [
            {
                "income": income,
                "credit_utilization": credit_utilization,
                "missed_payments": missed_payments,
                "debt_to_income": debt_to_income,
            }
        ]
    )

    show_input_bar_chart(features)

    if st.button("Predict Delinquency"):
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        color = "red" if prediction == 1 else "green"

        st.markdown(f"### Risk: :{color}[{risk_label}]")
        st.write(f"**Probability of Delinquency:** {probability:.2%}")

        st.subheader("Top Contributing Factors")
        contributions = get_factor_contributions(model, features)

        for feature, value in contributions[:3]:
            direction = "increases risk" if value > 0 else "decreases risk"
            st.write(f"- **{feature}**: {value:.4f} ({direction})")

        show_business_recommendation(prediction, probability)

        st.caption("Contributions are based on model coefficients x input values.")


if __name__ == "__main__":
    main()