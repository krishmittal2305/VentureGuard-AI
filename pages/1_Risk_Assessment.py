import streamlit as st

from core import (
    describe_relative_position,
    get_failure_probability,
    bucket_risk_level,
)


st.set_page_config(
    page_title="VentureGuard AI – Risk Assessment",
    layout="wide",
)


def render_header():
    st.markdown("### Risk Assessment")
    st.markdown(
        "Translate a startup's funding profile into an investor-style failure risk signal."
    )


def render_input_form():
    st.markdown("#### Startup funding profile")

    col1, col2 = st.columns(2)

    with col1:
        funding_musd = st.number_input(
            "Total funding raised to date (USD, in millions)",
            min_value=0.0,
            max_value=5_000.0,
            value=10.0,
            step=1.0,
            help="Aggregate primary capital raised across all equity and quasi-equity rounds.",
        )

    with col2:
        funding_rounds = st.number_input(
            "Number of funding rounds closed",
            min_value=1,
            max_value=25,
            value=2,
            step=1,
            help="Count of distinct priced or material rounds (e.g. Seed, Series A, Series B).",
        )

    submitted = st.button("Run VentureGuard risk assessment", type="primary")

    return submitted, funding_musd, int(funding_rounds)


def render_results(funding_musd: float, funding_rounds: int):
    total_funding_usd = funding_musd * 1_000_000

    with st.spinner("Scoring funding profile against historical outcomes..."):
        failure_prob, success_prob = get_failure_probability(
            total_funding_usd=total_funding_usd,
            funding_rounds=funding_rounds,
        )

    risk_score = round(failure_prob * 100, 1)
    label = bucket_risk_level(failure_prob)

    st.markdown("#### Summary assessment")

    col1, col2, col3 = st.columns(3)
    col1.metric("Failure probability", f"{failure_prob * 100:0.1f} %")
    col2.metric("VC-style risk score", f"{risk_score:0.1f} / 100")
    col3.metric("Risk tier", label)

    st.markdown("#### Interpretation")

    explanation = describe_relative_position(
        total_funding_usd=total_funding_usd,
        funding_rounds=funding_rounds,
    )

    st.write(
        f"Based on a total of **${total_funding_usd:,.0f}** raised across "
        f"**{funding_rounds} funding rounds**, VentureGuard AI estimates a "
        f"**{failure_prob * 100:0.1f}%** probability that this startup does **not** reach "
        "a sustained operating or acquired outcome."
    )

    st.write(explanation)

    with st.expander("How to use this signal in investment workflows"):
        st.markdown(
            """
            - Use the **risk score** as a *relative* rather than absolute input alongside qualitative work.  
            - Interpret **high risk** results as a prompt to double-click on capital efficiency, runway, and follow-on risk.  
            - Interpret **low risk** results as \"funding profile is typical for surviving companies\", **not** as a guarantee of success.  
            - The model currently only sees **funding_total_usd** and **funding_rounds** – it ignores team, product, market structure, and unit economics.
            """
        )


def main():
    render_header()
    st.markdown("---")
    submitted, funding_musd, funding_rounds = render_input_form()

    if submitted:
        render_results(funding_musd, funding_rounds)
    else:
        st.info(
            "Provide a funding profile above and run the assessment to generate a risk view."
        )


if __name__ == "__main__":
    main()

