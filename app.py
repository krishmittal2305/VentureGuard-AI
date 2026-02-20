import streamlit as st

from core import load_dataset


st.set_page_config(
    page_title="VentureGuard AI – Venture Risk Intelligence",
    layout="wide",
)


def render_header():
    st.title("VentureGuard AI")
    st.subheader("Data-driven startup failure risk intelligence for venture investors")


def render_intro():
    st.markdown("### Why VentureGuard AI")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **The problem**  
            Early-stage venture decisions are often made with incomplete information,  
            noisy benchmarks, and limited visibility into downside risk.

            Even simple questions – *\"How risky is this funding profile compared to the market?\"* –  
            usually require manual spreadsheet work and fragmented data.
            """
        )

    with col2:
        st.markdown(
            """
            **The approach**  
            VentureGuard AI combines historical startup funding data with a machine learning model  
            trained to estimate the probability that a company does **not** reach a sustainable,  
            operating or acquired outcome.

            The tool focuses on being:
            - **Fast** enough for live IC discussions  
            - **Transparent** about what the model sees and what it ignores  
            - **Consistent** across different deals and teams
            """
        )


def render_dataset_insights():
    st.markdown("### Portfolio-scale context from historical data")
    df, features, _ = load_dataset()

    total_companies = len(df)
    success_rate = df["success"].mean()
    median_funding = features["funding_total_usd"].median()
    median_rounds = features["funding_rounds"].median()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies in reference dataset", f"{total_companies:,}")
    col2.metric("Historical survival / exit rate", f"{success_rate * 100:0.1f}%")
    col3.metric("Median total funding (USD)", f"${median_funding:,.0f}")
    col4.metric("Median funding rounds", f"{median_rounds:0.1f}")

    with st.expander("How to interpret these numbers"):
        st.markdown(
            """
            - The **dataset** aggregates tens of thousands of historical startups with known outcomes.  
            - **Survival / exit rate** reflects the share of companies that remained operating or were acquired.  
            - **Median funding and rounds** are used as reference points when scoring an individual company.  

            VentureGuard AI does **not** attempt to predict valuation or return multiples – it focuses narrowly  
            on failure risk conditional on funding profile.
            """
        )


def render_navigation_help():
    st.markdown("### How to use VentureGuard AI")
    st.markdown(
        """
        - **Risk Assessment** – input a startup's current funding profile and get an investor-style risk view.  
        - **Analytics** – explore how funding amounts and rounds relate to outcomes across the dataset.  
        - **Model Insights** – understand how the model works and where it is reliable or weak.  
        - **About** – methodology, stack, and important caveats for using this tool in investment workflows.
        """
    )


def main():
    render_header()
    st.markdown("---")
    render_intro()
    st.markdown("---")
    render_dataset_insights()
    st.markdown("---")
    render_navigation_help()


if __name__ == "__main__":
    main()

