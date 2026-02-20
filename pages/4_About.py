import streamlit as st


st.set_page_config(
    page_title="VentureGuard AI – About",
    layout="wide",
)


def main():
    st.markdown("### About VentureGuard AI")
    st.markdown(
        """
        VentureGuard AI is a lightweight venture risk intelligence layer that turns basic funding data  
        into a consistent view of failure risk for startup investments.
        """
    )

    st.markdown("#### Product positioning")
    st.markdown(
        """
        - **Audience**: venture investors, LPs, and founders who want a structured view on downside risk.  
        - **Use cases**:
            - Triage inbound deal flow based on funding trajectory.  
            - Stress-test existing portfolios for undercapitalised companies.  
            - Provide a common quantitative reference point in investment committee discussions.  
        """
    )

    st.markdown("#### Technical stack")
    st.markdown(
        """
        - **Frontend / delivery**: Streamlit multi-page application.  
        - **Model**: scikit-learn `RandomForestClassifier` trained on historical startup outcome data.  
        - **Data**: reference dataset of global startups with funding history and final status labels.  
        - **Infrastructure**: designed to run as a simple container or Streamlit app service without external dependencies.  
        """
    )

    st.markdown("#### Methodology in brief")
    st.markdown(
        """
        1. Derive a binary **success** label from company status (operating / acquired vs everything else).  
        2. Extract two numeric features per company: `funding_total_usd` and `funding_rounds`.  
        3. Clean the data by converting funding to numeric values and imputing missing values with medians.  
        4. Train a Random Forest classifier to predict success from these features.  
        5. At inference, convert the model's success probability into a **failure probability** and risk score.  
        """
    )

    st.markdown("#### Important caveats")
    st.markdown(
        """
        - This is a **narrow model**: it looks only at capital raised and number of rounds.  
        - It should never be used as a sole decision-maker for capital allocation.  
        - Base rates in the reference dataset may differ from the specific market, stage, or geography of interest.  
        - The model is not calibrated for expected return; it addresses **downside risk**, not upside potential.  
        """
    )

    st.markdown("#### Next steps and extensions")
    st.markdown(
        """
        - Extend the feature set to include timing between rounds, investor quality, and market-level signals.  
        - Calibrate scores against realised fund outcomes (TVPI / DPI) rather than company-level survival alone.  
        - Wrap the model in an API and integrate with CRM or portfolio management systems.  
        """
    )


if __name__ == "__main__":
    main()

