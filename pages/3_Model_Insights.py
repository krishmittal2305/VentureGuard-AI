import streamlit as st
import pandas as pd
import altair as alt

from core import FEATURE_COLUMNS, load_dataset, load_model


st.set_page_config(
    page_title="VentureGuard AI – Model Insights",
    layout="wide",
)


def render_header():
    st.markdown("### Model insights")
    st.markdown(
        "Understand what VentureGuard AI sees, how it was trained, and where it should be used with caution."
    )


def render_model_overview():
    st.markdown("#### High-level design")
    st.markdown(
        """
        - **Model family**: Random Forest classifier from scikit-learn.  
        - **Target**: Binary label where companies that are *operating* or *acquired* are treated as successful,  
          and all other statuses are treated as failures.  
        - **Features used**:
            - `funding_total_usd` – total capital raised.  
            - `funding_rounds` – count of distinct funding rounds.  
        - **Handling imbalance**: The training notebook used SMOTE to balance classes before fitting.  

        The result is a model that focuses narrowly on **funding trajectory as a proxy for survivability**,  
        not on product-market fit, team calibre, or market structure.
        """
    )


def feature_importance_chart(importances: pd.DataFrame) -> alt.Chart:
    chart = (
        alt.Chart(importances)
        .mark_bar()
        .encode(
            x=alt.X("importance", title="Relative importance"),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=[
                alt.Tooltip("feature", title="Feature"),
                alt.Tooltip("importance", title="Importance", format=".3f"),
            ],
        )
    )
    return chart.properties(height=220)


def render_feature_importance():
    model = load_model()
    if not hasattr(model, "feature_importances_"):
        st.info(
            "This model type does not expose feature importances in a structured way."
        )
        return

    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    st.markdown("#### Feature importance")
    st.altair_chart(feature_importance_chart(importances), use_container_width=True)
    st.caption(
        "Higher values indicate that splits on the feature tend to reduce predictive uncertainty more. "
        "With only two numeric inputs, this view is intentionally simple."
    )


def render_limitations():
    st.markdown("#### Assumptions and limitations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **What the model captures**

            - Capital intensity and funding trajectory as observed in historical data.  
            - The fact that companies raising materially more capital and more rounds tend, on average,  
              to remain operating or achieve an exit more often.  
            - Non-linear relationships between funding amounts and observed outcomes via the random forest structure.  
            """
        )

    with col2:
        st.markdown(
            """
            **What the model does *not* see**

            - Team quality, founder-market fit, or investor syndicate strength.  
            - Market structure, competitive dynamics, or macro regime shifts.  
            - Unit economics, margins, retention, or customer quality.  
            - Round structure details (secondary, inside rounds, flat / down rounds, etc.).  
            """
        )

    with st.expander("How to safely use this model in an investment setting"):
        st.markdown(
            """
            - Treat VentureGuard AI as a **risk-sensing layer**, not an IC decision engine.  
            - Use it to **stress-test portfolios**: identify clusters of companies with thin funding profiles relative to peers.  
            - Combine the scores with **qualitative memos**, board insight, and market work before making allocation decisions.  
            - Periodically **re-benchmark** the model as new vintages of data become available.
            """
        )


def render_data_coverage():
    st.markdown("#### Data coverage snapshot")
    df, _, _ = load_dataset()

    by_country = (
        df["country_code"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "country_code", "country_code": "count"})
        .head(10)
    )

    chart = (
        alt.Chart(by_country)
        .mark_bar()
        .encode(
            x=alt.X("country_code", title="Country"),
            y=alt.Y("count", title="Companies"),
            tooltip=[
                alt.Tooltip("country_code", title="Country"),
                alt.Tooltip("count", title="Companies"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Understanding where the reference data comes from is important when applying the model "
        "to companies in underrepresented geographies."
    )


def main():
    render_header()
    st.markdown("---")
    render_model_overview()
    st.markdown("---")
    render_feature_importance()
    st.markdown("---")
    render_data_coverage()
    st.markdown("---")
    render_limitations()


if __name__ == "__main__":
    main()

