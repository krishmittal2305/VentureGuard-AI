import streamlit as st
import pandas as pd
import altair as alt

from core import load_dataset


st.set_page_config(
    page_title="VentureGuard AI – Analytics",
    layout="wide",
)


def render_header():
    st.markdown("### Portfolio analytics")
    st.markdown(
        "Explore how funding amounts and round dynamics relate to observed startup outcomes."
    )


def funding_vs_outcome_chart(df: pd.DataFrame) -> alt.Chart:
    data = df.copy()
    data["success_label"] = data["success"].map(
        {1: "Operating / Acquired", 0: "Closed / Other"}
    )

    # Convert funding to numeric and millions for readability
    data["funding_total_usd"] = pd.to_numeric(
        data["funding_total_usd"].replace("-", pd.NA), errors="coerce"
    )
    data = data.dropna(subset=["funding_total_usd"])
    data["funding_musd"] = data["funding_total_usd"] / 1_000_000

    agg = (
        data.groupby("success_label")["funding_musd"]
        .median()
        .reset_index(name="median_funding_musd")
    )

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "success_label",
                title="Outcome",
            ),
            y=alt.Y(
                "median_funding_musd",
                title="Median total funding (USD, millions)",
            ),
            color=alt.Color("success_label", legend=None),
            tooltip=[
                alt.Tooltip("success_label", title="Outcome"),
                alt.Tooltip(
                    "median_funding_musd",
                    title="Median funding (M USD)",
                    format=".2f",
                ),
            ],
        )
    )

    return chart.properties(height=320)


def funding_rounds_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    data = df.copy()
    data["funding_rounds"] = pd.to_numeric(data["funding_rounds"], errors="coerce")
    data = data.dropna(subset=["funding_rounds"])

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "funding_rounds:Q",
                bin=alt.Bin(maxbins=15),
                title="Number of funding rounds",
            ),
            y=alt.Y("count()", title="Number of companies"),
            tooltip=[
                alt.Tooltip("count()", title="Companies"),
            ],
        )
    )

    return chart.properties(height=320)


def outcome_proportions_chart(df: pd.DataFrame) -> alt.Chart:
    data = df.copy()
    data["success_label"] = data["success"].map(
        {1: "Operating / Acquired", 0: "Closed / Other"}
    )

    agg = (
        data.groupby("success_label")
        .size()
        .reset_index(name="count")
        .assign(share=lambda d: d["count"] / d["count"].sum())
    )

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("success_label", title="Outcome"),
            y=alt.Y("share", title="Share of companies", axis=alt.Axis(format="%")),
            color=alt.Color("success_label", legend=None),
            tooltip=[
                alt.Tooltip("success_label", title="Outcome"),
                alt.Tooltip("count", title="Companies"),
                alt.Tooltip("share", title="Share", format=".1%"),
            ],
        )
    )

    return chart.properties(height=320)


def main():
    render_header()
    st.markdown("---")

    df, _, _ = load_dataset()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Funding versus outcome")
        st.altair_chart(funding_vs_outcome_chart(df), use_container_width=True)
        st.caption(
            "Median total funding is noticeably higher for companies that remain operating "
            "or are acquired compared to those that close or otherwise disappear."
        )

    with col2:
        st.markdown("#### Distribution of funding rounds")
        st.altair_chart(funding_rounds_distribution_chart(df), use_container_width=True)
        st.caption(
            "Most companies in the dataset raise only one or two material funding rounds; "
            "multi-round journeys are comparatively rare."
        )

    st.markdown("#### Outcome mix across the dataset")
    st.altair_chart(outcome_proportions_chart(df), use_container_width=True)
    st.caption(
        "This high-level base rate is important context: in most venture portfolios, a minority of "
        "companies ultimately sustain operations or achieve an exit."
    )


if __name__ == "__main__":
    main()

