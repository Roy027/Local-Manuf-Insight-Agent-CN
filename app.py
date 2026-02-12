import os
from datetime import datetime
from typing import List, Optional
import sys  # Add this import

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from agents.insight_agent import generate_insights
from agents.knowledge_agent import retrieve_knowledge
from agents.report_agent import generate_report
from core.config import get_api_key, get_client
from tools.data_analysis import analyze_dataset, SAMPLE_DATA_CSV

st.set_page_config(
    page_title="Manufacturing Insight Copilot",
    page_icon="🏭",
    layout="wide",
)


def load_uploaded_file(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def load_sample_df() -> pd.DataFrame:
    from io import StringIO

    return pd.read_csv(StringIO(SAMPLE_DATA_CSV))


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name.startswith("category")]


def render_data_profiling_tab(container, summary, df):
    with container:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", summary.n_rows)
        c2.metric("Columns", summary.n_cols)
        c3.metric("Flagged Rows", summary.anomalies.total_flagged_rows)

        schema_rows = []
        for name, schema in summary.schema.items():
            schema_rows.append({"column": name, "dtype": schema.dtype, "role": schema.role})
        with st.expander("Schema & Roles", expanded=False):
            st.dataframe(pd.DataFrame(schema_rows))


def render_distribution_tab(container, df, numeric_cols):
    with container:
        if not numeric_cols:
            st.info("No numeric columns available for distribution analysis.")
            return
        st.subheader("Histograms")
        for col in numeric_cols[:6]:
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)


def render_correlation_tab(container, df, numeric_cols):
    with container:
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns for correlation heatmap.")
            return
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Matrix", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)


def render_time_series_tab(container, df, summary, numeric_cols):
    with container:
        time_profile = summary.time_profiles
        index_col = None
        if time_profile and time_profile.index_column in df.columns:
            index_col = time_profile.index_column
        elif "id" in df.columns:
            index_col = "id"
        if not index_col:
            st.info("No index/time column detected.")
            return
        st.subheader(f"Trends Over {index_col}")
        for col in numeric_cols[:4]:
            fig = px.line(df, x=index_col, y=col, title=f"{col} over {index_col}")
            st.plotly_chart(fig, use_container_width=True)


def render_anomaly_tab(container, df, numeric_cols):
    with container:
        if not numeric_cols:
            st.info("No numeric columns for anomaly detection.")
            return
        std = df[numeric_cols].std(ddof=0).replace(0, np.nan)
        zscores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / std).fillna(0)
        anomaly_mask = (zscores > 3).any(axis=1)
        anomalies = df[anomaly_mask]
        st.metric("Flagged Rows (>3σ)", len(anomalies))
        if anomalies.empty:
            st.success("No significant anomalies detected.")
            return
        st.dataframe(anomalies.head(100))
        color_labels = anomaly_mask.map({True: "Anomaly", False: "Normal"})
        x_col = numeric_cols[0]
        y_col = numeric_cols[1] if len(numeric_cols) > 1 else None
        if y_col:
            plot_df = df[[x_col, y_col]].copy()
            plot_df["Status"] = color_labels
            fig = px.scatter(plot_df, x=x_col, y=y_col, color="Status", title="Anomaly Highlight")
        else:
            plot_df = df[[x_col]].copy().reset_index(drop=False)
            plot_df["Status"] = color_labels.reset_index(drop=True)
            fig = px.scatter(plot_df, x="index", y=x_col, color="Status", title="Anomaly Highlight")
        st.plotly_chart(fig, use_container_width=True)


def render_batch_variation_tab(container, df, numeric_cols):
    with container:
        keywords = ("batch", "line", "machine", "tool", "equip", "shift")
        categorical_cols = _categorical_columns(df)
        candidates = [c for c in categorical_cols if any(k in c.lower() for k in keywords) and df[c].nunique() <= 50]
        if not candidates:
            st.info("No batch/equipment style columns detected.")
            return
        metric_col = numeric_cols[0] if numeric_cols else None
        for cat in candidates[:3]:
            if metric_col:
                fig = px.box(df, x=cat, y=metric_col, title=f"{metric_col} variation by {cat}")
                st.plotly_chart(fig, use_container_width=True)
            freq = df[cat].value_counts().reset_index()
            freq.columns = [cat, "count"]
            fig_freq = px.bar(freq, x=cat, y="count", title=f"{cat} distribution")
            st.plotly_chart(fig_freq, use_container_width=True)


def render_yield_tab(container, df):
    with container:
        target_cols = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ("yield", "defect", "quality", "ppm", "scrap"))
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not target_cols:
            st.info("No yield/quality columns detected.")
            return
        for col in target_cols[:3]:
            temp = df.reset_index().rename(columns={"index": "Sequence"})
            fig = px.line(temp, x="Sequence", y=col, title=f"{col} trend")
            st.plotly_chart(fig, use_container_width=True)
            st.metric(f"{col} mean", f"{df[col].mean():.2f}")


def render_dimensionality_tab(container, df, numeric_cols):
    with container:
        if len(numeric_cols) < 3:
            st.info("Need at least three numeric columns for PCA.")
            return
        matrix = df[numeric_cols].dropna()
        if matrix.empty:
            st.info("Not enough data for PCA.")
            return
        centered = matrix - matrix.mean()
        arr = centered.to_numpy()
        arr = np.nan_to_num(arr)
        u, s, vt = np.linalg.svd(arr, full_matrices=False)
        coords = arr @ vt[:2].T
        comp_df = pd.DataFrame({"Component 1": coords[:, 0], "Component 2": coords[:, 1]})
        color_col = None
        cats = _categorical_columns(df)
        if cats:
            color_col = cats[0]
            comp_df[color_col] = matrix.index.map(lambda idx: df.loc[idx, color_col])
        fig = px.scatter(comp_df, x="Component 1", y="Component 2", color=color_col, title="PCA Overview")
        st.plotly_chart(fig, use_container_width=True)


def render_summary_tab(container, summary):
    with container:
        st.subheader("Recommended Dashboard Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", summary.n_rows)
        c2.metric("Anomalies", summary.anomalies.total_flagged_rows)
        c3.metric("Correlations Tracked", len(summary.top_correlations))
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if summary.top_correlations:
            st.write("Key Correlations to Monitor:")
            for corr in summary.top_correlations[:3]:
                st.write(f"- {corr.pair[0]} vs {corr.pair[1]}: {corr.pearson:.2f}")
def main():
    st.title("Manufacturing Insight Copilot")
    st.caption(
        "Upload production batches, profile the data locally, then let Gemini generate insights, "
        "link SOP knowledge, and produce technical/executive reports."
    )

    # --- API Key ---
    api_key = st.text_input(
        "Google API Key (kept local; required for Gemini calls)",
        value=get_api_key(allow_missing=True),
        type="password",
        placeholder="Enter your Google API Key",
    )
    client = None
    if not api_key:
        st.warning("Enter a Google API Key to enable Gemini calls.", icon="🔑")
    else:
        try:
            client = get_client(api_key)
        except Exception as exc:
            st.error(f"Failed to initialize Gemini client: {exc}")
            return
    # --- File Upload / Sample Data ---
    st.sidebar.header("Data Source")
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    sample_btn = st.sidebar.button("Load Sample Batch Data")

    df: Optional[pd.DataFrame] = None
    file_name: str = ""
    if uploaded:
        df = load_uploaded_file(uploaded)
        file_name = uploaded.name
    elif sample_btn:
        df = load_sample_df()
        file_name = "synthetic_production_data.csv"

    if df is None:
        st.info("Upload a dataset or click 'Load Sample Batch Data' to begin.")
        return

    # --- Local Profiling ---
    try:
        summary = analyze_dataset(df, file_name)
    except Exception as exc:  # pragma: no cover - surfaced to UI
        st.error(f"Data processing failed: {exc}")
        return

    with st.expander("Sample Rows", expanded=False):
        st.dataframe(pd.DataFrame(summary.sample_rows))

    df_viz = pd.DataFrame(summary.raw_data_subset)
    if not df_viz.empty:
        numeric_cols = _numeric_columns(df_viz)
        tab_labels = [
            "Data Profiling",
            "Distributions",
            "Correlations",
            "Time Trends",
            "Anomalies",
            "Batch/Eqp Variation",
            "Yield / Quality",
            "Dimensionality Reduction",
            "Summary Dashboard",
        ]
        tabs = st.tabs(tab_labels)
        render_data_profiling_tab(tabs[0], summary, df_viz)
        render_distribution_tab(tabs[1], df_viz, numeric_cols)
        render_correlation_tab(tabs[2], df_viz, numeric_cols)
        render_time_series_tab(tabs[3], df_viz, summary, numeric_cols)
        render_anomaly_tab(tabs[4], df_viz, numeric_cols)
        render_batch_variation_tab(tabs[5], df_viz, numeric_cols)
        render_yield_tab(tabs[6], df_viz)
        render_dimensionality_tab(tabs[7], df_viz, numeric_cols)
        render_summary_tab(tabs[8], summary)

    # --- Run Agents ---
    if st.button("Run Multi-Agent Analysis", disabled=client is None):
        with st.status("Running agents...", expanded=True) as status:
            try:
                st.write("1) InsightAgent: generating technical insights with pruned profile...")
                insights = generate_insights(client, summary)
                st.write(insights)

                st.write("2) KnowledgeAgent: mapping insights to SOPs and history...")
                knowledge = retrieve_knowledge(client, insights)
                st.write(knowledge)

                st.write("3) ReportAgent: synthesizing reports...")
                report = generate_report(client, summary, insights, knowledge)
                status.update(label="Agents completed", state="complete")
            except Exception as exc:  # pragma: no cover
                status.update(label="Agent run failed", state="error")
                st.error(f"Pipeline failed: {exc}")
                return

        st.success("Reports generated")
        st.markdown("### Executive Summary")
        st.markdown(report.executiveSummary)

        st.markdown("### Technical Report")
        st.markdown(report.technicalReport)

        st.markdown("### Detected Anomalies")
        st.write(report.anomalies)
    else:
        st.info("Press 'Run Multi-Agent Analysis' to call Gemini.")


if __name__ == "__main__":
    main()
