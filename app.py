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
from core.config import get_api_key, get_llm_client
from tools.data_analysis import analyze_dataset, SAMPLE_DATA_CSV

st.set_page_config(
    page_title="制造产线数据分析助手",
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
        c1.metric("行数", summary.n_rows)
        c2.metric("列数", summary.n_cols)
        c3.metric("标记异常行", summary.anomalies.total_flagged_rows)

        schema_rows = []
        for name, schema in summary.schema.items():
            schema_rows.append({"列名": name, "类型": schema.dtype, "角色": schema.role})
        with st.expander("数据模式与角色 (Schema & Roles)", expanded=False):
            st.dataframe(pd.DataFrame(schema_rows))


def render_distribution_tab(container, df, numeric_cols):
    with container:
        if not numeric_cols:
            st.info("没有可用的数值列进行分布分析。")
            return
        st.subheader("直方图 (Histograms)")
        for col in numeric_cols[:6]:
            fig = px.histogram(df, x=col, nbins=30, title=f"{col} 分布")
            st.plotly_chart(fig, use_container_width=True)


def render_correlation_tab(container, df, numeric_cols):
    with container:
        if len(numeric_cols) < 2:
            st.info("需要至少两个数值列来生成相关性热力图。")
            return
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="相关性矩阵", aspect="auto", color_continuous_scale="RdBu_r")
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
            st.info("未检测到索引或时间列。")
            return
        st.subheader(f"{index_col} 随时间趋势")
        for col in numeric_cols[:4]:
            fig = px.line(df, x=index_col, y=col, title=f"{col} 随 {index_col} 变化")
            st.plotly_chart(fig, use_container_width=True)


def render_anomaly_tab(container, df, numeric_cols):
    with container:
        if not numeric_cols:
            st.info("没有用于异常检测的数值列。")
            return
        st.subheader("异常行标记 (>3σ)")
        std = df[numeric_cols].std(ddof=0).replace(0, np.nan)
        zscores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / std).fillna(0)
        anomaly_mask = (zscores > 3).any(axis=1)
        anomalies = df[anomaly_mask]
        st.metric("异常总数", len(anomalies))
        if anomalies.empty:
            st.success("未发现显著异常。")
            return
        st.dataframe(anomalies.head(100))
        color_labels = anomaly_mask.map({True: "异常", False: "正常"})
        x_col = numeric_cols[0]
        y_col = numeric_cols[1] if len(numeric_cols) > 1 else None
        if y_col:
            plot_df = df[[x_col, y_col]].copy()
            plot_df["状态"] = color_labels
            fig = px.scatter(plot_df, x=x_col, y=y_col, color="状态", title="异常高亮显示")
        else:
            plot_df = df[[x_col]].copy().reset_index(drop=False)
            plot_df["状态"] = color_labels.reset_index(drop=True)
            fig = px.scatter(plot_df, x="index", y=x_col, color="状态", title="异常高亮显示")
        st.plotly_chart(fig, use_container_width=True)


def render_batch_variation_tab(container, df, numeric_cols):
    with container:
        keywords = ("batch", "line", "machine", "tool", "equip", "shift")
        categorical_cols = _categorical_columns(df)
        candidates = [c for c in categorical_cols if any(k in c.lower() for k in keywords) and df[c].nunique() <= 50]
        if not candidates:
            st.info("未检测到批次或设备相关的列。")
            return
        metric_col = numeric_cols[0] if numeric_cols else None
        for cat in candidates[:3]:
            if metric_col:
                fig = px.box(df, x=cat, y=metric_col, title=f"{metric_col} 按 {cat} 分布")
                st.plotly_chart(fig, use_container_width=True)
            freq = df[cat].value_counts().reset_index()
            freq.columns = [cat, "计数"]
            fig_freq = px.bar(freq, x=cat, y="计数", title=f"{cat} 分布")
            st.plotly_chart(fig_freq, use_container_width=True)


def render_yield_tab(container, df):
    with container:
        target_cols = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ("yield", "defect", "quality", "ppm", "scrap", "良率", "缺陷"))
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not target_cols:
            st.info("未检测到良率/质量相关的列。")
            return
        for col in target_cols[:3]:
            temp = df.reset_index().rename(columns={"index": "序号"})
            fig = px.line(temp, x="序号", y=col, title=f"{col} 趋势")
            st.plotly_chart(fig, use_container_width=True)
            st.metric(f"{col} 均值", f"{df[col].mean():.2f}")


def render_dimensionality_tab(container, df, numeric_cols):
    with container:
        if len(numeric_cols) < 3:
            st.info("需要至少三个数值列进行 PCA 分析。")
            return
        matrix = df[numeric_cols].dropna()
        if matrix.empty:
            st.info("数据不足，无法进行 PCA。")
            return
        centered = matrix - matrix.mean()
        arr = centered.to_numpy()
        arr = np.nan_to_num(arr)
        u, s, vt = np.linalg.svd(arr, full_matrices=False)
        coords = arr @ vt[:2].T
        comp_df = pd.DataFrame({"主成分 1": coords[:, 0], "主成分 2": coords[:, 1]})
        color_col = None
        cats = _categorical_columns(df)
        if cats:
            color_col = cats[0]
            comp_df[color_col] = matrix.index.map(lambda idx: df.loc[idx, color_col])
        fig = px.scatter(comp_df, x="主成分 1", y="主成分 2", color=color_col, title="PCA 降维概览")
        st.plotly_chart(fig, use_container_width=True)


def render_summary_tab(container, summary):
    with container:
        st.subheader("仪表盘快照建议")
        c1, c2, c3 = st.columns(3)
        c1.metric("总行数", summary.n_rows)
        c2.metric("异常数", summary.anomalies.total_flagged_rows)
        c3.metric("关键相关性", len(summary.top_correlations))
        st.caption(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if summary.top_correlations:
            st.write("需要关注的强相关关系:")
            for corr in summary.top_correlations[:3]:
                st.write(f"- {corr.pair[0]} vs {corr.pair[1]}: {corr.pearson:.2f}")

def main():
    st.title("制造产线数据分析助手 (CN Local)")
    st.caption(
        "本地数据安全分析 + 本地 AI 智能体 (Ollama) 或 Gemini。"
    )

    # --- LLM Configuration (Sidebar) ---
    st.sidebar.header("模型设置 (LLM Settings)")
    llm_provider = st.sidebar.selectbox("模型提供商", ["Ollama", "Gemini"])
    
    api_key = ""
    ollama_base_url = "http://localhost:11434/v1"
    ollama_model = "qwen3:8b"

    if llm_provider == "Gemini":
        api_key = st.sidebar.text_input(
            "Google API Key",
            value=get_api_key(allow_missing=True),
            type="password",
        )
    else:
        ollama_base_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434/v1")
        ollama_model = st.sidebar.text_input("Ollama Model", value="qwen3:8b")

    client = None
    if llm_provider == "Gemini" and not api_key:
        st.warning("请在侧边栏输入 Google API Key 以启用 AI 功能。", icon="🔑")
    else:
        try:
            client = get_llm_client(
                provider=llm_provider,
                api_key=api_key,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model
            )
        except Exception as exc:
            st.error(f"LLM 客户端初始化失败: {exc}")

    # --- File Upload / Sample Data ---
    st.sidebar.header("数据源")
    uploaded = st.sidebar.file_uploader("上传 CSV 或 Excel", type=["csv", "xlsx", "xls"])
    sample_btn = st.sidebar.button("加载示例批次数据")

    df: Optional[pd.DataFrame] = None
    file_name: str = ""
    if uploaded:
        df = load_uploaded_file(uploaded)
        file_name = uploaded.name
    elif sample_btn:
        df = load_sample_df()
        file_name = "synthetic_production_data.csv"

    if df is None:
        st.info("请上传数据集或点击 '加载示例批次数据' 开始分析。")
        return

    # --- Local Profiling ---
    try:
        summary = analyze_dataset(df, file_name)
    except Exception as exc:  # pragma: no cover - surfaced to UI
        st.error(f"数据处理失败: {exc}")
        return

    with st.expander("数据预览 (Sample Rows)", expanded=False):
        st.dataframe(pd.DataFrame(summary.sample_rows))

    df_viz = pd.DataFrame(summary.raw_data_subset)
    if not df_viz.empty:
        numeric_cols = _numeric_columns(df_viz)
        tab_labels = [
            "数据概览",
            "分布直方图",
            "相关性分析",
            "时间趋势",
            "异常检测",
            "批次/设备差异",
            "良率/质量",
            "降维分析",
            "汇总仪表盘",
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
    if st.button("运行多智能体分析 (Run Multi-Agent Analysis)", disabled=client is None):
        with st.status("正在运行智能体...", expanded=True) as status:
            try:
                st.write("1) 洞察智能体 (InsightAgent): 分析技术指标...")
                insights = generate_insights(client, summary)
                st.write(insights)

                st.write("2) 知识智能体 (KnowledgeAgent): 检索 SOP 和历史案例...")
                knowledge = retrieve_knowledge(client, insights)
                st.write(knowledge)

                st.write("3) 报告智能体 (ReportAgent): 生成最终报告...")
                report = generate_report(client, summary, insights, knowledge)
                status.update(label="智能体分析完成", state="complete")
            except Exception as exc:  # pragma: no cover
                status.update(label="智能体运行失败", state="error")
                st.error(f"分析流程出错: {exc}")
                return

        st.success("报告已生成")
        st.markdown("### 管理摘要 (Executive Summary)")
        st.markdown(report.executiveSummary)

        st.markdown("### 技术分析报告 (Technical Report)")
        st.markdown(report.technicalReport)

        st.markdown("### 检测到的异常 (Detected Anomalies)")
        st.write(report.anomalies)
    else:
        st.info("点击 '运行多智能体分析' 按钮开始调用 AI 模型。")


if __name__ == "__main__":
    main()
