import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.helpers import load_jsonl, project_path


DEFAULT_RESULTS_DIR = "experiments/baselines"

REQUIRED_COLUMNS = [
    "sample_id",
    "model_name",
    "source_text",
    "reference_text",
    "prediction_text",
    "target_sentiment",
]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid #374151;
            border-radius: 18px;
            padding: 26px;
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            color: #d1d5db;
            font-size: 15px;
            line-height: 1.7;
        }

        .info-box {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 14px 16px;
            color: #cbd5e1;
            margin-bottom: 22px;
            line-height: 1.7;
        }

        .comparison-card {
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 16px;
            min-height: 170px;
            background: #111827;
            line-height: 1.8;
        }

        .card-title {
            color: #9ca3af;
            font-size: 14px;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .card-text {
            color: #f9fafb;
            font-size: 15px;
            line-height: 1.8;
        }

        textarea, input {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">📊 Model Results Viewer</div>
            <div class="hero-subtitle">
                Select one model result file and review its original text, reference text,
                and generated prediction in a clean comparison table.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_prediction_files() -> list[Path]:
    results_dir = project_path(DEFAULT_RESULTS_DIR)

    if not results_dir.exists():
        return []

    return sorted(results_dir.glob("*.jsonl"))


def load_records_from_path(file_path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(file_path)

    for record in records:
        record["source_file"] = file_path.name

    return records


def build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(records)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    if "source_file" not in df.columns:
        df["source_file"] = ""

    display_columns = [
        "model_name",
        "sample_id",
        "target_sentiment",
        "source_text",
        "reference_text",
        "prediction_text",
        "source_file",
    ]

    return df[display_columns]


def render_file_selector(prediction_files: list[Path]) -> Path:
    st.sidebar.header("Model selection")

    selected_file = st.sidebar.selectbox(
        "Choose model predictions",
        options=prediction_files,
        format_func=lambda path: path.name,
    )

    return selected_file


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    sentiments = sorted(df["target_sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Target sentiment",
        options=sentiments,
        default=sentiments,
    )

    search_query = st.sidebar.text_input(
        "Search",
        placeholder="Search original, reference, or prediction...",
    )

    filtered_df = df.copy()

    if selected_sentiments:
        filtered_df = filtered_df[
            filtered_df["target_sentiment"].isin(selected_sentiments)
        ]

    if search_query.strip():
        query = search_query.strip()

        mask = (
            filtered_df["source_text"].astype(str).str.contains(query, case=False, na=False)
            | filtered_df["reference_text"].astype(str).str.contains(query, case=False, na=False)
            | filtered_df["prediction_text"].astype(str).str.contains(query, case=False, na=False)
        )

        filtered_df = filtered_df[mask]

    return filtered_df


def render_metrics(df: pd.DataFrame, selected_file: Path) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected file", selected_file.name)

    with col2:
        st.metric("Predictions", len(df))

    with col3:
        st.metric("Target sentiments", df["target_sentiment"].nunique() if not df.empty else 0)


def render_results_table(df: pd.DataFrame) -> None:
    table_df = df.rename(
        columns={
            "model_name": "Model",
            "sample_id": "Sample ID",
            "target_sentiment": "Target Sentiment",
            "source_text": "Original Text",
            "reference_text": "Reference Text",
            "prediction_text": "Model Prediction",
            "source_file": "Source File",
        }
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Original Text": st.column_config.TextColumn(width="large"),
            "Reference Text": st.column_config.TextColumn(width="large"),
            "Model Prediction": st.column_config.TextColumn(width="large"),
            "Source File": st.column_config.TextColumn(width="small"),
        },
    )


def render_example_inspector(df: pd.DataFrame) -> None:
    if df.empty:
        return

    st.subheader("Example Inspector")

    sample_options = df["sample_id"].astype(str).unique().tolist()

    selected_sample_id = st.selectbox(
        "Select sample",
        options=sample_options,
    )

    selected_row = df[df["sample_id"].astype(str) == selected_sample_id].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="card-title">Original Text</div>
                <div class="card-text">{selected_row["source_text"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="card-title">Reference Text</div>
                <div class="card-text">{selected_row["reference_text"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="card-title">Model Prediction</div>
                <div class="card-text">{selected_row["prediction_text"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"Model: {selected_row['model_name']} | "
        f"Target sentiment: {selected_row['target_sentiment']}"
    )


def main() -> None:
    st.set_page_config(
        page_title="Model Results Viewer",
        page_icon="📊",
        layout="wide",
    )

    inject_custom_css()
    render_header()

    prediction_files = find_prediction_files()

    if not prediction_files:
        st.warning("No prediction files found under experiments/baselines.")
        return

    selected_file = render_file_selector(prediction_files)

    records = load_records_from_path(selected_file)
    df = build_dataframe(records)

    filtered_df = apply_filters(df)

    render_metrics(filtered_df, selected_file)

    st.divider()

    st.subheader("Results Table")
    render_results_table(filtered_df)

    st.divider()

    render_example_inspector(filtered_df)


if __name__ == "__main__":
    main()