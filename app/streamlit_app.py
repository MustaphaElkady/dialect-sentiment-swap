import streamlit as st

from src.controllers.SwapController import SwapController
from src.models import SwapRequestModel
from src.models.enums import SentimentEnum


SENTIMENT_LABELS = {
    SentimentEnum.POSITIVE.value: "😊 Positive",
    SentimentEnum.NEUTRAL.value: "😐 Neutral",
    SentimentEnum.NEGATIVE.value: "😞 Negative",
}

EXAMPLES = {
    "Negative → Positive": {
        "text": "الخدمة كانت سيئة جدًا",
        "target": SentimentEnum.POSITIVE.value,
        "dialect": "Egyptian",
    },
    "Positive → Negative": {
        "text": "التجربة كانت ممتازة",
        "target": SentimentEnum.NEGATIVE.value,
        "dialect": "Egyptian",
    },
    "Emotional → Neutral": {
        "text": "الأكل كان رائع جدًا",
        "target": SentimentEnum.NEUTRAL.value,
        "dialect": "Egyptian",
    },
}


def inject_custom_css() -> None:
    """Inject custom CSS to improve layout, spacing, and Arabic text rendering."""
    st.markdown(
        """
        <style>
        .main {
            direction: rtl;
        }

        .block-container {
            max-width: 950px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        textarea, input {
            direction: rtl !important;
            text-align: right !important;
        }

        .hero-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid #374151;
            border-radius: 18px;
            padding: 28px;
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            color: #d1d5db;
            font-size: 16px;
            line-height: 1.8;
        }

        .info-box {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 18px;
            color: #cbd5e1;
        }

        .result-card {
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 18px;
            min-height: 150px;
            background: #111827;
        }

        .card-title {
            color: #9ca3af;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .sentence-text {
            font-size: 18px;
            line-height: 1.9;
            color: #f9fafb;
        }

        .success-badge {
            display: inline-block;
            background: #064e3b;
            color: #d1fae5;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
            margin-bottom: 16px;
        }

        .warning-badge {
            display: inline-block;
            background: #7c2d12;
            color: #ffedd5;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
            margin-bottom: 16px;
        }

        .small-note {
            color: #9ca3af;
            font-size: 13px;
            line-height: 1.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Initialize Streamlit session state with default demo values."""
    if "source_text" not in st.session_state:
        st.session_state.source_text = "الخدمة كانت سيئة جدًا"

    if "target_sentiment" not in st.session_state:
        st.session_state.target_sentiment = SentimentEnum.POSITIVE.value

    if "dialect" not in st.session_state:
        st.session_state.dialect = "Egyptian"


def apply_example(example_name: str) -> None:
    """Load a predefined example into the input form."""
    example = EXAMPLES[example_name]
    st.session_state.source_text = example["text"]
    st.session_state.target_sentiment = example["target"]
    st.session_state.dialect = example["dialect"]


def render_header() -> None:
    """Render the main page header."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🔁 Dialect Sentiment Swap</div>
            <div class="hero-subtitle">
                Enter an Arabic dialect sentence, choose a target sentiment,
                and generate a rewritten version that attempts to preserve the meaning
                while changing the sentiment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render sidebar controls and example shortcuts."""
    with st.sidebar:
        st.header("⚙️ Controls")

        st.markdown(
            """
            Current version: **Rule-Based Prototype**
            
            The goal of this stage is to verify the project pipeline:
            
            `UI → Controller → Service → Result`
            """
        )

        st.divider()

        st.subheader("Ready Examples")

        for example_name in EXAMPLES:
            if st.button(example_name, use_container_width=True):
                apply_example(example_name)

        st.divider()

        st.caption("Later, the rule-based service will be replaced by a real generation model.")


def render_input_form():
    """Render the user input form and return submitted values."""
    st.subheader("📝 Input")

    with st.form("swap_form"):
        source_text = st.text_area(
            "Original sentence",
            key="source_text",
            height=140,
            placeholder="Write an Arabic dialect sentence here...",
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            target_sentiment = st.selectbox(
                "Target sentiment",
                options=[
                    SentimentEnum.POSITIVE.value,
                    SentimentEnum.NEUTRAL.value,
                    SentimentEnum.NEGATIVE.value,
                ],
                format_func=lambda value: SENTIMENT_LABELS[value],
                key="target_sentiment",
            )

        with col2:
            dialect = st.text_input(
                "Dialect, optional",
                key="dialect",
                placeholder="Example: Egyptian, Saudi, Jordanian...",
            )

        submitted = st.form_submit_button(
            "Generate Swap",
            type="primary",
            use_container_width=True,
        )

    return submitted, source_text, target_sentiment, dialect


def render_result(result) -> None:
    """Render the swap result in a before/after layout."""
    st.subheader("✨ Result")

    if result.success:
        st.markdown(
            '<div class="success-badge">✅ Swap completed successfully</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="warning-badge">⚠️ The sentence did not change. The current rules may not cover this case.</div>',
            unsafe_allow_html=True,
        )

    before_col, after_col = st.columns(2)

    with before_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">Original Sentence</div>
                <div class="sentence-text">{result.source_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with after_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">Generated Sentence</div>
                <div class="sentence-text">{result.generated_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    meta_col1, meta_col2, meta_col3 = st.columns(3)

    with meta_col1:
        st.metric("Target Sentiment", result.target_sentiment.value)

    with meta_col2:
        st.metric("Dialect", result.dialect or "Not provided")

    with meta_col3:
        st.metric("Success", "Yes" if result.success else "No")

    if result.error_message:
        st.error(result.error_message)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Dialect Sentiment Swap",
        page_icon="🔁",
        layout="wide",
    )

    inject_custom_css()
    initialize_state()
    render_sidebar()
    render_header()

    st.markdown(
        """
        <div class="info-box">
            <b>Note:</b> This is not the final model version.
            The current interface is used to test communication between project layers
            before training a real generation model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    submitted, source_text, target_sentiment, dialect = render_input_form()

    if submitted:
        if not source_text.strip():
            st.warning("Please enter a sentence first.")
            return

        request = SwapRequestModel(
            source_text=source_text.strip(),
            target_sentiment=SentimentEnum(target_sentiment),
            dialect=dialect.strip() if dialect.strip() else None,
        )

        controller = SwapController()
        result = controller.swap(request)

        render_result(result)


if __name__ == "__main__":
    main()