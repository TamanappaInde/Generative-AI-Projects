"""Streamlit UI for the Requirement Quality Analysis Agent."""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from analyzer import DEFAULT_EXPECTED_ROLES, AnalysisResult, analyze_requirements, synthesize_with_openai

load_dotenv()

SAMPLE_REQUIREMENTS = """BR-001: As a customer, the user must be able to reset a password quickly.
BR-002: As a customer, the user must be able to reset their password quickly.
BR-003: The platform should provide an easy dashboard for reporting.
BR-004: As a support agent, the system shall show a customer's order history.
"""

st.set_page_config(page_title="Requirement Quality Analyst", page_icon="RQ", layout="wide")


def _finding_rows(result: AnalysisResult) -> list[dict[str, object]]:
    return [
        {
            "ID": finding.finding_id,
            "Severity": finding.severity,
            "Category": finding.category,
            "Title": finding.title,
            "Requirements": ", ".join(finding.requirement_ids),
            "Recommendation": finding.recommendation,
        }
        for finding in result.findings
    ]


def _render_result(result: AnalysisResult) -> None:
    summary = result.summary
    metric_columns = st.columns(4)
    metric_columns[0].metric("Requirements", summary["requirement_count"])
    metric_columns[1].metric("Findings", summary["finding_count"])
    metric_columns[2].metric("High severity", summary["severity"].get("High", 0))
    metric_columns[3].metric("Retrieved evidence", summary["retrieval_count"])

    st.subheader("Quality findings")
    if result.findings:
        st.dataframe(_finding_rows(result), use_container_width=True, hide_index=True)
    else:
        st.success("No quality issues were detected by the configured checks.")

    with st.expander("Finding details", expanded=False):
        for finding in result.findings:
            st.markdown(f"**{finding.finding_id} | {finding.severity} | {finding.title}**")
            st.write(finding.description)
            if finding.requirement_ids:
                st.caption(f"Requirements: {', '.join(finding.requirement_ids)}")
            if finding.evidence:
                st.code("\n".join(finding.evidence))
            st.info(finding.recommendation)

    left, right = st.columns(2)
    with left:
        st.subheader("Role coverage")
        detected = summary["roles_detected"] or ["No explicit roles detected"]
        st.write(", ".join(detected))
        if summary["expected_roles"]:
            st.caption("Expected roles: " + ", ".join(summary["expected_roles"]))
    with right:
        st.subheader("Retrieved requirement context")
        for requirement in result.retrieved_context:
            st.markdown(f"**{requirement.requirement_id}** {requirement.text}")

    st.download_button(
        "Download JSON report",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="requirement-quality-report.json",
        mime="application/json",
    )


def main() -> None:
    st.title("Requirement Quality Analyst")
    st.write("Upload or paste business requirements to identify duplicates, missing roles, ambiguity, and missing acceptance criteria.")

    with st.sidebar:
        st.header("Review settings")
        api_key = st.text_input(
            "OpenAI API key (optional)",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Local checks and retrieval work without an API key.",
        )
        model = st.text_input("Synthesis model", value="gpt-4o-mini")
        expected_role_text = st.text_area(
            "Expected roles (comma-separated)",
            value=", ".join(DEFAULT_EXPECTED_ROLES),
        )
        use_llm = st.checkbox("Generate grounded narrative review", value=bool(api_key))

    uploaded_file = st.file_uploader("Upload a .txt, .md, or .csv requirements file", type=["txt", "md", "csv"])
    default_text = ""
    source = "pasted requirements"
    if uploaded_file is not None:
        default_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
        source = uploaded_file.name
    else:
        default_text = SAMPLE_REQUIREMENTS

    requirements_text = st.text_area(
        "Requirements",
        value=default_text,
        height=260,
        help="Use one requirement per line. IDs such as BR-001 are preserved in the report.",
    )
    analyze_button = st.button("Analyze requirements", type="primary", use_container_width=True)

    if analyze_button:
        if not requirements_text.strip():
            st.error("Add or upload at least one requirement.")
            return
        expected_roles = [role.strip() for role in expected_role_text.split(",") if role.strip()]
        with st.spinner("Checking requirements and retrieving evidence..."):
            result = analyze_requirements(requirements_text, expected_roles, source)
        if use_llm:
            if not api_key:
                st.warning("Add an OpenAI API key or disable narrative review. The deterministic report is still available.")
            else:
                with st.spinner("Generating an evidence-grounded narrative review..."):
                    try:
                        result.llm_review = synthesize_with_openai(result, api_key, model)
                    except Exception as error:
                        st.error(f"Narrative review failed: {error}")
        st.session_state["analysis_result"] = result

    result = st.session_state.get("analysis_result")
    if result is not None:
        _render_result(result)
        if result.llm_review:
            st.subheader("Grounded narrative review")
            st.markdown(result.llm_review)


if __name__ == "__main__":
    main()
