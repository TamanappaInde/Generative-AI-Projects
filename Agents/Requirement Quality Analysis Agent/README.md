# Requirement Quality Analysis Agent

A RAG-based QA agent for business requirements. It combines deterministic quality checks with local TF-IDF retrieval and optional OpenAI synthesis grounded in the retrieved requirements.

## Checks

- Exact duplicates and highly similar requirements
- Missing responsible actor or stakeholder role
- Missing or weak obligation language
- Vague and subjective wording
- Missing acceptance criteria
- Expected roles with no represented requirements

## Run the Streamlit app

```powershell
cd "Agents\Requirement Quality Analysis Agent"
pip install -r requirements.txt
streamlit run app.py
```

The local checks, retrieval, report download, and sample review work without an API key. Add `OPENAI_API_KEY` in the sidebar to generate an evidence-grounded narrative review.

## Run from the command line

```powershell
python cli.py path\to\requirements.txt --json
python cli.py path\to\requirements.txt --llm
```

Input files are UTF-8 text, Markdown, or CSV-like one-requirement-per-line files. For reliable role coverage, configure the expected roles in the app or with `--roles`.
