# Business Performance Agent

A work-in-progress AI agent that answers questions about a company's financial performance and helps explain the trends behind the numbers.

The goal is to bring financial disclosures, calculations, and business context into one question-led analysis workflow. **Honasa Consumer** is the first company used to test the pipeline, but the code is designed to work with other companies too.

## What's built so far

- **Document discovery:** Finds financial reports and related PDFs from a company's website.
- **Financial extraction:** Reads selected PDFs with an LLM and records financial figures alongside their sources.
- **Data preparation:** Standardises metric names, units and reporting periods, then consolidates the extracted figures. Source conflicts and annual-versus-quarterly differences are retained as flags.
- **Calculation engine:** Computes growth rates, margins, cost ratios, year-on-year changes and trailing-twelve-month metrics using Python, without relying on the LLM for arithmetic.
- **Financial Q&A (Stage 1):** A LangChain agent can look up reported figures, calculated metrics and data-quality flags to answer questions about the available financial data.

The Streamlit interface currently covers document discovery and extraction. Data preparation, calculations and the first version of the agent run separately in Python.

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your LLM API key locally in `.streamlit/secrets.toml` or through the relevant environment variable. **Do not commit API keys.**

3. Start the document discovery and extraction interface:

   ```bash
   streamlit run app.py
   ```

4. Once financial data has been extracted, prepare it and run the calculations:

   ```bash
   python financial_data_preparation.py
   python financial_calculations.py --run-dir data/runs/<run-folder>
   ```

5. Ask questions using the prepared financial data:

   ```bash
   python financial_reasoning_agent.py --run-dir data/runs/<run-folder>
   ```

Downloaded PDFs, cached outputs, run history and API keys are not included in this repository.

## Coming next

The next step is to let the agent search investor presentations and earnings-call transcripts for explanations behind financial trends. Later, it will be able to request additional documents and market research when a question needs more evidence, with the full experience brought into a conversational Streamlit interface.
