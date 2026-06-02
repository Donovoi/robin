# Robin: A multi-agent system for automating scientific discovery

See our [blog](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system) or [arXiv](https://arxiv.org/abs/2505.13400) preprint for more info.

## What Robin needs from you

Before running Robin or the OpenCode-backed agent, decide these inputs:

1. **Task:** What should the agent work on? For Robin's normal workflow this is usually a disease-focused discovery goal; for engineering tests it can be a repo or algorithm question.
2. **Scope:** Should the agent run a smoke test, produce a research-only report, modify code, produce a patch, or commit changes?
3. **Target domain:** What disease, experiment area, repository, algorithm family, or scientific question should the agent focus on?
4. **Web search:** Should the agent use web search, and which SearXNG `/search` endpoint should it use if enabled?
5. **Edison access:** Should the workflow use Edison-backed literature and data-analysis agents? Full Robin scientific workflows need `EDISON_API_KEY`; lightweight OpenCode smoke tests do not.
6. **Output:** What should Robin produce: a notebook run, recommendation memo, benchmark plan, code changes, PR-ready commit, or all of those?

For a quick OpenCode agent test, the minimum useful prompt is:

```text
Task: Find a better file-search/signature-matching algorithm than bulk_extractor currently uses.
Scope: Research and produce a recommendation memo first.
Web search: Use SearXNG.
Output: Save findings and commit any repo config/doc changes if needed.
```

## Prerequisites

- **Python:** Version 3.12 or higher.
- **OpenCode:** Install the `opencode` CLI and authenticate OpenAI through provider OAuth:

  ```bash
  opencode auth login
  opencode auth list
  ```

  The default Robin LLM backend uses this OpenCode provider credential. It does not require an `OPENAI_API_KEY`.
- **API Keys:**
  - `EDISON_API_KEY`: For accessing Edison platform agents (Crow, Falcon - now called 'Literature'). Obtain from https://platform.edisonscientific.com/profile. You must first create an Edison profile, purchase credits and then create an API key (Account -> Profile -> API Tokens).
  - If you explicitly switch `RobinConfiguration(llm_backend="litellm")`, provide the credentials required by your LiteLLM provider.
  - The data analysis portion of this repo requires access to the Edison platform. Without access, all the hypothesis and experiment generation code can still be run.

## Current LLM and agent defaults

Robin's direct LLM calls now default to OpenCode-backed OpenAI provider auth instead of token-based OpenAI API key configuration:

- `llm_backend="opencode"`
- `llm_name="openai/gpt-5.5"`
- `llm_variant="xhigh"` for extra-high reasoning
- `opencode_agent_instructions` tells the agent to parallelize independent work and delegate research, coding, review, or verification subtasks to sub-agents when useful
- `web_search_url` can point OpenCode agents at a SearXNG `/search` endpoint for current web research

Edison platform agents are still used for Robin's existing literature-search and data-analysis workflows. The SearXNG option is an OpenCode agent research aid; it does not replace Edison-backed literature search.

## Docker (Alternative Setup)

Docker is a tool that packages software into a self-contained "container" that runs the same way on any computer, regardless of your operating system or what else is installed. It's the recommended approach for Robin as it avoids the most common installation issues.

**Install Docker first:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system (Mac or Windows). Once installed, open Docker Desktop and make sure it is running (you should see the Docker icon in your menu bar/system tray) before proceeding.

For a fully self-contained environment that avoids OS-level dependency conflicts, Docker is the recommended approach:

1. **Build the image:**

   ```bash
   docker build -t robin .
   ```

2. **Set up API keys:**

   ```bash
   cp .env.example .env
   # Edit .env and fill in your EDISON_API_KEY
   ```

   Important: do **not** wrap values in quotes (e.g. `EDISON_API_KEY=abc123`, not `EDISON_API_KEY="abc123"`). Docker reads the file differently from Python and will include the quotes as part of the key.

3. **Run Jupyter:**
   ```bash
   docker run -p 8888:8888 --env-file .env robin
   ```
   Jupyter will print three URLs — use only the one that starts with `http://127.0.0.1:8888/` (the other two are internal container addresses and will not work). Your URL will look like: `http://127.0.0.1:8888/lab/tree/robin_demo.ipynb?token=...`

   Note: Robin's default OpenAI model path uses the local `opencode` command and its provider auth. Docker containers do not automatically inherit the host's OpenCode OAuth session, so either authenticate OpenCode inside the container or run Robin locally for the OAuth-backed default.

### Local SearXNG for OpenCode agents

Robin can optionally give OpenCode-backed agents a local SearXNG `/search` endpoint for web research. This is separate from Robin's Edison literature-search flow and is intended as an agent search aid for tasks that benefit from current web context.

Start the local SearXNG sidecar:

```bash
docker compose -f docker-compose.search.yml up -d
```

The Compose service binds SearXNG to localhost only and enables JSON output for agent use. For host-based Robin/OpenCode runs, use:

```bash
export ROBIN_WEB_SEARCH_URL=http://127.0.0.1:8080/search
```

If OpenCode is running inside a separate Docker container on Docker Desktop, use:

```bash
export ROBIN_WEB_SEARCH_URL=http://host.docker.internal:8080/search
```

If Robin and SearXNG are attached to the same Compose network, use `http://searxng:8080/search`. You can also pass the URL directly:

```python
config = RobinConfiguration(
    disease_name="DISEASE_NAME",
    web_search_url="http://127.0.0.1:8080/search",
)
```

When `web_search_url` is set, Robin injects that endpoint into the OpenCode prompt so agents prefer SearXNG for web search, request `format=json` when structured results are useful, and verify important claims against primary sources.

---

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Donovoi/robin.git
    cd robin
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

    OR

    ```bash
    python3 -m venv .robin_env
    source .robin_env/bin/activate
    ```

3.  **Install Dependencies:**
    The project uses `pyproject.toml` for dependency management. Install the base package and development dependencies (which include Jupyter):

    ```bash
    uv pip install -e '.[dev]'
    ```

    OR

    ```bash
    pip install -e '.[dev]'
    ```

4.  **Set API Keys:**
    Copy the provided template and fill in your Edison key:
    ```bash
    cp .env.example .env
    # Then edit .env with your actual Edison key
    ```
    Robin will automatically load this `.env` file at startup. Alternatively, you can export the variable in your shell, or pass it directly when creating the `RobinConfiguration` object.

5.  **Authenticate OpenAI through OpenCode:**

    ```bash
    opencode auth login
    opencode auth list
    ```

    The default Robin LLM backend uses the OpenCode OpenAI provider credential rather than a static API key. `opencode auth list` should show OpenAI as `oauth`.

## Running Robin via `robin_demo.ipynb`

_In order to run Robin as used in the manuscript, only input the name of a disease, with no other text. If you wish to optimize how Robin searches for experimental models and therapeutic candidates, we suggest changing the internal prompts of Robin (via prompts.py), not the initial input to the pipeline._

1.  **Launch Jupyter Notebook or JupyterLab:**
    Navigate to the `robin` directory in your terminal (ensure your virtual environment is activated) and run:

    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```

2.  **Open the Notebook:**
    In the Jupyter interface, open `robin_demo.ipynb`.

3.  **Configure Robin:**
    Locate the cell where the `RobinConfiguration` object is created:

    ```python
    config = RobinConfiguration(
        disease_name="DISEASE_NAME",  # <-- Customize the disease name here
        # You can also explicitly set API keys here if not using environment variables:
        # edison_api_key="your_edison_api_key_here"
    )
    ```

    - **Modify `disease_name`**: Change `"DISEASE_NAME"` to your target disease.
    - **Edison Key**: If you didn't set `EDISON_API_KEY`, you can provide `edison_api_key` directly in the `RobinConfiguration` instantiation.
    - **LLM Choice**: The default is `openai/gpt-5.5` through OpenCode OAuth with `llm_variant="xhigh"` for extra-high reasoning. You can change `llm_name`, `llm_variant`, or `llm_backend` in `RobinConfiguration`; use `llm_backend="litellm"` only if you want LiteLLM/API-key behavior.
    - **OpenCode Agent Behavior**: The default OpenCode-backed LLM calls include `opencode_agent_instructions`, which tells the agent to parallelize independent work and hand off research, coding, review, or verification subtasks to sub-agents when useful. Override or clear this field if you need strictly serial behavior.
    - **Optional Agent Web Search**: Set `web_search_url` or `ROBIN_WEB_SEARCH_URL` to a SearXNG `/search` endpoint when you want OpenCode-backed agents to use local web search during research-heavy calls.
    - Other parameters like `num_queries`, `num_assays`, `num_candidates` can also be adjusted here if needed.

4.  **Run the Notebook Cells:**
    Execute the cells in the notebook sequentially. The notebook is structured to guide you through:
    - **Experimental Assay Generation:** Generates and ranks potential experimental assays.
    - **Therapeutic Candidate Generation:** Based on the top assay, generates and ranks therapeutic candidates.
    - **(Optional) Experimental Data Analysis:** If you have experimental data, this section can analyze it and feed insights back into candidate generation. This requires access to the Edison platform data analysis features.

## Expected Output

- **Logs:** Detailed logs will be printed in the notebook output and/or your console, showing the progress of each step (e.g., query generation, literature search, candidate proposal, ranking).

- **Files:** Results are saved in a new subdirectory within `robin_output/`, named after the `disease_name` and a timestamp (e.g., `robin_output/DISEASE_NAME_YYYY-MM-DD_HH-MM/`). This directory contains a structured set of outputs, including:
  - Folders for detailed hypotheses and literature reviews for both experimental assays and therapeutic candidates (e.g., `experimental_assay_detailed_hypotheses/`, `therapeutic_candidate_literature_reviews/`).
  - CSV files for ranking results and final ranked lists (e.g., `experimental_assay_ranking_results.csv`, `ranked_therapeutic_candidates.csv`).
  - Text summaries for proposed assays and candidates (e.g., `experimental_assay_summary.txt`, `therapeutic_candidates_summary.txt`).
  - If the optional data analysis step is run (using the `data_analysis` function), there will be an additional `data_analysis/` subfolder containing outputs from the Finch agent (e.g., `consensus_results.csv`). Correspondingly, some therapeutic candidate-related files generated after this step may have an `_experimental` suffix (e.g., `ranked_therapeutic_candidates_experimental.csv`, `therapeutic_candidate_detailed_hypotheses_experimental/`).

## Overview of `examples` Folder:

The `examples` folder provides practical usage demonstrations of pre-generated output directories from complete Robin runs for 10 diseases:

- Age-Related Hearing Loss
- Celiac Disease
- Charcot-Marie-Tooth Disease
- Chronic Kidney Disease
- Friedreich's Ataxia
- Glaucoma
- Idiopathic Pulmonary Fibrosis
- Non-alcoholic Steatohepatitis
- Polycystic Ovary Syndrome
- Sarcopenia

Each disease-specific subfolder mirrors the exact file and directory structure a user would obtain in their own `robin_output/` directory after a run:

- `experimental_assay_detailed_hypotheses/`: Text files containing detailed reports for each proposed experimental assay.
- `experimental_assay_literature_reviews/`: Text files of literature reviews generated from queries related to assay development.
- `experimental_assay_ranking_results.csv`: CSV file showing pairwise comparison results for assay ranking.
- `experimental_assay_summary.txt`: A textual summary of the proposed experimental assays.
- `ranked_therapeutic_candidates.csv`: CSV file listing the final ranked therapeutic candidates and their strength scores.
- `therapeutic_candidate_detailed_hypotheses/`: Text files with detailed reports for each proposed therapeutic candidate.
- `therapeutic_candidate_literature_reviews/`: Text files of literature reviews for therapeutic candidate queries.
- `therapeutic_candidate_ranking_results.csv`: CSV file of pairwise comparison results for candidate ranking.
- `therapeutic_candidates_summary.txt`: A textual summary of the proposed therapeutic candidates.

These example outputs are provided to help users to understand the depth, format, and typical errors seen in Robin runs across various diseases.

## Advanced Usage

A full example trajectory of both the initial therapeutic candidate generation and experimental data analysis can be found in the `robin_full.ipynb` notebook. This notebook includes the parameters and agents used in the paper.

While this guide focuses on the `robin_demo.ipynb` notebook, the `robin` Python module (in the `robin/` directory) can be imported and its functions (`experimental_assay`, `therapeutic_candidates`, `data_analysis`) can be used programmatically in your own Python scripts for more customized workflows.
