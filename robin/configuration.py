import copy
import os
import re
from datetime import datetime
from typing import Any, Literal

from dotenv import load_dotenv
from edison_client import EdisonClient, JobNames
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .opencode_llm import (
    DEFAULT_OPENCODE_AGENT_INSTRUCTIONS,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENCODE_VARIANT,
    OpenCodeLLMModel,
    RobinLLMClient,
)
from .prompts import (
    ANALYSIS_QUERIES,
    ASSAY_HYPOTHESIS_FORMAT,
    ASSAY_HYPOTHESIS_SYSTEM_PROMPT,
    ASSAY_LITERATURE_SYSTEM_MESSAGE,
    ASSAY_LITERATURE_USER_MESSAGE,
    ASSAY_PROPOSAL_SYSTEM_MESSAGE,
    ASSAY_PROPOSAL_USER_MESSAGE,
    ASSAY_RANKING_PROMPT_FORMAT,
    ASSAY_RANKING_SYSTEM_PROMPT,
    CANDIDATE_GENERATION_SYSTEM_MESSAGE,
    CANDIDATE_GENERATION_USER_MESSAGE,
    CANDIDATE_LIT_REVIEW_DIRECTION_PROMPT,
    CANDIDATE_QUERY_GENERATION_CONTENT_MESSAGE,
    CANDIDATE_QUERY_GENERATION_SYSTEM_MESSAGE,
    CANDIDATE_RANKING_PROMPT_FORMAT,
    CANDIDATE_RANKING_SYSTEM_PROMPT,
    CANDIDATE_REPORT_FORMAT,
    CHAIN_OF_THOUGHT_AGNOSTIC,
    CONSENSUS_QUERIES,
    COT,
    DATA_INTERPRETATION_CONTENT_MESSAGE,
    DATA_INTERPRETATION_SYSTEM_MESSAGE,
    EXPERIMENTAL_INSIGHTS_APPENDAGE,
    EXPERIMENTAL_INSIGHTS_FOR_CANDIDATE_GENERATION,
    FOLLOWUP_CONTENT_MESSAGE,
    FOLLOWUP_SYSTEM_MESSAGE,
    GENERAL_NOTEBOOK_GUIDELINES,
    GUIDELINE,
    R_SPECIFIC_GUIDELINES,
    SYNTHESIZE_SYSTEM_MESSAGE_CONTENT,
    SYNTHESIZE_USER_CONTENT,
)

load_dotenv()

_EDISON_PLACEHOLDERS = {
    "",
    "insert_edison_api_key_here",
    "your_edison_api_key_here",
    "none",
    "null",
}

_DEFAULT_LLM_CONFIG_DATA = {
    "model_list": [
        {
            "model_name": DEFAULT_OPENCODE_MODEL,
            "litellm_params": {
                "model": DEFAULT_OPENCODE_MODEL,
                "api_key": "",
                "timeout": 300,
            },
        }
    ]
}


def get_default_llm_config() -> dict[str, Any]:
    # Key is read on each instantiation so env vars set after import are picked up.
    data: dict[str, Any] = copy.deepcopy(_DEFAULT_LLM_CONFIG_DATA)
    data["model_list"][0]["litellm_params"]["api_key"] = os.getenv(
        "OPENAI_API_KEY", "insert_openai_key_here"
    )
    return data


def _default_web_search_url() -> str | None:
    return os.getenv("ROBIN_WEB_SEARCH_URL") or os.getenv("SEARXNG_SEARCH_URL")


def _default_literature_backend() -> str:
    return os.getenv("ROBIN_LITERATURE_BACKEND", "auto")


def _default_open_literature_email() -> str | None:
    return (
        os.getenv("ROBIN_LITERATURE_EMAIL")
        or os.getenv("OPENALEX_MAILTO")
        or os.getenv("UNPAYWALL_EMAIL")
        or os.getenv("NCBI_EMAIL")
    )


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in _EDISON_PLACEHOLDERS:
        return None
    return cleaned


def _get_prompt_args(template_string: str) -> set[str]:
    """
    Extracts root variable names from f-string like placeholders (e.g., {variable})
    using a direct regex approach.
    """  # noqa: D205
    placeholders: set[str] = set()
    placeholders.update(
        match.group(1)
        for match in re.finditer(
            r"(?<!{){([a-zA-Z_][a-zA-Z0-9_]*)[^}]*}(?!})", template_string
        )
    )
    return placeholders


class Prompts(BaseModel):
    analysis_queries: dict[str, str] = Field(default_factory=lambda: ANALYSIS_QUERIES)
    consensus_queries: dict[str, str] = Field(default_factory=lambda: CONSENSUS_QUERIES)
    assay_literature_system_message: str = Field(
        default=ASSAY_LITERATURE_SYSTEM_MESSAGE
    )
    assay_literature_user_message: str = Field(default=ASSAY_LITERATURE_USER_MESSAGE)
    assay_proposal_system_message: str = Field(default=ASSAY_PROPOSAL_SYSTEM_MESSAGE)
    assay_proposal_user_message: str = Field(default=ASSAY_PROPOSAL_USER_MESSAGE)
    assay_hypothesis_system_prompt: str = Field(default=ASSAY_HYPOTHESIS_SYSTEM_PROMPT)
    assay_hypothesis_format: str = Field(default=ASSAY_HYPOTHESIS_FORMAT)
    assay_ranking_system_prompt: str = Field(default=ASSAY_RANKING_SYSTEM_PROMPT)
    assay_ranking_prompt_format: str = Field(default=ASSAY_RANKING_PROMPT_FORMAT)
    synthesize_user_content: str = Field(default=SYNTHESIZE_USER_CONTENT)
    synthesize_system_message_content: str = Field(
        default=SYNTHESIZE_SYSTEM_MESSAGE_CONTENT
    )
    candidate_query_generation_system_message: str = Field(
        default=CANDIDATE_QUERY_GENERATION_SYSTEM_MESSAGE
    )
    experimental_insights_appendage: str = Field(
        default=EXPERIMENTAL_INSIGHTS_APPENDAGE
    )
    candidate_query_generation_content_message: str = Field(
        default=CANDIDATE_QUERY_GENERATION_CONTENT_MESSAGE
    )
    candidate_generation_system_message: str = Field(
        default=CANDIDATE_GENERATION_SYSTEM_MESSAGE
    )
    candidate_generation_user_message: str = Field(
        default=CANDIDATE_GENERATION_USER_MESSAGE
    )
    experimental_insights_for_candidate_generation: str = Field(
        default=EXPERIMENTAL_INSIGHTS_FOR_CANDIDATE_GENERATION
    )
    candidate_lit_review_direction_prompt: str = Field(
        default=CANDIDATE_LIT_REVIEW_DIRECTION_PROMPT
    )
    candidate_report_format: str = Field(default=CANDIDATE_REPORT_FORMAT)
    candidate_ranking_system_prompt: str = Field(
        default=CANDIDATE_RANKING_SYSTEM_PROMPT
    )
    candidate_ranking_prompt_format: str = Field(
        default=CANDIDATE_RANKING_PROMPT_FORMAT
    )
    cot: str = Field(default=COT)
    guideline: str = Field(default=GUIDELINE)
    data_interpretation_system_message: str = Field(
        default=DATA_INTERPRETATION_SYSTEM_MESSAGE
    )
    data_interpretation_content_message: str = Field(
        default=DATA_INTERPRETATION_CONTENT_MESSAGE
    )
    followup_system_message: str = Field(default=FOLLOWUP_SYSTEM_MESSAGE)
    followup_content_message: str = Field(default=FOLLOWUP_CONTENT_MESSAGE)
    general_notebook_guidelines: str = Field(default=GENERAL_NOTEBOOK_GUIDELINES)
    r_specific_guidelines: str = Field(default=R_SPECIFIC_GUIDELINES)
    cot_agnostic: str = Field(default=CHAIN_OF_THOUGHT_AGNOSTIC)

    @model_validator(mode="after")
    def validate_all_prompts(self) -> "Prompts":
        current_prompt_expectations: dict[str, set[str]] = {
            "data_interpretation_content_message": {"goal", "data_html"},
            "followup_content_message": {
                "goal",
                "analysis_summary",
                "mechanistic_insights",
                "questions_raised",
            },
            "assay_literature_system_message": {"num_assays"},
            "assay_literature_user_message": {"num_queries", "disease_name"},
            "assay_proposal_system_message": {"num_assays"},
            "assay_proposal_user_message": {
                "num_assays",
                "disease_name",
                "assay_lit_review_output",
            },
            "assay_hypothesis_system_prompt": {"disease_name"},
            "assay_hypothesis_format": {"disease_name"},
            "assay_ranking_system_prompt": {"disease_name"},
            "synthesize_user_content": {"assay_name", "disease_name"},
            "synthesize_system_message_content": {"disease_name"},
            "candidate_query_generation_system_message": {"disease_name"},
            "experimental_insights_appendage": {
                "candidate_generation_goal",
                "experimental_insights_analysis_summary",
                "experimental_insights_mechanistic_insights",
                "experimental_insights_questions_raised",
            },
            "candidate_query_generation_content_message": {
                "num_queries",
                "double_queries",
                "candidate_generation_goal",
                "disease_name",
            },
            "candidate_generation_system_message": {"disease_name", "num_candidates"},
            "candidate_generation_user_message": {
                "num_candidates",
                "disease_name",
                "therapeutic_candidate_review_output",
            },
            "experimental_insights_for_candidate_generation": {
                "candidate_generation_goal",
                "experimental_insights_analysis_summary",
                "experimental_insights_mechanistic_insights",
                "experimental_insights_questions_raised",
            },
            "candidate_lit_review_direction_prompt": {"disease_name"},
            "candidate_report_format": {"disease_name"},
            "candidate_ranking_system_prompt": {"disease_name"},
            "cot": set(),
            "guideline": set(),
            "assay_ranking_prompt_format": set(),
            "candidate_ranking_prompt_format": set(),
            "data_interpretation_system_message": set(),
            "followup_system_message": set(),
            "analysis_queries": set(),
            "consensus_queries": set(),
        }

        for field_name, expected_args in current_prompt_expectations.items():
            if not hasattr(self, field_name):
                raise ValueError(
                    f"Prompt field '{field_name}' defined in PROMPT_EXPECTATIONS but"
                    " not found in Prompts model."
                )

            prompt_template_value = getattr(self, field_name)

            if isinstance(prompt_template_value, dict):
                continue

            if not isinstance(prompt_template_value, str):
                raise TypeError(f"Prompt field '{field_name}' is not a string type.")

            actual_placeholders = _get_prompt_args(prompt_template_value)

            missing_in_template = expected_args - actual_placeholders
            if missing_in_template:
                raise ValueError(
                    f"Prompt '{field_name}' is missing expected placeholders:"
                    f" {missing_in_template}. Expected: {sorted(expected_args)}, Found:"
                    f" {sorted(actual_placeholders)}"
                )

            unexpected_in_template = actual_placeholders - expected_args
            if unexpected_in_template:
                raise ValueError(
                    f"Prompt '{field_name}' contains unexpected placeholders:"
                    f" {unexpected_in_template}. Expected: {sorted(expected_args)},"
                    f" Found: {sorted(actual_placeholders)}"
                )

        return self


class AgentConfig(BaseModel):
    assay_lit_search_agent: JobNames = Field(
        default=JobNames.CROW,
        description="Agent to use for literature search during assay idea generation.",
    )
    assay_hypothesis_report_agent: JobNames = Field(
        default=JobNames.CROW,
        description="Agent to use for generating detailed reports on assay hypotheses.",
    )
    candidate_lit_search_agent: JobNames = Field(
        default=JobNames.CROW,
        description=(
            "Agent to use for literature search during therapeutic candidate idea"
            " generation."
        ),
    )
    candidate_hypothesis_report_agent: JobNames = Field(
        default=JobNames.FALCON,
        description=(
            "Agent to use for generating detailed reports on therapeutic candidates."
        ),
    )


class RobinConfiguration(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    prompts: Prompts = Field(default_factory=Prompts)
    num_queries: int = Field(
        default=3,
        description=(
            "Number of queries to generate for each step, more means more data but also"
            " more cost."
        ),
    )
    num_assays: int = Field(default=3, description="Number of assay to generate.")
    num_candidates: int = Field(
        default=5, description="Number of candidates to generate for each query."
    )
    disease_name: str = Field(
        default="input_disease", description="Name of the disease to focus on."
    )
    run_folder_name: str | None = Field(
        default=None,
        description=(
            "Name of the folder where results will be stored. "
            "If not provided or None, it will be auto-generated "
            "using the disease_name and the timestamp."
        ),
    )
    edison_api_key: str | None = None
    literature_backend: Literal["auto", "edison", "open"] = Field(
        default_factory=_default_literature_backend,
        description=(
            "Literature research backend. 'auto' uses Edison when a real key is"
            " configured and otherwise falls back to open literature APIs."
        ),
    )
    llm_backend: Literal["opencode", "litellm"] = Field(
        default="opencode",
        description=(
            "Backend for direct LLM calls. The default uses OpenCode provider auth,"
            " which supports OAuth-backed OpenAI credentials."
        ),
    )
    llm_name: str = DEFAULT_OPENCODE_MODEL
    llm_variant: str | None = Field(
        default=DEFAULT_OPENCODE_VARIANT,
        description=(
            "OpenCode model variant. For OpenAI reasoning models, 'xhigh' maps to"
            " extra-high reasoning."
        ),
    )
    opencode_command: str = Field(
        default="opencode",
        description="OpenCode CLI command used when llm_backend='opencode'.",
    )
    opencode_agent_instructions: str = Field(
        default=DEFAULT_OPENCODE_AGENT_INSTRUCTIONS,
        description=(
            "Default instruction injected into OpenCode-backed LLM calls to encourage"
            " parallel execution and useful sub-agent delegation."
        ),
    )
    web_search_url: str | None = Field(
        default_factory=_default_web_search_url,
        description=(
            "Optional SearXNG /search endpoint made available to OpenCode-backed"
            " agents for web research. Defaults to ROBIN_WEB_SEARCH_URL or"
            " SEARXNG_SEARCH_URL when set."
        ),
    )
    open_literature_email: str | None = Field(
        default_factory=_default_open_literature_email,
        description=(
            "Optional contact email sent to polite/open scholarly APIs such as"
            " OpenAlex, Crossref, and NCBI E-utilities."
        ),
    )
    semantic_scholar_api_key: str | None = Field(
        default_factory=lambda: _optional_env("SEMANTIC_SCHOLAR_API_KEY"),
        description=(
            "Optional free Semantic Scholar API key for higher rate limits in the"
            " open literature fallback."
        ),
    )
    openalex_api_key: str | None = Field(
        default_factory=lambda: _optional_env("OPENALEX_API_KEY"),
        description=(
            "Optional free OpenAlex API key for higher rate limits in the open"
            " literature fallback."
        ),
    )
    open_literature_max_results_per_source: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum records retrieved from each open literature source per query."
        ),
    )
    open_literature_max_evidence_records: int = Field(
        default=15,
        ge=1,
        description="Maximum deduplicated records supplied to the synthesis LLM.",
    )
    open_literature_timeout: float = Field(
        default=20.0,
        gt=0,
        description="HTTP timeout for each open literature API call.",
    )
    open_literature_max_concurrent_queries: int = Field(
        default=4,
        ge=1,
        description="Maximum open literature query syntheses running at once.",
    )
    llm_config: dict | None = None
    agent_settings: AgentConfig = Field(default_factory=AgentConfig)
    _edison_client: EdisonClient | None = PrivateAttr(default=None)
    _llm_client: RobinLLMClient | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_run_folder_name_default(self) -> "RobinConfiguration":
        if self.run_folder_name is None:
            disease_part = self.disease_name[:70].replace(" ", "_")
            timestamp_part = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_folder_name = f"{disease_part}_{timestamp_part}"
        return self

    @property
    def resolved_edison_api_key(self) -> str | None:
        return _clean_secret(os.getenv("EDISON_API_KEY")) or _clean_secret(
            self.edison_api_key
        )

    @property
    def has_edison_api_key(self) -> bool:
        return self.resolved_edison_api_key is not None

    @property
    def resolved_literature_backend(self) -> Literal["edison", "open"]:
        if self.literature_backend == "auto":
            return "edison" if self.has_edison_api_key else "open"
        if self.literature_backend == "edison" and not self.has_edison_api_key:
            raise ValueError(
                "literature_backend='edison' was requested, but no real"
                " EDISON_API_KEY was configured. Set EDISON_API_KEY or use"
                " literature_backend='auto'/'open'."
            )
        return self.literature_backend

    @property
    def edison_client(self) -> EdisonClient:
        if self._edison_client is None:
            api_key = self.resolved_edison_api_key
            if not api_key:
                raise ValueError(
                    "Edison API key is not set. Please provide it in the"
                    " configuration or set EDISON_API_KEY env variable."
                )
            self._edison_client = EdisonClient(api_key=api_key)
        return self._edison_client

    @property
    def llm_client(self) -> RobinLLMClient:
        if self._llm_client is None:
            if self.llm_backend == "opencode":
                self._llm_client = OpenCodeLLMModel(
                    model=self.llm_name,
                    variant=self.llm_variant,
                    command=self.opencode_command,
                    agent_instructions=self.opencode_agent_instructions,
                    web_search_url=self.web_search_url,
                )
            else:
                from lmi import LiteLLMModel

                llm_config = self.llm_config or get_default_llm_config()
                self._llm_client = LiteLLMModel(
                    name=self.llm_name, config=llm_config
                )
        return self._llm_client

    def get_da_client(self):
        from .multitrajectory_runner import MultiTrajectoryRunner

        return MultiTrajectoryRunner(configuration=self)
