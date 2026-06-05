from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from robin.configuration import RobinConfiguration
from robin.open_literature import (
    LiteratureRecord,
    OpenLiteratureSearcher,
    call_open_literature,
)
from robin.opencode_llm import LLMResponse


class FakeLLM:
    def __init__(self) -> None:
        self.messages = []

    async def call_single(self, messages):
        self.messages.append(messages)
        return LLMResponse(
            text=(
                "Bottom line: the evidence supports a cautious conclusion [R1].\n\n"
                "Conflicting evidence, gaps, and detractor concerns: the record set is"
                " small."
            )
        )


class FakeSearcher:
    async def search(self, query: str):
        return [
            LiteratureRecord(
                title="Systematic review of example disease diagnostics",
                source_names={"OpenAlex"},
                year=2025,
                doi="10.1000/example",
                abstract="A systematic review and meta-analysis of diagnostic certainty.",
                citation_count=40,
                is_open_access=True,
            ),
            LiteratureRecord(
                title="Systematic review of example disease diagnostics",
                source_names={"PubMed"},
                year=2024,
                doi="10.1000/example",
                pmid="12345",
                citation_count=30,
            ),
        ]


class OpenLiteratureTest(TestCase):
    def test_auto_backend_uses_open_when_edison_key_is_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = RobinConfiguration(disease_name="example disease")

        assert not config.has_edison_api_key
        assert config.resolved_literature_backend == "open"

    def test_auto_backend_ignores_placeholder_edison_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"EDISON_API_KEY": "your_edison_api_key_here"},
            clear=True,
        ):
            config = RobinConfiguration(disease_name="example disease")

        assert not config.has_edison_api_key
        assert config.resolved_literature_backend == "open"

    def test_auto_backend_uses_edison_when_real_key_is_present(self) -> None:
        with patch.dict("os.environ", {"EDISON_API_KEY": "edison-real-key"}, clear=True):
            config = RobinConfiguration(disease_name="example disease")

            assert config.has_edison_api_key
            assert config.resolved_literature_backend == "edison"

    def test_forced_edison_without_key_fails_loudly(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = RobinConfiguration(
                disease_name="example disease", literature_backend="edison"
            )

        with self.assertRaises(ValueError):
            _ = config.resolved_literature_backend

    def test_dedupe_and_rank_merges_records_and_prioritizes_reviews(self) -> None:
        records = [
            LiteratureRecord(
                title="Small case report",
                source_names={"Crossref"},
                year=2026,
                citation_count=1,
            ),
            LiteratureRecord(
                title="Systematic review of example disease diagnostics",
                source_names={"OpenAlex"},
                year=2025,
                doi="10.1000/example",
                abstract="A systematic review and meta-analysis.",
                citation_count=40,
            ),
            LiteratureRecord(
                title="Systematic review of example disease diagnostics",
                source_names={"PubMed"},
                year=2024,
                doi="10.1000/example",
                pmid="12345",
                citation_count=30,
            ),
        ]

        ranked = OpenLiteratureSearcher.dedupe_and_rank(
            records, max_records=5, current_year=2026
        )

        assert len(ranked) == 2
        assert ranked[0].doi == "10.1000/example"
        assert ranked[0].pmid == "12345"
        assert ranked[0].source_names == {"OpenAlex", "PubMed"}

    def test_call_open_literature_synthesizes_from_ranked_records(self) -> None:
        llm = FakeLLM()

        result = self._run(
            call_open_literature(
                queries={"diagnosis": "diagnose example disease with certainty"},
                llm_client=llm,
                searcher=FakeSearcher(),
            )
        )

        assert result["count"] == 1
        assert not result["has_errors"]
        assert result["results"][0]["status"] == "success"
        assert "10.1000/example" in result["results"][0]["sources"]
        assert "detractor concerns" in result["results"][0]["answer"].lower()
        prompt_text = "\n".join(
            str(getattr(message, "content", ""))
            for message in llm.messages[0]
            if isinstance(message, SimpleNamespace) or hasattr(message, "content")
        )
        assert "diagnose example disease with certainty" in prompt_text
        assert "Systematic review of example disease diagnostics" in prompt_text

    @staticmethod
    def _run(coro):
        import asyncio

        return asyncio.run(coro)
