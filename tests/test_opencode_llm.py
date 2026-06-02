from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from robin.configuration import RobinConfiguration
from robin.opencode_llm import (
    DEFAULT_OPENCODE_AGENT_INSTRUCTIONS,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENCODE_VARIANT,
    WEB_SEARCH_INSTRUCTIONS,
    OpenCodeLLMModel,
)


class OpenCodeLLMModelTest(TestCase):
    def test_defaults_use_gpt_5_5_xhigh(self) -> None:
        client = OpenCodeLLMModel()

        assert client.model == DEFAULT_OPENCODE_MODEL
        assert client.variant == DEFAULT_OPENCODE_VARIANT
        assert client.agent_instructions == DEFAULT_OPENCODE_AGENT_INSTRUCTIONS

    def test_format_messages_preserves_roles_and_content(self) -> None:
        messages = [
            SimpleNamespace(role="system", content="Follow the format."),
            SimpleNamespace(role="user", content="Return JSON."),
        ]

        formatted = OpenCodeLLMModel._format_messages(
            messages, agent_instructions="Delegate independent work."
        )

        assert "SYSTEM:\nDelegate independent work." in formatted
        assert "SYSTEM:\nFollow the format." in formatted
        assert "USER:\nReturn JSON." in formatted

    def test_format_messages_can_disable_default_instructions(self) -> None:
        messages = [SimpleNamespace(role="user", content="Return JSON.")]

        formatted = OpenCodeLLMModel._format_messages(messages)

        assert formatted == "USER:\nReturn JSON."

    def test_default_model_build_prompt_injects_agent_instructions(self) -> None:
        client = OpenCodeLLMModel()

        formatted = client._build_prompt(
            [SimpleNamespace(role="user", content="Return JSON.")]
        )

        assert DEFAULT_OPENCODE_AGENT_INSTRUCTIONS in formatted
        assert formatted.endswith("USER:\nReturn JSON.")

    def test_build_prompt_injects_web_search_endpoint_when_configured(self) -> None:
        client = OpenCodeLLMModel(web_search_url="http://searxng:8080/search")

        formatted = client._build_prompt(
            [SimpleNamespace(role="user", content="Find current sources.")]
        )

        assert WEB_SEARCH_INSTRUCTIONS.format(
            web_search_url="http://searxng:8080/search"
        ) in formatted
        assert "format=json" in formatted
        assert formatted.endswith("USER:\nFind current sources.")

    def test_configuration_passes_web_search_url_to_opencode_client(self) -> None:
        config = RobinConfiguration(
            disease_name="example disease",
            web_search_url="http://127.0.0.1:8080/search",
        )

        client = config.llm_client

        assert isinstance(client, OpenCodeLLMModel)
        assert client.web_search_url == "http://127.0.0.1:8080/search"

    def test_configuration_loads_web_search_url_from_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {"ROBIN_WEB_SEARCH_URL": "http://host.docker.internal:8080/search"},
        ):
            config = RobinConfiguration(disease_name="example disease")

        assert config.web_search_url == "http://host.docker.internal:8080/search"

    def test_extract_text_uses_only_text_events(self) -> None:
        output = "\n".join(
            [
                '{"type":"step_start","part":{}}',
                '{"type":"text","part":{"text":"hello"}}',
                "not json",
                '{"type":"text","part":{"text":" world"}}',
                '{"type":"step_finish","part":{}}',
            ]
        )

        assert OpenCodeLLMModel._extract_text(output) == "hello world"
