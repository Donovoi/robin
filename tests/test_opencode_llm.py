from types import SimpleNamespace
from unittest import TestCase

from robin.opencode_llm import (
    DEFAULT_OPENCODE_AGENT_INSTRUCTIONS,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENCODE_VARIANT,
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
