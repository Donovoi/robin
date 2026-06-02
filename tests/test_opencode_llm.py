from types import SimpleNamespace
from unittest import TestCase

from robin.opencode_llm import (
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENCODE_VARIANT,
    OpenCodeLLMModel,
)


class OpenCodeLLMModelTest(TestCase):
    def test_defaults_use_gpt_5_5_xhigh(self) -> None:
        client = OpenCodeLLMModel()

        assert client.model == DEFAULT_OPENCODE_MODEL
        assert client.variant == DEFAULT_OPENCODE_VARIANT

    def test_format_messages_preserves_roles_and_content(self) -> None:
        messages = [
            SimpleNamespace(role="system", content="Follow the format."),
            SimpleNamespace(role="user", content="Return JSON."),
        ]

        formatted = OpenCodeLLMModel._format_messages(messages)

        assert "SYSTEM:\nFollow the format." in formatted
        assert "USER:\nReturn JSON." in formatted

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
