import asyncio
import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aviary.core import Message

logger = logging.getLogger(__name__)

DEFAULT_OPENCODE_MODEL = "openai/gpt-5.5"
DEFAULT_OPENCODE_VARIANT = "xhigh"
DEFAULT_OPENCODE_AGENT_INSTRUCTIONS = (
    "Parallelize independent work as aggressively as correctness allows. "
    "When a task naturally splits into independent research, coding, review, "
    "or verification subtasks, hand those subtasks off to sub-agents and "
    "integrate their results instead of doing everything serially. Keep "
    "shared context explicit, avoid duplicated work, and preserve final "
    "scientific and engineering accuracy over speed."
)


@dataclass(frozen=True)
class LLMResponse:
    text: str


class RobinLLMClient(Protocol):
    async def call_single(self, messages: Sequence[Message]) -> LLMResponse:
        """Return one LLM response for a message list."""


class OpenCodeLLMModel:
    """LLM client backed by OpenCode provider auth instead of a static API key."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENCODE_MODEL,
        variant: str | None = DEFAULT_OPENCODE_VARIANT,
        command: str = "opencode",
        cwd: str | Path | None = None,
        timeout: int = 600,
        agent_instructions: str = DEFAULT_OPENCODE_AGENT_INSTRUCTIONS,
    ) -> None:
        self.model = model
        self.variant = variant
        self.command = command
        self.cwd = Path(cwd) if cwd is not None else None
        self.timeout = timeout
        self.agent_instructions = agent_instructions

    async def call_single(self, messages: Sequence[Message]) -> LLMResponse:
        prompt = self._build_prompt(messages)
        command = [
            self._resolve_command(),
            "run",
            "--model",
            self.model,
            "--format",
            "json",
        ]
        if self.variant:
            command.extend(["--variant", self.variant])

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd) if self.cwd is not None else None,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(
                f"OpenCode LLM call timed out after {self.timeout}s."
            ) from None

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            raise RuntimeError(
                "OpenCode LLM call failed with exit code"
                f" {process.returncode}: {stderr.strip()}"
            )

        response_text = self._extract_text(stdout)
        if not response_text.strip():
            raise RuntimeError(
                "OpenCode LLM call completed without a text response."
                f" Stderr: {stderr.strip()}"
            )

        return LLMResponse(text=response_text)

    def _build_prompt(self, messages: Sequence[Message]) -> str:
        return self._format_messages(
            messages, agent_instructions=self.agent_instructions
        )

    @staticmethod
    def _format_messages(
        messages: Sequence[Message], *, agent_instructions: str = ""
    ) -> str:
        formatted_messages = []
        if agent_instructions.strip():
            formatted_messages.append(
                f"SYSTEM:\n{agent_instructions.strip()}"
            )
        for message in messages:
            role = str(getattr(message, "role", "user")).upper()
            content = getattr(message, "content", "")
            formatted_messages.append(f"{role}:\n{content}")
        return "\n\n".join(formatted_messages)

    @staticmethod
    def _extract_text(output: str) -> str:
        text_parts: list[str] = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON OpenCode output line: %s", line)
                continue

            if event.get("type") != "text":
                continue

            part = event.get("part", {})
            text = part.get("text", "")
            if isinstance(text, str):
                text_parts.append(text)

        return "".join(text_parts)

    def _resolve_command(self) -> str:
        if os.name == "nt" and not Path(self.command).suffix:
            windows_command = shutil.which(f"{self.command}.cmd")
            if windows_command:
                return windows_command

        return shutil.which(self.command) or self.command
