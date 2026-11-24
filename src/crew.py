"""BioCellAgents — Genuinely Agentic 3-Agent CrewAI Orchestration.

Quantitative vision inference, temperature calibration, spatial graph metrics,
and exact Markdown tables are computed deterministically in Python
(``src/pipeline.py`` + ``src/reporting.py``) with 0% hallucination.

``PathologyCrew`` then orchestrates 3 tool-using / reasoning agents:
1. ``literature_researcher`` — wired with ``LiteratureSearchTool`` for adaptive
   PubMed + Europe PMC query refinement.
2. ``clinical_report_writer`` — synthesizes verified quantitative tables and
   curated literature into the 11-section RUO computational pathology report.
3. ``fidelity_auditor`` — wired with ``ReportFidelityVerifierTool`` to audit
   numeric fidelity, 11-section structure, tables, PMID grounding, and RUO
   compliance in a closed verification loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, cast

from crewai import LLM, Agent, Crew, Process, Task
from crewai.llms.cache import CACHE_BREAKPOINT_KEY
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from src.tools.pubmed_search_tool import LiteratureSearchTool
from src.tools.report_verifier_tool import ReportFidelityVerifierTool

load_dotenv()

DEFAULT_MODEL = "groq/qwen/qwen3.8-27b"
logger = logging.getLogger(__name__)


def _strip_cache_breakpoint_from_item(item: Any) -> Any:
    if isinstance(item, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in item.items():
            if k == CACHE_BREAKPOINT_KEY:
                continue
            if k == "content" and isinstance(v, list):
                cleaned[k] = [_strip_cache_breakpoint_from_item(sub) for sub in v]
            else:
                cleaned[k] = v
        return cleaned
    return item


def _strip_cache_breakpoint(messages: Optional[List[Any]]) -> List[Any]:
    """Remove OpenAI-specific ``cache_breakpoint`` keys rejected by Groq."""
    if not messages:
        return []
    return [_strip_cache_breakpoint_from_item(msg) for msg in messages]


def _is_rate_limit_error(error_str: str) -> bool:
    lowered = error_str.lower()
    return "ratelimiterror" in lowered or "rate_limit" in lowered or "rate limit" in lowered


class GroqCompatibleLLM(LLM):
    """Groq-compatible CrewAI LLM wrapper.

    Strips OpenAI-only ``cache_breakpoint`` keys and retries transient Groq
    rate limits with exponential backoff while preserving full native tool-use
    capabilities for tool-wired agents.
    """

    def _format_messages_for_provider(self, messages: list) -> list:
        cleaned = _strip_cache_breakpoint(messages)
        return super()._format_messages_for_provider(cast(Any, cleaned))

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        """Invoke LLM with exponential backoff retry on Groq rate limits."""
        max_retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                result = super().call(
                    messages=messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )
                logger.info("GroqCompatibleLLM.call() succeeded")
                return result
            except Exception as exc:
                error_str = str(exc)
                logger.error(
                    "GroqCompatibleLLM.call() failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    error_str[:500],
                )
                if _is_rate_limit_error(error_str):
                    wait_time = (attempt + 1) * 12  # 12s, 24s, 36s
                    logger.warning("Rate limit hit. Retrying in %ds...", wait_time)
                    time.sleep(wait_time)
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("GroqCompatibleLLM.call() exhausted retries without captured exception")

    async def acall(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        """Async invoke LLM with exponential backoff retry on Groq rate limits."""
        max_retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                result = await super().acall(
                    messages=messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )
                logger.info("GroqCompatibleLLM.acall() succeeded")
                return result
            except Exception as exc:
                error_str = str(exc)
                logger.error(
                    "GroqCompatibleLLM.acall() failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    error_str[:500],
                )
                if _is_rate_limit_error(error_str):
                    wait_time = (attempt + 1) * 12  # 12s, 24s, 36s
                    logger.warning("Rate limit hit in acall(). Retrying in %ds...", wait_time)
                    await asyncio.sleep(wait_time)
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("GroqCompatibleLLM.acall() exhausted retries without captured exception")


def create_llm(model_name: Optional[str] = None) -> LLM:
    """Instantiate the configured CrewAI LLM (Groq-compatible or generic)."""
    if model_name is None:
        model_name = os.getenv("MODEL", DEFAULT_MODEL)
    if "groq" in model_name.lower():
        return cast(Any, GroqCompatibleLLM)(model=model_name, drop_params=True)
    return LLM(model=model_name)


@CrewBase
class PathologyCrew:
    """3-agent tool-using computational pathology crew."""

    agents_config: Any = "../config/agents.yaml"
    tasks_config: Any = "../config/tasks.yaml"

    def __init__(
        self,
        llm: Optional[LLM] = None,
        literature_tool: Optional[Any] = None,
        verifier_tool: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self._llm = llm or create_llm()
        self._literature_tool = literature_tool or LiteratureSearchTool()
        self._verifier_tool = verifier_tool or ReportFidelityVerifierTool()

    def _agent_cfg(self, name: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.agents_config)[name]

    def _task_cfg(self, name: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.tasks_config)[name]

    @agent
    def literature_researcher(self) -> Agent:
        return Agent(
            config=self._agent_cfg("literature_researcher"),
            llm=self._llm,
            tools=[self._literature_tool],
            verbose=True,
        )

    @agent
    def clinical_report_writer(self) -> Agent:
        return Agent(
            config=self._agent_cfg("clinical_report_writer"),
            llm=self._llm,
            verbose=True,
        )

    @agent
    def fidelity_auditor(self) -> Agent:
        return Agent(
            config=self._agent_cfg("fidelity_auditor"),
            llm=self._llm,
            tools=[self._verifier_tool],
            verbose=True,
        )

    @task
    def literature_research_task(self) -> Task:
        return cast(Any, Task)(config=self._task_cfg("literature_research_task"))

    @task
    def pathology_synthesis_task(self) -> Task:
        return cast(Any, Task)(config=self._task_cfg("pathology_synthesis_task"))

    @task
    def fidelity_audit_task(self) -> Task:
        return cast(Any, Task)(
            config=self._task_cfg("fidelity_audit_task"),
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        agents = cast(list[Any], getattr(self, "agents", []))
        tasks = cast(list[Any], getattr(self, "tasks", []))
        return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)
