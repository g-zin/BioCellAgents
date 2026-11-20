import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, cast

from crewai import LLM, Agent, Crew, Process, Task
from crewai.llms.cache import CACHE_BREAKPOINT_KEY
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "groq/qwen/qwen3.8-27b"
logger = logging.getLogger(__name__)

ANTI_TOOL_CALL_PROMPT = (
    "CRITICAL: You do NOT have access to any tools, functions, or external APIs. "
    "You must NEVER output tool calls, function calls, JSON tool schemas, or any "
    "structured tool-use format. Respond ONLY with plain text or Markdown. "
    "If a task asks you to search, look up, or retrieve external information, "
    "state that you cannot do so and provide your best answer based on the "
    "information given to you in the prompt."
)


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


def _strip_cache_breakpoint(messages: Optional[list[Any]]) -> list[Any]:
    if not messages:
        return []
    return [_strip_cache_breakpoint_from_item(msg) for msg in messages]


def _inject_anti_tool_call_prompt(messages: Optional[list[Any]]) -> list[Any]:
    if not messages:
        return [{"role": "system", "content": ANTI_TOOL_CALL_PROMPT}]

    new_messages = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
    for msg in new_messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            existing = msg.get("content", "")
            if isinstance(existing, str):
                if ANTI_TOOL_CALL_PROMPT not in existing:
                    msg["content"] = existing + "\n\n" + ANTI_TOOL_CALL_PROMPT
                return new_messages
            if isinstance(existing, list):
                already_present = any(
                    (isinstance(part, dict) and ANTI_TOOL_CALL_PROMPT in str(part.get("text", "")))
                    or (isinstance(part, str) and ANTI_TOOL_CALL_PROMPT in part)
                    for part in existing
                )
                if not already_present:
                    msg["content"] = list(existing) + [
                        {"type": "text", "text": f"\n\n{ANTI_TOOL_CALL_PROMPT}"}
                    ]
                return new_messages
            msg["content"] = ANTI_TOOL_CALL_PROMPT
            return new_messages

    new_messages.insert(0, {"role": "system", "content": ANTI_TOOL_CALL_PROMPT})
    return new_messages


class GroqCompatibleLLM(LLM):
    def _format_messages_for_provider(self, messages: list) -> list:
        messages = _strip_cache_breakpoint(messages)
        messages = _inject_anti_tool_call_prompt(messages)
        return super()._format_messages_for_provider(cast(Any, messages))

    def _handle_non_streaming_response(
        self,
        params,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        """Override to catch tool-call hallucination errors at the source."""
        try:
            return super()._handle_non_streaming_response(
                params=params,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )
        except Exception as e:
            error_str = str(e)
            is_tool_call_error = (
                "tool_use_failed" in error_str
                or "Tool choice is none" in error_str
                or "model called a tool" in error_str
            )
            if not is_tool_call_error:
                raise
            logger.warning(
                "Tool-call hallucination detected in _handle_non_streaming_response. Returning fallback."
            )
            return (
                "I am unable to perform external searches or tool calls as I have "
                "no tools available. Based on the information provided in the prompt, "
                "I cannot retrieve additional data. Please ensure all required data "
                "(including literature evidence, pre-computed results, and reference "
                "materials) is provided directly in the input for me to analyze and "
                "report on. If literature evidence was expected, it appears to be "
                "missing or empty in the provided data."
            )

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
        """Override call() with rate-limit retry + tool-call hallucination fallback."""
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
            except Exception as e:
                error_str = str(e)
                logger.error(
                    "GroqCompatibleLLM.call() failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    error_str[:500],
                )
                # Rate limit error — retry with exponential backoff
                if (
                    "RateLimitError" in error_str
                    or "rate_limit" in error_str.lower()
                    or "rate limit" in error_str.lower()
                ):
                    wait_time = (attempt + 1) * 12  # 12s, 24s, 36s
                    logger.warning("Rate limit hit. Retrying in %ds...", wait_time)
                    time.sleep(wait_time)
                    last_error = e
                    continue
                # Tool-call hallucination error — return fallback
                is_tool_call_error = (
                    "tool_use_failed" in error_str
                    or "Tool choice is none" in error_str
                    or "model called a tool" in error_str
                )
                if is_tool_call_error:
                    logger.warning(
                        "Tool-call hallucination detected in call(). Returning fallback."
                    )
                    return (
                        "I am unable to perform external searches or tool calls as I have "
                        "no tools available. Based on the information provided in the prompt, "
                        "I cannot retrieve additional data. Please ensure all required data "
                        "(including literature evidence, pre-computed results, and reference "
                        "materials) is provided directly in the input for me to analyze and "
                        "report on. If literature evidence was expected, it appears to be "
                        "missing or empty in the provided data."
                    )
                raise
        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise RuntimeError("GroqCompatibleLLM.call() exhausted retries without captured exception")

    async def _ahandle_non_streaming_response(
        self,
        params,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        """Async version of _handle_non_streaming_response with fallback logic."""
        try:
            return await super()._ahandle_non_streaming_response(
                params=params,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )
        except Exception as e:
            error_str = str(e)
            is_tool_call_error = (
                "tool_use_failed" in error_str
                or "Tool choice is none" in error_str
                or "model called a tool" in error_str
            )
            if not is_tool_call_error:
                raise
            logger.warning(
                "Tool-call hallucination detected in _ahandle_non_streaming_response. Returning fallback."
            )
            return (
                "I am unable to perform external searches or tool calls as I have "
                "no tools available. Based on the information provided in the prompt, "
                "I cannot retrieve additional data. Please ensure all required data "
                "(including literature evidence, pre-computed results, and reference "
                "materials) is provided directly in the input for me to analyze and "
                "report on. If literature evidence was expected, it appears to be "
                "missing or empty in the provided data."
            )

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
        """Async version of call() with rate-limit retry + tool-call hallucination fallback."""
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
            except Exception as e:
                error_str = str(e)
                logger.error(
                    "GroqCompatibleLLM.acall() failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    error_str[:500],
                )
                if (
                    "RateLimitError" in error_str
                    or "rate_limit" in error_str.lower()
                    or "rate limit" in error_str.lower()
                ):
                    wait_time = (attempt + 1) * 12  # 12s, 24s, 36s
                    logger.warning("Rate limit hit in acall(). Retrying in %ds...", wait_time)
                    await asyncio.sleep(wait_time)
                    last_error = e
                    continue
                is_tool_call_error = (
                    "tool_use_failed" in error_str
                    or "Tool choice is none" in error_str
                    or "model called a tool" in error_str
                )
                if is_tool_call_error:
                    logger.warning(
                        "Tool-call hallucination detected in acall(). Returning fallback."
                    )
                    return (
                        "I am unable to perform external searches or tool calls as I have "
                        "no tools available. Based on the information provided in the prompt, "
                        "I cannot retrieve additional data. Please ensure all required data "
                        "(including literature evidence, pre-computed results, and reference "
                        "materials) is provided directly in the input for me to analyze and "
                        "report on. If literature evidence was expected, it appears to be "
                        "missing or empty in the provided data."
                    )
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("GroqCompatibleLLM.acall() exhausted retries without captured exception")


def create_llm(model_name: Optional[str] = None) -> LLM:
    if model_name is None:
        model_name = os.getenv("MODEL", DEFAULT_MODEL)
    if "groq" in model_name.lower():
        return cast(Any, GroqCompatibleLLM)(model=model_name, drop_params=True)
    return LLM(model=model_name)


@CrewBase
class PathologyCrew:
    agents_config: Any = "../config/agents.yaml"
    tasks_config: Any = "../config/tasks.yaml"

    def __init__(self, llm: Optional[LLM] = None, **kwargs: Any) -> None:
        self._llm = llm or create_llm()

    def _agent_cfg(self, name: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.agents_config)[name]

    def _task_cfg(self, name: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.tasks_config)[name]

    @agent
    def pathology_cv_analyst(self) -> Agent:
        return Agent(config=self._agent_cfg("pathology_cv_analyst"), llm=self._llm, verbose=True)

    @agent
    def spatial_bio_analyst(self) -> Agent:
        return Agent(config=self._agent_cfg("spatial_bio_analyst"), llm=self._llm, verbose=True)

    @agent
    def literature_researcher(self) -> Agent:
        return Agent(config=self._agent_cfg("literature_researcher"), llm=self._llm, verbose=True)

    @agent
    def clinical_report_writer(self) -> Agent:
        return Agent(config=self._agent_cfg("clinical_report_writer"), llm=self._llm, verbose=True)

    @task
    def image_parsing_task(self) -> Task:
        return cast(Any, Task)(config=self._task_cfg("image_parsing_task"))

    @task
    def spatial_analysis_task(self) -> Task:
        return cast(Any, Task)(config=self._task_cfg("spatial_analysis_task"))

    @task
    def literature_research_task(self) -> Task:
        return cast(Any, Task)(config=self._task_cfg("literature_research_task"))

    @task
    def pathology_synthesis_task(self) -> Task:
        return cast(Any, Task)(
            config=self._task_cfg("pathology_synthesis_task"), output_file="report.md"
        )

    @crew
    def crew(self) -> Crew:
        agents = cast(list[Any], getattr(self, "agents", []))
        tasks = cast(list[Any], getattr(self, "tasks", []))
        return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)
