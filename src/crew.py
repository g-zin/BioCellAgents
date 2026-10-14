import os
import time
import logging

from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.llms.cache import CACHE_BREAKPOINT_KEY

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


def _strip_cache_breakpoint(messages: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in msg.items() if k != CACHE_BREAKPOINT_KEY}
        for msg in messages
    ]


def _inject_anti_tool_call_prompt(messages: list[dict]) -> list[dict]:
    for msg in messages:
        if msg.get("role") == "system":
            existing = msg.get("content", "")
            if isinstance(existing, str) and ANTI_TOOL_CALL_PROMPT not in existing:
                msg["content"] = existing + "\n\n" + ANTI_TOOL_CALL_PROMPT
            return messages
    messages.insert(0, {"role": "system", "content": ANTI_TOOL_CALL_PROMPT})
    return messages


class GroqCompatibleLLM(LLM):
    def _format_messages_for_provider(self, messages: list) -> list:
        messages = _strip_cache_breakpoint(messages)
        messages = _inject_anti_tool_call_prompt(messages)
        return super()._format_messages_for_provider(messages)

    def _handle_non_streaming_response(self, params, callbacks=None,
                                       available_functions=None, from_task=None,
                                       from_agent=None, response_model=None):
        """Override to catch tool-call hallucination errors at the source."""
        try:
            return super()._handle_non_streaming_response(
                params=params, callbacks=callbacks,
                available_functions=available_functions, from_task=from_task,
                from_agent=from_agent, response_model=response_model,
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
            logger.warning("Tool-call hallucination detected in _handle_non_streaming_response. Returning fallback.")
            return (
                "I am unable to perform external searches or tool calls as I have "
                "no tools available. Based on the information provided in the prompt, "
                "I cannot retrieve additional data. Please ensure all required data "
                "(including literature evidence, pre-computed results, and reference "
                "materials) is provided directly in the input for me to analyze and "
                "report on. If literature evidence was expected, it appears to be "
                "missing or empty in the provided data."
            )

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        """Override call() with rate-limit retry + tool-call hallucination fallback."""
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                result = super().call(
                    messages=messages, tools=tools, callbacks=callbacks,
                    available_functions=available_functions, from_task=from_task,
                    from_agent=from_agent, response_model=response_model,
                )
                logger.info("GroqCompatibleLLM.call() succeeded")
                return result
            except Exception as e:
                error_str = str(e)
                logger.error("GroqCompatibleLLM.call() failed (attempt %d/%d): %s",
                             attempt + 1, max_retries, error_str[:500])
                # Rate limit error — retry with exponential backoff
                if "RateLimitError" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
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
                    logger.warning("Tool-call hallucination detected in call(). Returning fallback.")
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
        raise last_error

    async def _ahandle_non_streaming_response(self, params, callbacks=None,
                                               available_functions=None, from_task=None,
                                               from_agent=None, response_model=None):
        """Async version of _handle_non_streaming_response with fallback logic."""
        try:
            return await super()._ahandle_non_streaming_response(
                params=params, callbacks=callbacks,
                available_functions=available_functions, from_task=from_task,
                from_agent=from_agent, response_model=response_model,
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
            logger.warning("Tool-call hallucination detected in _ahandle_non_streaming_response. Returning fallback.")
            return (
                "I am unable to perform external searches or tool calls as I have "
                "no tools available. Based on the information provided in the prompt, "
                "I cannot retrieve additional data. Please ensure all required data "
                "(including literature evidence, pre-computed results, and reference "
                "materials) is provided directly in the input for me to analyze and "
                "report on. If literature evidence was expected, it appears to be "
                "missing or empty in the provided data."
            )

    async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                    from_task=None, from_agent=None, response_model=None):
        """Async version of call() with the same fallback logic."""
        try:
            return await super().acall(
                messages=messages, tools=tools, callbacks=callbacks,
                available_functions=available_functions, from_task=from_task,
                from_agent=from_agent, response_model=response_model,
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
            logger.warning("Tool-call hallucination detected in acall(). Returning fallback.")
            return (
                "I am unable to perform external searches or tool calls as I have "
                "no tools available. Based on the information provided in the prompt, "
                "I cannot retrieve additional data. Please ensure all required data "
                "(including literature evidence, pre-computed results, and reference "
                "materials) is provided directly in the input for me to analyze and "
                "report on. If literature evidence was expected, it appears to be "
                "missing or empty in the provided data."
            )


def create_llm(model_name: str = None) -> LLM:
    if model_name is None:
        model_name = os.getenv("MODEL", DEFAULT_MODEL)
    kwargs = {"model": model_name}
    if "groq" in model_name.lower():
        return GroqCompatibleLLM(**kwargs, drop_params=True)
    return LLM(**kwargs)


@CrewBase
class PathologyCrew:
    agents_config = "../config/agents.yaml"
    tasks_config = "../config/tasks.yaml"

    def __init__(self, llm: LLM = None, **kwargs):
        self._llm = llm or create_llm()

    @agent
    def pathology_cv_analyst(self) -> Agent:
        return Agent(config=self.agents_config["pathology_cv_analyst"], llm=self._llm, verbose=True)

    @agent
    def spatial_bio_analyst(self) -> Agent:
        return Agent(config=self.agents_config["spatial_bio_analyst"], llm=self._llm, verbose=True)

    @agent
    def literature_researcher(self) -> Agent:
        return Agent(config=self.agents_config["literature_researcher"], llm=self._llm, verbose=True)

    @agent
    def clinical_report_writer(self) -> Agent:
        return Agent(config=self.agents_config["clinical_report_writer"], llm=self._llm, verbose=True)

    @task
    def image_parsing_task(self) -> Task:
        return Task(config=self.tasks_config["image_parsing_task"])

    @task
    def spatial_analysis_task(self) -> Task:
        return Task(config=self.tasks_config["spatial_analysis_task"])

    @task
    def literature_research_task(self) -> Task:
        return Task(config=self.tasks_config["literature_research_task"])

    @task
    def pathology_synthesis_task(self) -> Task:
        return Task(config=self.tasks_config["pathology_synthesis_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True)
