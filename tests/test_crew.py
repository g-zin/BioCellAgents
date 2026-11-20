"""Offline unit tests for src.crew and YAML agent/task configurations."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from crewai import LLM, BaseLLM
from crewai.llms.cache import CACHE_BREAKPOINT_KEY

from src.crew import (
    ANTI_TOOL_CALL_PROMPT,
    GroqCompatibleLLM,
    PathologyCrew,
    _inject_anti_tool_call_prompt,
    _strip_cache_breakpoint,
    create_llm,
)


class TestCrewHelpers:
    """Tests for message sanitization, retry/fallback, LLM factory, and YAML configs."""

    def test_strip_cache_breakpoint_removes_only_breakpoint_key(self):
        msgs = [
            {"role": "system", "content": "sys", CACHE_BREAKPOINT_KEY: True},
            {"role": "user", "content": "hello"},
        ]
        cleaned = _strip_cache_breakpoint(msgs)
        assert cleaned == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

    def test_inject_anti_tool_call_prompt_existing_and_missing_system_and_idempotent(self):
        with_sys = [{"role": "system", "content": "Be helpful."}]
        out1 = _inject_anti_tool_call_prompt(with_sys)
        assert ANTI_TOOL_CALL_PROMPT in out1[0]["content"]
        out2 = _inject_anti_tool_call_prompt(out1)
        assert out2[0]["content"].count(ANTI_TOOL_CALL_PROMPT) == 1

        no_sys = [{"role": "user", "content": "Analyze"}]
        out_no_sys = _inject_anti_tool_call_prompt(no_sys)
        assert out_no_sys[0]["role"] == "system"
        assert out_no_sys[0]["content"] == ANTI_TOOL_CALL_PROMPT
        assert out_no_sys[1]["role"] == "user"

    def test_groq_compatible_llm_call_rate_limit_retry_and_exhaustion(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr("src.crew.time.sleep", lambda s: sleep_calls.append(s))

        attempts = {"n": 0}

        def _flaky_call(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("RateLimitError: 429")
            return "ok"

        monkeypatch.setattr(LLM, "call", _flaky_call)
        llm = GroqCompatibleLLM(model="groq/qwen/qwen3.8-27b", api_key="test-key")
        assert llm.call([{"role": "user", "content": "hi"}]) == "ok"
        assert sleep_calls == [12, 24]

        sleep_calls.clear()

        def _always_rate_limit(self, *args, **kwargs):
            raise RuntimeError("rate limit exceeded")

        monkeypatch.setattr(LLM, "call", _always_rate_limit)
        with pytest.raises(RuntimeError, match="rate limit"):
            llm.call([{"role": "user", "content": "hi"}])
        assert len(sleep_calls) >= 3
        assert sleep_calls[:3] == [12, 24, 36]
        assert all(sleep_calls[i : i + 3] == [12, 24, 36] for i in range(0, len(sleep_calls), 3))

    @pytest.mark.parametrize(
        "err_msg",
        ["tool_use_failed", "Tool choice is none", "model called a tool"],
    )
    def test_groq_compatible_llm_tool_call_fallback_and_non_retry_error(self, monkeypatch, err_msg):
        sleep_calls = []
        monkeypatch.setattr("src.crew.time.sleep", lambda s: sleep_calls.append(s))

        def _tool_err(self, *args, **kwargs):
            raise RuntimeError(err_msg)

        async def _async_tool_err(self, *args, **kwargs):
            raise RuntimeError(err_msg)

        monkeypatch.setattr(LLM, "call", _tool_err)
        monkeypatch.setattr(LLM, "_handle_non_streaming_response", _tool_err)
        monkeypatch.setattr(LLM, "acall", _async_tool_err)
        monkeypatch.setattr(LLM, "_ahandle_non_streaming_response", _async_tool_err)

        llm = GroqCompatibleLLM(model="groq/qwen/qwen3.8-27b", api_key="test-key")

        res_call = llm.call([{"role": "user", "content": "hi"}])
        assert "unable to perform external searches" in res_call
        assert sleep_calls == []

        res_nonstream = llm._handle_non_streaming_response({})
        assert "unable to perform external searches" in res_nonstream

        res_acall = asyncio.run(llm.acall([{"role": "user", "content": "hi"}]))
        assert "unable to perform external searches" in res_acall

        res_anonstream = asyncio.run(llm._ahandle_non_streaming_response({}))
        assert "unable to perform external searches" in res_anonstream

        def _fatal(self, *args, **kwargs):
            raise RuntimeError("invalid key")

        monkeypatch.setattr(LLM, "call", _fatal)
        with pytest.raises(RuntimeError, match="invalid key"):
            llm.call([{"role": "user", "content": "hi"}])

    def test_create_llm_selects_groq_vs_generic(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        groq_llm = create_llm("groq/qwen/qwen3.8-27b")
        assert isinstance(groq_llm, GroqCompatibleLLM)

        generic_llm = create_llm("openai/gpt-4o")
        assert isinstance(generic_llm, BaseLLM)
        assert not isinstance(generic_llm, GroqCompatibleLLM)

    def test_agents_and_tasks_yaml_integrity_and_crew_assembly(self):
        root = Path(__file__).resolve().parents[1]
        agents_cfg = yaml.safe_load((root / "config" / "agents.yaml").read_text(encoding="utf-8"))
        tasks_cfg = yaml.safe_load((root / "config" / "tasks.yaml").read_text(encoding="utf-8"))

        expected_agents = {
            "pathology_cv_analyst",
            "spatial_bio_analyst",
            "literature_researcher",
            "clinical_report_writer",
        }
        assert expected_agents.issubset(agents_cfg.keys())
        integrity = agents_cfg["scientific_integrity"]
        assert integrity["min_total_cells"] == 30
        assert integrity["min_per_class"] == 5
        assert integrity["confidence_calibration"]["default_temperature"] == 2.0

        expected_tasks = {
            "image_parsing_task",
            "spatial_analysis_task",
            "literature_research_task",
            "pathology_synthesis_task",
        }
        assert expected_tasks == set(tasks_cfg.keys())
        for task_name, cfg in tasks_cfg.items():
            assert cfg["agent"] in expected_agents, f"Task {task_name} references unknown agent"

        mock_llm = MagicMock(spec=LLM)
        mock_llm.model = "groq/qwen/qwen3.8-27b"
        crew_obj = PathologyCrew(llm=mock_llm).crew()
        assert len(crew_obj.agents) == 4
        assert len(crew_obj.tasks) == 4

    def test_strip_cache_breakpoint_edge_cases_and_nested_content(self):
        assert _strip_cache_breakpoint(None) == []
        assert _strip_cache_breakpoint([]) == []

        nested = [
            "plain_string",
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", CACHE_BREAKPOINT_KEY: True},
                    {"type": "text", "text": "world"},
                ],
                CACHE_BREAKPOINT_KEY: True,
            },
        ]
        cleaned = _strip_cache_breakpoint(nested)
        assert cleaned[0] == "plain_string"
        assert CACHE_BREAKPOINT_KEY not in cleaned[1]
        assert CACHE_BREAKPOINT_KEY not in cleaned[1]["content"][0]
        assert cleaned[1]["content"][0] == {"type": "text", "text": "hello"}
        # Original input dict is not mutated
        assert CACHE_BREAKPOINT_KEY in nested[1]
        assert CACHE_BREAKPOINT_KEY in nested[1]["content"][0]

    def test_inject_anti_tool_call_prompt_list_content_and_non_mutating(self):
        original = [{"role": "system", "content": [{"type": "text", "text": "Sys"}]}]
        out1 = _inject_anti_tool_call_prompt(original)
        assert len(original[0]["content"]) == 1  # caller list/dict not mutated
        assert len(out1[0]["content"]) == 2
        assert ANTI_TOOL_CALL_PROMPT in out1[0]["content"][1]["text"]

        out2 = _inject_anti_tool_call_prompt(out1)
        assert len(out2[0]["content"]) == 2  # idempotent for list content

    def test_groq_compatible_llm_acall_rate_limit_retry_and_exhaustion(self, monkeypatch):
        sleep_calls = []

        async def _fake_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr("src.crew.asyncio.sleep", _fake_sleep)

        attempts = {"n": 0}

        async def _flaky_acall(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("RateLimitError: 429")
            return "async_ok"

        monkeypatch.setattr(LLM, "acall", _flaky_acall)
        llm = GroqCompatibleLLM(model="groq/qwen/qwen3.8-27b", api_key="test-key")
        assert asyncio.run(llm.acall([{"role": "user", "content": "hi"}])) == "async_ok"
        assert sleep_calls == [12, 24]

        sleep_calls.clear()

        async def _always_rate_limit(self, *args, **kwargs):
            raise RuntimeError("rate limit exceeded")

        monkeypatch.setattr(LLM, "acall", _always_rate_limit)
        with pytest.raises(RuntimeError, match="rate limit"):
            asyncio.run(llm.acall([{"role": "user", "content": "hi"}]))
        assert len(sleep_calls) >= 3
        assert sleep_calls[:3] == [12, 24, 36]
        assert all(sleep_calls[i : i + 3] == [12, 24, 36] for i in range(0, len(sleep_calls), 3))
