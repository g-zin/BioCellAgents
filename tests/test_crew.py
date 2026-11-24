"""Offline unit tests for the tool-using 3-agent PathologyCrew and YAML configs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from crewai import LLM, BaseLLM
from crewai.llms.cache import CACHE_BREAKPOINT_KEY

from src.crew import (
    GroqCompatibleLLM,
    PathologyCrew,
    _strip_cache_breakpoint,
    create_llm,
)
from src.reporting import render_deterministic_report
from src.tools.pubmed_search_tool import LiteratureSearchTool
from src.tools.report_verifier_tool import ReportFidelityVerifierTool


class TestCrewHelpersAndToolWiring:
    """Tests for message sanitization, rate-limit retries, LLM factory, and 3-agent tools."""

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
        assert CACHE_BREAKPOINT_KEY in nested[1]
        assert CACHE_BREAKPOINT_KEY in nested[1]["content"][0]

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
        assert sleep_calls[:3] == [12, 24, 36]

        def _fatal(self, *args, **kwargs):
            raise RuntimeError("invalid key")

        monkeypatch.setattr(LLM, "call", _fatal)
        with pytest.raises(RuntimeError, match="invalid key"):
            llm.call([{"role": "user", "content": "hi"}])

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
        assert sleep_calls[:3] == [12, 24, 36]

    def test_create_llm_selects_groq_vs_generic(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        groq_llm = create_llm("groq/qwen/qwen3.8-27b")
        assert isinstance(groq_llm, GroqCompatibleLLM)

        generic_llm = create_llm("openai/gpt-4o")
        assert isinstance(generic_llm, BaseLLM)
        assert not isinstance(generic_llm, GroqCompatibleLLM)

    def test_agents_and_tasks_yaml_integrity_and_3_agent_tool_wiring(self):
        root = Path(__file__).resolve().parents[1]
        agents_cfg = yaml.safe_load((root / "config" / "agents.yaml").read_text(encoding="utf-8"))
        tasks_cfg = yaml.safe_load((root / "config" / "tasks.yaml").read_text(encoding="utf-8"))

        expected_agents = {
            "literature_researcher",
            "clinical_report_writer",
            "fidelity_auditor",
        }
        assert expected_agents.issubset(agents_cfg.keys())
        integrity = agents_cfg["scientific_integrity"]
        assert integrity["min_total_cells"] == 30
        assert integrity["min_per_class"] == 5
        assert integrity["confidence_calibration"]["default_temperature"] == 2.0

        expected_tasks = {
            "literature_research_task",
            "pathology_synthesis_task",
            "fidelity_audit_task",
        }
        assert expected_tasks == set(tasks_cfg.keys())
        for task_name, cfg in tasks_cfg.items():
            assert cfg["agent"] in expected_agents, f"Task {task_name} references unknown agent"

        mock_llm = MagicMock(spec=LLM)
        mock_llm.model = "groq/qwen/qwen3.8-27b"
        crew_obj = PathologyCrew(llm=mock_llm).crew()
        assert len(crew_obj.agents) == 3
        assert len(crew_obj.tasks) == 3

        # Verify genuine tool wiring on literature_researcher and fidelity_auditor
        lit_agent, writer_agent, auditor_agent = crew_obj.agents
        assert any(isinstance(t, LiteratureSearchTool) for t in lit_agent.tools)
        assert writer_agent.tools == []
        assert any(isinstance(t, ReportFidelityVerifierTool) for t in auditor_agent.tools)

    def test_report_fidelity_verifier_tool_run(self):
        tool = ReportFidelityVerifierTool()
        payload = {
            "schema_version": "2.0",
            "organ": {
                "predicted_organ": "Breast",
                "confidence_score": 0.9,
                "calibration_status": {"temperature": 2.0, "fitted": False, "method": "temp"},
            },
            "cells": {
                "total_cell_count": 30,
                "class_counts": {
                    "Neoplastic": 10,
                    "Inflammatory": 6,
                    "Connective": 6,
                    "Dead": 3,
                    "Epithelial": 5,
                },
                "composition_percentages": {
                    "Neoplastic": 33.33,
                    "Inflammatory": 20.0,
                    "Connective": 20.0,
                    "Dead": 10.0,
                    "Epithelial": 16.67,
                },
            },
            "spatial": {
                "graph_topology": {"total_nodes": 30, "total_edges": 40},
                "interaction_matrix": {"Neoplastic-Inflammatory": 10},
                "immune_infiltration": {
                    "mean_immune_to_tumor_dist_px": 25.0,
                    "closest_immune_to_tumor_dist_px": 8.0,
                },
                "validity_assessment": {"is_valid": True, "warnings": [], "recommendations": []},
            },
            "literature_evidence": {"status": "no_results", "results": []},
        }
        report = render_deterministic_report(payload)
        res = tool._run(report_markdown=report, precomputed_results_json=json.dumps(payload))
        assert res["passed"] is True
        assert res["score"] == 100.0

        bad = tool._run(report_markdown="bad", precomputed_results_json="{not json")
        assert bad["passed"] is False
        assert "error" in bad
