import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


def _run_trace_utils(repo_root: Path, command: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.fspath(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, os.fspath(Path(__file__).with_name("trace_utils.py")), command],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTrace(unittest.TestCase):
    def test_external_collector_skips_local_collector_and_propagates_endpoint(self):
        from xtuner.v1.rl.trace import runtime as trace_runtime

        class Provider:
            def shutdown(self):
                return None

        with TemporaryDirectory() as temp_dir:
            config = trace_runtime.TraceConfig(
                enabled=True,
                output_dir=temp_dir,
                external_otlp_endpoint="http://otel-collector.namespace.svc:4317",
            )
            handle = trace_runtime._build_trace_runtime_handle(config)

            self.assertFalse(handle.start_local_collector)
            self.assertIsNone(handle.collector_port)
            self.assertIsNone(handle.runtime.trace_jsonl_path)
            self.assertEqual(
                handle.env_vars["OTEL_EXPORTER_OTLP_ENDPOINT"],
                "http://otel-collector.namespace.svc:4317",
            )
            self.assertNotIn("XTUNER_OTEL_JSONL_PATH", handle.env_vars)

            with (
                mock.patch.object(trace_runtime._OTelCollector, "start") as start_collector,
                mock.patch.object(trace_runtime, "_configure_tracer_provider", return_value=Provider()),
            ):
                handle.start()
                handle.close()
            start_collector.assert_not_called()
            trace_runtime.clear_trace_env()

    def test_local_collector_remains_the_default(self):
        from xtuner.v1.rl.trace import runtime as trace_runtime

        with TemporaryDirectory() as temp_dir:
            with mock.patch.object(trace_runtime, "find_free_ports", return_value=[4317]):
                handle = trace_runtime._build_trace_runtime_handle(
                    trace_runtime.TraceConfig(enabled=True, output_dir=temp_dir)
                )

            self.assertTrue(handle.start_local_collector)
            self.assertIsNotNone(handle.collector_port)
            self.assertIsNotNone(handle.runtime.trace_jsonl_path)
            self.assertTrue(handle.runtime.trace_jsonl_path.is_file())
            self.assertTrue(handle.endpoint.startswith("http://127.0.0.1:"))

    def test_external_trace_jsonl_is_shared_with_viewer_and_ray_children(self):
        from xtuner.v1.rl.trace import runtime as trace_runtime

        with TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "shared" / "traces.jsonl"
            handle = trace_runtime._build_trace_runtime_handle(
                trace_runtime.TraceConfig(
                    enabled=True,
                    output_dir=Path(temp_dir) / "runs",
                    external_otlp_endpoint="http://otel-collector.namespace.svc:4317",
                    external_trace_jsonl_path=trace_path,
                    xtuner_viewer_enabled=True,
                )
            )

            self.assertEqual(handle.runtime.trace_jsonl_path, trace_path)
            self.assertEqual(handle.env_vars["XTUNER_OTEL_JSONL_PATH"], os.fspath(trace_path))
            self.assertTrue(trace_path.is_file())

    def test_external_viewer_requires_shared_trace_jsonl(self):
        from pydantic import ValidationError

        from xtuner.v1.rl.trace import runtime as trace_runtime

        with self.assertRaisesRegex(ValidationError, "external_trace_jsonl_path"):
            trace_runtime.TraceConfig(
                enabled=True,
                external_otlp_endpoint="http://otel-collector.namespace.svc:4317",
                xtuner_viewer_enabled=True,
            )

    def test_external_trace_jsonl_requires_external_endpoint(self):
        from pydantic import ValidationError

        from xtuner.v1.rl.trace import runtime as trace_runtime

        with self.assertRaisesRegex(ValidationError, "external_otlp_endpoint"):
            trace_runtime.TraceConfig(
                enabled=True,
                external_trace_jsonl_path="/shared/traces.jsonl",
            )

    def test_trace_span_records_attributes_events_and_errors(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "record-span")

        self.assertEqual(output["success_attributes"]["xtuner.stage"], "unit")
        self.assertEqual(output["success_attributes"]["unit.count"], 1)
        self.assertEqual(output["success_events"], ["unit.event"])
        self.assertEqual(output["failure_status"], "ERROR")
        self.assertEqual(output["failure_attributes"]["error"], True)
        self.assertEqual(output["failure_attributes"]["error.type"], "RuntimeError")
        self.assertEqual(output["failure_attributes"]["error.message"], "boom")

    def test_injected_parent_carrier_links_child_span_in_another_process(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "parent-child")

        self.assertEqual(output["child"]["trace_id"], output["parent_trace_id"])
        self.assertEqual(output["child"]["parent_span_id"], output["parent_span_id"])

    def test_nested_trace_span_preserves_parent_to_child_order(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "nested-span-order")

        self.assertEqual(output["child_parent_span_id"], output["parent_span_id"])
        self.assertEqual(output["span_name_paths"]["order.parent"], ["order.parent"])
        self.assertEqual(output["span_name_paths"]["order.child"], ["order.parent", "order.child"])

    def test_synthetic_spans_preserve_historical_times_and_parentage(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "synthetic-spans")

        self.assertEqual(output["root_ids"]["span_id"], output["child_parent_span_id"])
        self.assertEqual(output["root_ids"]["trace_id"], output["child_ids"]["trace_id"])
        self.assertEqual(output["root_start_time"], 1_000_000_000)
        self.assertEqual(output["root_end_time"], 3_000_000_000)
        self.assertEqual(output["child_status"], "ERROR")
        self.assertTrue(output["child_attributes"]["xtuner.synthetic"])
        self.assertEqual(output["child_attributes"]["error.message"], "tool failed")

    def test_synthetic_span_validates_interval_before_runtime_setup(self):
        from xtuner.v1.rl.trace import api as trace_api

        with self.assertRaisesRegex(ValueError, "greater than"):
            trace_api.record_synthetic_span(
                "invalid.interval",
                start_time_unix_ns=2,
                end_time_unix_ns=1,
            )

    def test_rollout_remote_propagates_and_cleans_batch_carriers(self):
        from xtuner.v1.data_proto.rl_data import RolloutState
        from xtuner.v1.rl.trace import rollout_api

        states = [RolloutState(message=[], rollout_id=index) for index in (1, 2)]
        observed_carriers = []

        class RemoteMethod:
            def remote(self, rollout_states):
                observed_carriers.extend(
                    dict(state.extra_fields[rollout_api.TRACE_CARRIER_EXTRA_FIELD]) for state in rollout_states
                )
                return "object-ref"

        with (
            mock.patch.object(rollout_api, "is_rollout_trace_enabled", return_value=True),
            mock.patch.object(
                rollout_api.trace_api,
                "inject_trace_context",
                return_value={"traceparent": "00-trace-span-01"},
            ),
        ):
            result = rollout_api.trace_rollout_remote(
                RemoteMethod(),
                states,
                target=states,
            )

        self.assertEqual(result, "object-ref")
        self.assertEqual(
            observed_carriers,
            [{"traceparent": "00-trace-span-01"}, {"traceparent": "00-trace-span-01"}],
        )
        self.assertTrue(all(rollout_api.TRACE_CARRIER_EXTRA_FIELD not in state.extra_fields for state in states))

    def test_viewer_uses_span_name_path_for_display_chain(self):
        from recipe.trace.viewer.payload import build_rollout_view_payload_from_jaeger_traces

        traces = [
            {
                "traceID": "trace-1",
                "processes": {"p1": {"serviceName": "xtuner-test", "tags": []}},
                "spans": [
                    {
                        "traceID": "trace-1",
                        "spanID": "span-1",
                        "operationName": "parent.phase",
                        "processID": "p1",
                        "startTime": 1_000,
                        "duration": 2_000,
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-1"},
                            {"key": "xtuner.span_name_path", "value": ["parent.phase"]},
                        ],
                    },
                    {
                        "traceID": "trace-1",
                        "spanID": "span-2",
                        "operationName": "child.phase",
                        "processID": "p1",
                        "startTime": 2_000,
                        "duration": 1_000,
                        "references": [{"refType": "CHILD_OF", "traceID": "trace-1", "spanID": "span-1"}],
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-1"},
                            {"key": "xtuner.span_name_path", "value": ["parent.phase", "child.phase"]},
                        ],
                    },
                ],
            }
        ]

        payload = build_rollout_view_payload_from_jaeger_traces(traces, train_step="all")

        self.assertEqual(
            [node["name"] for node in payload["samples"][0]["display_path"]],
            ["parent.phase", "child.phase"],
        )
        self.assertEqual(payload["samples"][0]["chain"], "parent.phase -> child.phase")

    def test_viewer_filters_latest_train_step_and_renders_payload(self):
        from recipe.trace.viewer.payload import build_rollout_view_payload_from_jaeger_traces
        from recipe.trace.viewer.render import render_rollout_trace_html

        traces = [
            {
                "traceID": "trace-1",
                "processes": {"p1": {"serviceName": "xtuner-test", "tags": []}},
                "spans": [
                    {
                        "traceID": "trace-1",
                        "spanID": "span-1",
                        "operationName": "old.operation",
                        "processID": "p1",
                        "startTime": 1_000,
                        "duration": 1_000,
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-1"},
                            {"key": "xtuner.producer_future_step", "value": 1},
                            {"key": "xtuner.stage", "value": "stage_one"},
                        ],
                    }
                ],
            },
            {
                "traceID": "trace-2",
                "processes": {"p1": {"serviceName": "xtuner-test", "tags": []}},
                "spans": [
                    {
                        "traceID": "trace-2",
                        "spanID": "span-2",
                        "operationName": "new.operation",
                        "processID": "p1",
                        "startTime": 2_000,
                        "duration": 1_000,
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-2"},
                            {"key": "xtuner.producer_future_step", "value": 2},
                            {"key": "xtuner.stage", "value": "stage_two"},
                        ],
                    }
                ],
            },
        ]

        payload = build_rollout_view_payload_from_jaeger_traces(traces)
        html = render_rollout_trace_html(payload)

        self.assertEqual(payload["selected_train_step"], 2)
        self.assertEqual(payload["available_train_steps"], [1, 2])
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["samples"][0]["rollout_id"], "rollout-2")
        self.assertEqual(payload["samples"][0]["stage"], "stage_two")
        self.assertIn("XTuner Rollout Trace Viewer", html)
        self.assertIn("stage_two", html)


class TestSessionServerTrace(unittest.IsolatedAsyncioTestCase):
    async def test_request_span_extracts_parent_and_records_status(self):
        from xtuner.v1.rl.rollout import session_server

        server = object.__new__(session_server.SessionServer)
        request = types.SimpleNamespace(
            match_info={"path": "v1/chat/completions"},
            headers={"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"},
            method="POST",
        )
        response = types.SimpleNamespace(status=201)
        observed = {}

        @contextmanager
        def capture_span(name, attributes=None, *, parent_carrier=None):
            observed.update(
                name=name,
                attributes=dict(attributes or {}),
                parent_carrier=dict(parent_carrier or {}),
            )
            yield

        with (
            mock.patch.object(session_server, "trace_span", side_effect=capture_span),
            mock.patch.object(session_server, "set_trace_attributes") as set_attributes,
            mock.patch.object(
                session_server.SessionServer,
                "_handle_request_impl",
                new=mock.AsyncMock(return_value=response),
            ),
        ):
            result = await server._handle_request(request)

        self.assertIs(result, response)
        self.assertEqual(observed["name"], "session_server.request")
        self.assertEqual(observed["parent_carrier"], request.headers)
        set_attributes.assert_called_once_with({"http.response.status_code": 201, "error": False})

    async def test_send_request_injects_current_context(self):
        from xtuner.v1.rl.rollout import session_server

        server = object.__new__(session_server.SessionServer)
        response = types.SimpleNamespace(status=200)
        forwarded = {}

        class RequestContext:
            async def __aenter__(self):
                return response

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Client:
            def request(self, **kwargs):
                forwarded.update(kwargs)
                return RequestContext()

        @contextmanager
        def passthrough_span(*args, **kwargs):
            yield

        def inject(headers):
            headers["traceparent"] = "injected"
            return headers

        headers = {}
        with (
            mock.patch.object(session_server, "trace_span", side_effect=passthrough_span),
            mock.patch.object(session_server, "inject_trace_context", side_effect=inject),
        ):
            async with server._send_request(Client(), method="POST", url="http://worker", headers=headers):
                pass

        self.assertEqual(headers["traceparent"], "injected")
        self.assertIs(forwarded["headers"], headers)


class TestSandboxTraceBridge(unittest.TestCase):
    def test_legacy_span_is_preserved_and_sensitive_annotations_stay_out_of_otel(self):
        trace_path = Path(__file__).resolve().parents[2] / "xtuner/v1/rl/agent_loop/sandbox_agent_loop/trace.py"
        spec = importlib.util.spec_from_file_location("sandbox_trace_test", trace_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        sandbox_trace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sandbox_trace)

        observed_spans = []
        observed_final_attributes = []

        @contextmanager
        def capture_span(name, attributes=None):
            observed_spans.append((name, dict(attributes or {})))
            yield

        with TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {"WORK_DIR": temp_dir}):
            sandbox_trace._reset_for_testing()
            sandbox_trace.init_writer(actor_id="test")
            with (
                mock.patch.object(sandbox_trace, "trace_span", side_effect=capture_span),
                mock.patch.object(
                    sandbox_trace,
                    "set_trace_attributes",
                    side_effect=lambda attrs: observed_final_attributes.append(dict(attrs)),
                ),
            ):
                with sandbox_trace.span("session-1", "acquire", task_id="task-1") as handle:
                    handle.annotate(
                        sandbox_name="default",
                        sandbox_image="sandbox:latest",
                        sandbox_url="http://secret.internal/sandbox",
                    )
            sandbox_trace._reset_for_testing()

            legacy_files = list((Path(temp_dir) / "trace").glob("spans.*.jsonl"))
            self.assertEqual(len(legacy_files), 1)
            legacy_records = [json.loads(line) for line in legacy_files[0].read_text().splitlines()]

        self.assertEqual(observed_spans[0][0], "sandbox.acquire")
        self.assertEqual(observed_spans[0][1]["xtuner.session_id"], "session-1")
        self.assertEqual(observed_final_attributes[0]["sandbox.sandbox_name"], "default")
        self.assertNotIn("sandbox.sandbox_url", observed_final_attributes[0])
        self.assertEqual([record["event"] for record in legacy_records], ["enter", "exit"])
        self.assertEqual(legacy_records[-1]["sandbox_url"], "http://secret.internal/sandbox")


if __name__ == "__main__":
    unittest.main()
