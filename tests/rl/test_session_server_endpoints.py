import json
import unittest
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout.session_server import (
    FMT_ANTHROPIC,
    FMT_OPENAI,
    SessionServer,
    _classify_endpoint,
    _detect_format,
)


class TestEndpointClassification(unittest.TestCase):
    def test_supported_endpoints(self):
        cases = [
            ("POST", "v1/messages", "anthropic_messages"),
            ("POST", "/proxy/v1/messages?beta=true", "anthropic_messages"),
            ("POST", "v1/messages/count_tokens", "anthropic_count_tokens"),
            ("POST", "/v1/chat/completions", "openai_chat_completions"),
            ("POST", "prefix/v1/responses", "openai_responses"),
            ("HEAD", "/api/hello", "hello"),
        ]
        for method, path, expected in cases:
            with self.subTest(method=method, path=path):
                self.assertEqual(_classify_endpoint(method, path), expected)

    def test_unknown_paths_and_wrong_methods_are_rejected(self):
        cases = [
            ("GET", "/v1/models"),
            ("GET", "/v1/messages"),
            ("POST", "/terminate"),
            ("POST", "/update_weights"),
            ("POST", "/sleep"),
            ("POST", "/wakeup"),
        ]
        for method, path in cases:
            with self.subTest(method=method, path=path):
                self.assertIsNone(_classify_endpoint(method, path))

    def test_count_tokens_uses_anthropic_format(self):
        self.assertEqual(_detect_format("v1/messages/count_tokens"), FMT_ANTHROPIC)
        self.assertEqual(_detect_format("v1/chat/completions"), FMT_OPENAI)


class TestSessionServerEndpointHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_calls = []
        self.upstream_body = b'{"input_tokens": 42}'

        async def upstream_handler(request):
            self.upstream_calls.append(
                {
                    "method": request.method,
                    "path": request.path,
                    "query": request.query_string,
                    "body": await request.read(),
                }
            )
            return web.Response(status=200, content_type="application/json", body=self.upstream_body)

        upstream_app = web.Application()
        upstream_app.router.add_route("*", "/{path:.*}", upstream_handler)
        self.upstream_server = TestServer(upstream_app)
        await self.upstream_server.start_server()

        self.session_server = SessionServer.__new__(SessionServer)
        self.session_server.worker_base_url = str(self.upstream_server.make_url("")).rstrip("/")
        self.session_server.request_timeout = 10.0
        self.session_server.read_bufsize = 2**20
        self.session_server.stop_word = "<eos>"
        self.session_server.on_request = AsyncMock(side_effect=lambda body, _fmt, **_kwargs: body)
        self.session_server.on_response = AsyncMock(return_value=None)

        proxy_app = web.Application()
        proxy_app.router.add_route("*", "/{path:.*}", self.session_server._handle_request)
        self.proxy_client = TestClient(TestServer(proxy_app))
        await self.proxy_client.start_server()

    async def asyncTearDown(self):
        await self.proxy_client.close()
        await self.upstream_server.close()

    async def test_count_tokens_is_transparent_and_untraced(self):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "system": "system",
            "tools": [{"name": "tool", "input_schema": {"type": "object"}}],
            "session_id": "trace-session",
        }

        response = await self.proxy_client.post("/v1/messages/count_tokens?beta=true", json=payload)

        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b'{"input_tokens": 42}')
        self.assertEqual(len(self.upstream_calls), 1)
        forwarded = json.loads(self.upstream_calls[0]["body"])
        self.assertEqual(forwarded, {key: value for key, value in payload.items() if key != "session_id"})
        self.assertEqual(self.upstream_calls[0]["query"], "beta=true")
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_generation_still_runs_both_hooks(self):
        self.upstream_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "output_ids": [1],
                        "output_token_logprobs": [[-0.1, 1]],
                    }
                ]
            }
        ).encode()
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "trace-session",
            "return_token_ids": True,
            "return_logprob": True,
        }

        response = await self.proxy_client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status, 200)
        self.session_server.on_request.assert_awaited_once()
        self.session_server.on_response.assert_awaited_once()

    async def test_hello_and_unknown_endpoints_never_reach_upstream(self):
        hello = await self.proxy_client.head("/api/hello")
        unknown = await self.proxy_client.post("/terminate", json={})
        wrong_method = await self.proxy_client.get("/v1/messages")
        responses = await self.proxy_client.post("/v1/responses", json={})

        self.assertEqual(hello.status, 204)
        self.assertEqual(unknown.status, 404)
        self.assertEqual(wrong_method.status, 404)
        self.assertEqual(responses.status, 501)
        self.assertEqual(self.upstream_calls, [])


if __name__ == "__main__":
    unittest.main()
