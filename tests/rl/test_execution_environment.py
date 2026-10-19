"""Exercise environment transport against a real local shell, without a backend."""

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import exec_in


class ShellClient:
    async def execute(self, command, cwd, timeout, detach):
        result = subprocess.run(["/bin/sh", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=5)
        return dict(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def test_shell_values_are_data_and_visible_to_children(tmp_path):
    marker = tmp_path / "must-not-exist"
    value = f"spaces ' \" $HOME `touch {marker}` $(touch {marker})\nnext line"
    env = {"REVIEW_VALUE": value}
    command = 'printf \'%s\' "$REVIEW_VALUE" && sh -c \'printf "%s" "$REVIEW_VALUE"\''
    result = asyncio.run(exec_in(ShellClient(), command, cwd=str(tmp_path), env=env))
    assert result["stdout"] == value * 2
    assert not marker.exists()
    assert env == {"REVIEW_VALUE": value}
    assert "REVIEW_VALUE" not in os.environ


def test_structured_transport_keeps_command_and_environment_separate():
    client = AsyncMock()
    client.execute_with_env.return_value = {"return_code": 0}
    env = {"TOKEN": "secret'$(false)"}
    asyncio.run(exec_in(client, 'echo "$TOKEN"', cwd="/work", timeout_sec=17, env=env, detach=True))
    client.execute_with_env.assert_awaited_once_with('echo "$TOKEN"', "/work", 17, True, env=env)
    client.execute.assert_not_called()


@pytest.mark.parametrize("env", [{"X; false": "value"}, {"1X": "value"}, {"X": "a\0b"}, {"X": 1}])
def test_invalid_environment_never_reaches_backend(env):
    client = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(exec_in(client, "true", env=env))
    client.execute_with_env.assert_not_called()
    client.execute.assert_not_called()


def test_failure_redacts_injected_values_but_preserves_return_code(tmp_path):
    secret = "review-secret-value"
    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            exec_in(ShellClient(), 'printf "%s" "$TOKEN" >&2; exit 7', cwd=str(tmp_path), env={"TOKEN": secret})
        )
    assert secret not in str(error.value)
    assert "return_code=7" in str(error.value)
    assert "<redacted>" in str(error.value)


def test_values_do_not_persist_between_entries(tmp_path):
    async def run():
        client = ShellClient()
        await exec_in(client, "true", cwd=str(tmp_path), env={"REVIEW_LOCAL": "only-first"})
        return await exec_in(client, 'printf "%s" "${REVIEW_LOCAL-unset}"', cwd=str(tmp_path))

    assert asyncio.run(run())["stdout"] == "unset"
