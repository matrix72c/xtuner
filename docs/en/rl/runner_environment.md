# Runner and environment ownership

`Runner` composes inference and validation stages. The caller owns an episode's
resources. Training policy, replay and model versions remain with the existing
XTuner loop and manager; neither Runner nor an environment provider decides them.

```python
provider = PoolEnvironmentProvider(pool={
    "type": SandboxPool,
    "provider": provider_config,
    "specs": sandbox_specs,
})
runner = Runner(infer=infer_stage, validate=judger)
try:
    async with provider.open(item) as environment:
        result = await runner.run(item, environment)
        await environment.check()
finally:
    await provider.close()
```

An environment exposes three operations:

- `get(name, record=...)`: return a ready stage client, allocating lazily if needed.
- `info(name)`: validate a name and return immutable image, working-directory,
  environment ID and optional endpoint metadata. This does not allocate resources.
- `check()`: raise on infrastructure failure, including renewal failure for providers
  that renew leases. Failure must not become a completed sample with a zero reward.

`EnvironmentProvider.open(item)` yields a fresh episode environment and owns
cleanup. Providers interpret `item.environment`; Runner treats it as opaque.
Create one provider per worker, drain its episode contexts, then call `close()`.
A future backend needs an environment provider and stage-client adapter, with no
backend dispatch in Runner or Judger.

The built-in `PoolEnvironmentProvider` keeps the existing lazy SandboxPool
allocation/retry behavior. It copies the template and merges the per-item request
into that copy. Live SandboxPool instances are not accepted as templates.
Cleanup failures propagate and retain failed resource IDs for a later close attempt.
The framework closes loops during normal and exceptional trainer shutdown.

## Migration

This is an API migration, with no second Runner implementation:

1. Move `Runner(pool=...)` to `PoolEnvironmentProvider(pool=...)`.
2. Move `item.pipeline_overrides["pool"]` to `item.environment`.
3. Set `AgentInSandboxLoopConfig.environment_provider` to the provider config or
   a module attribute containing that config.
4. For direct execution, use the context shown above and call
   `runner.run(item, environment)`.

The loop retains the environment context until the same episode completes or
is discarded. Do not create resources per trace segment,
reuse one environment across episodes, or reuse provider connections after close.

## Live episodes across model updates

`AgentInSandboxLoopConfig(partial_rollout=True)` retains the environment context
and Runner task in the local loop owner. The harness receives an ephemeral
`item.execution` context: call `checkpoint()` before starting a new phase and
`attach(controller, read_segments)` to install request-boundary control. The
controller accepts `pause`, `resume`, and `status`, and returns active/completed
request counts. `pause` must stop new requests while allowing existing responses
to drain. `read_segments(completed)` returns the recorded transcript branches.
These callbacks carry no sampling or policy-age decisions.

Both existing managers call `prepare_pause()` before global inference abort and
`resume()` after inference continuation. The loop exports exact trace-store token
IDs, masks, and log probabilities; an append-only response journal lets the
existing producer stamp model versions. On final export, each branch inherits
its recorded prefix versions. An expired multi-branch episode is rerun once.
Completed siblings from other episodes are retained.

Limits are explicit: this is live, in-process continuation, not durable recovery.
Do not configure the generic Ray worker router for partial episodes (it has no
owner affinity). A missing owner or changed trace fails the sample. Request drain
has a configured deadline; expiry cancels and reclaims the episode. Ordinary
queued work is cancelled on pause. `close()` drains owned episodes before closing
the provider; call it while Ray and the original asyncio loop are still alive.

Stage records can be written under `artifact_root/<session_id>/`; optional partial
lifecycle events contain session, environment, pause count, and version summaries.
Do not put credentials in stage command text; use structured environment transport.

Failed episodes also write `failure.json` with owner-only permissions (`0600`).
It retains task/session/environment IDs, the error, and the `messages` and
`response_message` artifacts when available. A final trace coverage failure adds
the missing stored branch and token counts so it can be investigated after
environment cleanup. This file contains task/model text; handle it with the same
access and retention policy as training trajectories. It is not copied to shared
logs. Diagnostic write failures are logged and do not replace the original
episode failure or turn it into a reward.
