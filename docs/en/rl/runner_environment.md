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
The legacy pool currently logs provider deletion failures; this extraction does
not upgrade that backend to durable or retryable cleanup. Backends with explicit
cleanup reports should raise failed cleanup through their context or `close()`.

Stage clients may implement `execute_with_env(command, cwd, timeout, detach,
env=...)` for structured environment transport. This avoids credentials in shell
argv and preserves per-entry environment scope. Existing clients keep the legacy
`execute` transport. Stage hooks and entries remain unchanged.

## Migration

This is an API migration, with no second Runner implementation:

1. Move `Runner(pool=...)` to `PoolEnvironmentProvider(pool=...)`.
2. Move `item.pipeline_overrides["pool"]` to `item.environment`.
3. Set `AgentInSandboxLoopConfig.environment_provider` to the provider config or
   a module attribute containing that config.
4. For direct execution, use the context shown above and call
   `runner.run(item, environment)`.
5. Treat `Status.FAILED` / missing reward as infrastructure or execution failure in
   evaluation. Only a completed verifier result supplies a task score.

Environment extraction alone does not implement live partial rollout. An owner
that retains an episode across pauses must keep its context open until that
same episode completes or is discarded. Do not create resources per trace segment,
reuse one environment across episodes, or reuse provider connections after close.
