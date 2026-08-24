import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class WandbWriter:
    """Track scalars to a Weights & Biases run.

    Configuration (project, entity, api key, run name, ...) is delegated to the
    standard ``WANDB_*`` environment variables and wandb settings; only the
    log directory is passed through so offline run files land next to the
    other experiment artifacts. ``wandb`` is imported lazily so environments
    without it keep working with the other writers.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
    ):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(dir=str(log_dir) if log_dir is not None else None)

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        self.add_scalars(tag_scalar_dict={tag: scalar_value}, global_step=global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        if not tag_scalar_dict:
            return
        try:
            self._run.log(dict(tag_scalar_dict), step=global_step)
        except Exception:
            logger.exception("wandb log failed; metrics for this step are dropped")

    def close(self):
        self._wandb.finish()
