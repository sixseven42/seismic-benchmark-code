"""Public API for the ``utils`` package.

Importing this package also triggers the registration side effects in each
submodule so that their decorators populate the global registries before any
``build_*`` factory is called.
"""

from .datasets import (
    DATASET_REGISTRY,
    BaseArrayDataset,
    as_path,
    build_dataloader,
    build_dataset,
    cap_split_samples,
    register_dataset,
    split_block_indices,
)
from .logger import StepLossLogger, TrainingLogger
from .losses import LOSS_REGISTRY, BaseLoss, build_loss, register_loss
from .eb_wse_metrics import energy_binned_weak_signal_metrics
from .fb_fre_metrics import (
    build_auto_bands,
    compute_average_amplitude_spectrum,
    estimate_effective_band,
    frequency_binned_fidelity_metrics,
)
from .metrics import (
    BAD_FIRST_BREAK_PICK_INDEX,
    METRIC_REGISTRY,
    BaseMetric,
    build_first_break_metrics,
    build_metrics,
    compute_metrics,
    register_metric,
)
from .train_utils import (
    apply_denoise_experiment_name_from_model,
    format_final_test_summary,
    unpack_first_break_batch,
    with_progress,
    write_final_test_metrics,
    barrier_if_distributed,
    build_loaders,
    build_optimizer,
    build_scheduler,
    build_shot_split_loaders,
    compute_length_stats,
    count_parameters,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate,
    evaluate_first_break,
    find_latest_checkpoint,
    format_length_stats,
    init_distributed,
    load_checkpoint,
    load_config,
    maybe_save_best_checkpoint,
    maybe_wrap_ddp,
    resolve_denoise_metrics,
    resolve_repo_root,
    save_checkpoint,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir,
    setup_experiment_dir_distributed,
    train_one_epoch,
    train_one_epoch_first_break,
    training_device,
    unwrap_ddp,
)
from .visualization import (
    plot_loss_curve,
    plot_step_loss_curve,
    plot_sample,
    plot_single_metric_curve,
    visualize_first_break_sample,
    visualize_random_sample,
)

__all__ = [
    # datasets
    "BaseArrayDataset",
    "DATASET_REGISTRY",
    "as_path",
    "build_dataloader",
    "build_dataset",
    "cap_split_samples",
    "register_dataset",
    "split_block_indices",
    # losses
    "BaseLoss",
    "LOSS_REGISTRY",
    "build_loss",
    "register_loss",
    # metrics
    "BAD_FIRST_BREAK_PICK_INDEX",
    "BaseMetric",
    "METRIC_REGISTRY",
    "build_first_break_metrics",
    "build_metrics",
    "compute_metrics",
    "register_metric",
    # energy-binned metrics
    "energy_binned_weak_signal_metrics",
    # frequency-binned metrics
    "compute_average_amplitude_spectrum",
    "estimate_effective_band",
    "build_auto_bands",
    "frequency_binned_fidelity_metrics",
    # visualization
    "plot_loss_curve",
    "plot_step_loss_curve",
    "plot_single_metric_curve",
    "plot_sample",
    "visualize_first_break_sample",
    "visualize_random_sample",
    # logger
    "StepLossLogger",
    "TrainingLogger",
    # training utilities
    "apply_denoise_experiment_name_from_model",
    "format_final_test_summary",
    "unpack_first_break_batch",
    "with_progress",
    "write_final_test_metrics",
    "barrier_if_distributed",
    "build_loaders",
    "build_optimizer",
    "build_scheduler",
    "build_shot_split_loaders",
    "compute_length_stats",
    "count_parameters",
    "default_config_relpath_for_train_script",
    "destroy_distributed",
    "evaluate",
    "evaluate_first_break",
    "find_latest_checkpoint",
    "format_length_stats",
    "init_distributed",
    "load_checkpoint",
    "load_config",
    "maybe_save_best_checkpoint",
    "maybe_wrap_ddp",
    "resolve_denoise_metrics",
    "resolve_repo_root",
    "save_checkpoint",
    "sampler_set_epoch",
    "set_seed",
    "setup_experiment_dir",
    "setup_experiment_dir_distributed",
    "train_one_epoch",
    "train_one_epoch_first_break",
    "training_device",
    "unwrap_ddp",
]
