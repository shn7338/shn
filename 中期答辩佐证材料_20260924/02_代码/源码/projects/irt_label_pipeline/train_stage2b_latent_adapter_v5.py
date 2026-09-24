#!/usr/bin/env python3
"""Train Stage2-B V5 multiscale latent adapters in two optimization phases."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from finetune_pilot_stage2b import (
    atomic_torch_save,
    atomic_write_json,
    load_estimator,
    make_loader,
    run_epoch,
    write_history,
)
from latent_bs_fusion import configured_stage2b_in_channels
from run_sionna_dataset import sha256_file
from stage2_models import LatentAdapterGeo2SigMapUNet, parameter_count
from train_stage2b_directional_ss import (
    build_configured_unet,
    load_frozen_models,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_initial_model(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[LatentAdapterGeo2SigMapUNet, Path, str, dict[str, Any]]:
    checkpoint_path = Path(config["initial_stage2b_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_config = checkpoint["config"]
    input_channels = configured_stage2b_in_channels(config)
    source_channels = configured_stage2b_in_channels(source_config)
    if source_channels != input_channels:
        raise ValueError(
            f"V5 source/input channels differ: {source_channels} != {input_channels}"
        )
    source = build_configured_unet(source_config, in_channels=source_channels).to(device)
    source.load_state_dict(checkpoint["model"], strict=True)
    source.eval().requires_grad_(False)
    model = build_configured_unet(config, in_channels=input_channels).to(device)
    if not isinstance(model, LatentAdapterGeo2SigMapUNet):
        raise TypeError("V5 config did not build LatentAdapterGeo2SigMapUNet")
    model.backbone.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if channels_last:
        source.to(memory_format=torch.channels_last)
        model.to(memory_format=torch.channels_last)

    generator = torch.Generator(device="cpu").manual_seed(20260810)
    probe = torch.randn(
        (1, input_channels, 128, 128), generator=generator, dtype=torch.float32
    ).to(device)
    if channels_last:
        probe = probe.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        source_output = source(probe)
        target_output = model(probe)
    difference = (source_output - target_output).abs()
    equivalence = {
        "source_architecture": type(source).__name__,
        "target_architecture": type(model).__name__,
        "input_channels": input_channels,
        "maximum_absolute_output_difference": float(difference.max().item()),
        "mean_absolute_output_difference": float(difference.mean().item()),
        "exactly_equivalent": bool(torch.equal(source_output, target_output)),
        "adapter_parameters": sum(p.numel() for p in model.adapter_parameters()),
        "total_parameters": parameter_count(model),
    }
    if not equivalence["exactly_equivalent"]:
        raise RuntimeError(f"V5 zero-init equivalence failed: {equivalence}")
    del source, probe, source_output, target_output, difference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, checkpoint_path, sha256_file(checkpoint_path), equivalence


def cosine_lambda(total_epochs: int, minimum_ratio: float):
    def schedule(completed_epochs: int) -> float:
        progress = min(max(completed_epochs, 0), total_epochs) / max(total_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return schedule


def build_phase_optimizer(
    model: LatentAdapterGeo2SigMapUNet,
    phase: dict[str, Any],
    weight_decay: float,
) -> tuple[AdamW, LambdaLR]:
    phase_name = str(phase["name"])
    model.set_optimization_phase(phase_name)
    adapter_lr = float(phase["adapter_learning_rate"])
    groups: list[dict[str, Any]] = [
        {
            "params": list(model.adapter_parameters()),
            "lr": adapter_lr,
            "name": "adapter",
        }
    ]
    lambdas = [
        cosine_lambda(
            int(phase["epochs"]),
            float(phase["adapter_minimum_learning_rate"]) / adapter_lr,
        )
    ]
    if phase_name == "decoder":
        backbone_lr = float(phase["backbone_learning_rate"])
        groups.append(
            {
                "params": list(model.late_backbone_parameters()),
                "lr": backbone_lr,
                "name": "backbone_late",
            }
        )
        lambdas.append(
            cosine_lambda(
                int(phase["epochs"]),
                float(phase["backbone_minimum_learning_rate"]) / backbone_lr,
            )
        )
    optimizer = AdamW(groups, weight_decay=weight_decay)
    return optimizer, LambdaLR(optimizer, lr_lambda=lambdas)


def learning_rates(optimizer: AdamW) -> tuple[float, float]:
    values = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    return values["adapter"], values.get("backbone_late", 0.0)


def checkpoint_payload(
    *,
    model: LatentAdapterGeo2SigMapUNet,
    config: dict[str, Any],
    initial_checkpoint: Path,
    initial_hash: str,
    estimator_checkpoint: Path,
    estimator_hash: str,
    frozen: dict[str, Any],
    global_epoch: int,
    phase_index: int,
    phase_name: str,
    phase_epoch: int,
    best_rmse: float,
    best_epoch: int,
    best_phase: str,
    best_val_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "model": model.state_dict(),
        "global_epoch": global_epoch,
        "phase_index": phase_index,
        "phase": phase_name,
        "phase_epoch": phase_epoch,
        "best_val_rmse_db": best_rmse,
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_val_metrics": best_val_metrics,
        "config": config,
        "initial_stage2b_checkpoint": str(initial_checkpoint),
        "initial_stage2b_checkpoint_sha256": initial_hash,
        "estimator_checkpoint": str(estimator_checkpoint),
        "estimator_checkpoint_sha256": estimator_hash,
        "stage1_checkpoint_sha256": frozen["stage1_checkpoint_sha256"],
        "stage2a_checkpoint_sha256": frozen["stage2a_checkpoint_sha256"],
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    normalization = load_json(Path(config["irt_normalization"]))
    iso_stats = normalization["statistics"]["train"]["p_iso"]
    ss_stats = normalization["stage2b_ss_normalization"]
    iso_mean_db, iso_std_db = float(iso_stats["mean_db"]), float(iso_stats["std_db"])
    ss_mean_db, ss_std_db = float(ss_stats["mean_db"]), float(ss_stats["std_db"])
    datasets = {
        split: BSInversionPilotDataset(
            config["dataset_root"],
            split=split,
            seed=int(config["seed"]),
            augment=split == "train" and bool(config["augmentation"]),
            signal_mean_db=ss_mean_db,
            signal_std_db=ss_std_db,
        )
        for split in ("train", "val")
    }
    loaders = {
        split: make_loader(dataset, config, shuffle=split == "train", device=device)
        for split, dataset in datasets.items()
    }
    stage1, stage2a, frozen = load_frozen_models(config, device, channels_last)
    estimator, estimator_metadata, estimator_checkpoint = load_estimator(
        config, device, channels_last
    )
    model, initial_checkpoint, initial_hash, equivalence = load_initial_model(
        config, device, channels_last
    )

    tx_normalization = load_json(
        Path(config["stage1_model_dir"]) / "tx_feature_normalization.json"
    )
    tx_height_cap_m = float(tx_normalization["transmitter_height"]["cap_m"])
    distance_max_m = float(tx_normalization["distance"]["max_m"])
    long_distance_max_m = float(
        tx_normalization.get("long_distance", {}).get(
            "max_m", math.sqrt(2.0) * 512.0
        )
    )
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and args.resume is None:
        raise FileExistsError(f"{best_path} already exists; choose a new V5 output")

    source_checkpoint = torch.load(
        initial_checkpoint, map_location="cpu", weights_only=False
    )
    best_rmse = float(source_checkpoint["best_val_rmse_db"])
    best_val_metrics = dict(source_checkpoint.get("best_val_metrics", {}))
    best_epoch = 0
    best_phase = "initial_v4_equivalent"
    global_epoch = 0
    history: list[dict[str, Any]] = []
    resume_data: dict[str, Any] | None = None
    if args.resume is not None:
        resume_data = torch.load(
            args.resume.resolve(), map_location=device, weights_only=False
        )
        model.load_state_dict(resume_data["model"], strict=True)
        history = list(resume_data["history"])
        global_epoch = int(resume_data["global_epoch"])
        best_rmse = float(resume_data["best_val_rmse_db"])
        best_epoch = int(resume_data["best_epoch"])
        best_phase = str(resume_data["best_phase"])
        best_val_metrics = dict(resume_data.get("best_val_metrics", {}))
    else:
        atomic_torch_save(
            best_path,
            checkpoint_payload(
                model=model,
                config=config,
                initial_checkpoint=initial_checkpoint,
                initial_hash=initial_hash,
                estimator_checkpoint=estimator_checkpoint,
                estimator_hash=estimator_metadata["checkpoint_sha256"],
                frozen=frozen,
                global_epoch=0,
                phase_index=-1,
                phase_name=best_phase,
                phase_epoch=0,
                best_rmse=best_rmse,
                best_epoch=0,
                best_phase=best_phase,
                best_val_metrics=best_val_metrics,
            ),
        )

    run_config = {
        **config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "initial_stage2b_checkpoint_sha256": initial_hash,
        "estimator_checkpoint_sha256": estimator_metadata["checkpoint_sha256"],
        "split_samples": {key: len(value) for key, value in datasets.items()},
        "zero_init_equivalence": equivalence,
        **frozen,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)

    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    started = time.perf_counter()
    phases = list(config["training_phases"])
    resume_phase_index = int(resume_data["phase_index"]) if resume_data else -1
    for phase_index, phase in enumerate(phases):
        if resume_data is not None and phase_index < resume_phase_index:
            continue
        phase_name = str(phase["name"])
        resuming_this_phase = resume_data is not None and phase_index == resume_phase_index
        if phase_index > 0 and not resuming_this_phase:
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint["model"], strict=True)
        optimizer, scheduler = build_phase_optimizer(
            model, phase, float(config["weight_decay"])
        )
        phase_start_epoch = 1
        phase_patience = 0
        if resuming_this_phase:
            optimizer.load_state_dict(resume_data["optimizer"])
            scheduler.load_state_dict(resume_data["scheduler"])
            if resume_data.get("scaler"):
                scaler.load_state_dict(resume_data["scaler"])
            phase_start_epoch = int(resume_data["phase_epoch"]) + 1
            phase_patience = int(resume_data.get("phase_patience", 0))
        for phase_epoch in range(phase_start_epoch, int(phase["epochs"]) + 1):
            global_epoch += 1
            datasets["train"].set_epoch(global_epoch)
            adapter_lr, backbone_lr = learning_rates(optimizer)
            train_metrics = run_epoch(
                loaders["train"], estimator, stage1, stage2a, model,
                device=device, amp_enabled=amp_enabled, channels_last=channels_last,
                frozen=frozen,
                estimator_signal_mean_db=float(estimator_metadata["signal_mean_db"]),
                estimator_signal_std_db=float(estimator_metadata["signal_std_db"]),
                estimator_power_mean_db=float(estimator_metadata["power_mean_db"]),
                estimator_power_std_db=float(estimator_metadata["power_std_db"]),
                iso_mean_db=iso_mean_db, iso_std_db=iso_std_db,
                ss_mean_db=ss_mean_db, ss_std_db=ss_std_db,
                tx_height_cap_m=tx_height_cap_m, distance_max_m=distance_max_m,
                stage1_in_channels=int(frozen.get("stage1_in_channels", 3)),
                long_distance_max_m=long_distance_max_m,
                optimizer=optimizer, scaler=scaler,
                gradient_clip=float(config["gradient_clip"]),
                progress_every=int(config.get("progress_every_batches", 0)),
                latent_fusion_config=config["latent_fusion"],
                phase=f"train_{phase_name}", epoch=global_epoch,
            )
            with torch.inference_mode():
                val_metrics = run_epoch(
                    loaders["val"], estimator, stage1, stage2a, model,
                    device=device, amp_enabled=amp_enabled, channels_last=channels_last,
                    frozen=frozen,
                    estimator_signal_mean_db=float(estimator_metadata["signal_mean_db"]),
                    estimator_signal_std_db=float(estimator_metadata["signal_std_db"]),
                    estimator_power_mean_db=float(estimator_metadata["power_mean_db"]),
                    estimator_power_std_db=float(estimator_metadata["power_std_db"]),
                    iso_mean_db=iso_mean_db, iso_std_db=iso_std_db,
                    ss_mean_db=ss_mean_db, ss_std_db=ss_std_db,
                    tx_height_cap_m=tx_height_cap_m, distance_max_m=distance_max_m,
                    stage1_in_channels=int(frozen.get("stage1_in_channels", 3)),
                    long_distance_max_m=long_distance_max_m,
                    optimizer=None, scaler=None,
                    gradient_clip=float(config["gradient_clip"]),
                    progress_every=int(config.get("progress_every_batches", 0)),
                    latent_fusion_config=config["latent_fusion"],
                    phase=f"val_{phase_name}", epoch=global_epoch,
                )
            row = {
                "epoch": global_epoch,
                "phase": phase_name,
                "phase_epoch": phase_epoch,
                "adapter_learning_rate": adapter_lr,
                "backbone_learning_rate": backbone_lr,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            write_history(output_dir / "history.csv", history)
            val_rmse = float(val_metrics["map_rmse_db"])
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_epoch = global_epoch
                best_phase = phase_name
                best_val_metrics = val_metrics
                phase_patience = 0
                atomic_torch_save(
                    best_path,
                    checkpoint_payload(
                        model=model, config=config,
                        initial_checkpoint=initial_checkpoint, initial_hash=initial_hash,
                        estimator_checkpoint=estimator_checkpoint,
                        estimator_hash=estimator_metadata["checkpoint_sha256"],
                        frozen=frozen, global_epoch=global_epoch,
                        phase_index=phase_index, phase_name=phase_name,
                        phase_epoch=phase_epoch, best_rmse=best_rmse,
                        best_epoch=best_epoch, best_phase=best_phase,
                        best_val_metrics=best_val_metrics,
                    ),
                )
            else:
                phase_patience += 1
            scheduler.step()
            last_payload = checkpoint_payload(
                model=model, config=config,
                initial_checkpoint=initial_checkpoint, initial_hash=initial_hash,
                estimator_checkpoint=estimator_checkpoint,
                estimator_hash=estimator_metadata["checkpoint_sha256"],
                frozen=frozen, global_epoch=global_epoch,
                phase_index=phase_index, phase_name=phase_name,
                phase_epoch=phase_epoch, best_rmse=best_rmse,
                best_epoch=best_epoch, best_phase=best_phase,
                best_val_metrics=best_val_metrics,
            )
            last_payload.update(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "phase_patience": phase_patience,
                }
            )
            atomic_torch_save(output_dir / "last.pt", last_payload)
            print(json.dumps({
                "event": "epoch", "epoch": global_epoch, "phase": phase_name,
                "phase_epoch": phase_epoch,
                "train_rmse_db": train_metrics["map_rmse_db"],
                "val_rmse_db": val_rmse, "best_val_rmse_db": best_rmse,
                "best_epoch": best_epoch, "best_phase": best_phase,
                "phase_patience": phase_patience,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }, ensure_ascii=False), flush=True)
            if phase_patience >= int(phase["patience"]):
                break
        resume_data = None

    summary = {
        "version": 1,
        "status": "ok",
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_val_rmse_db": best_rmse,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "zero_init_equivalence": equivalence,
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
