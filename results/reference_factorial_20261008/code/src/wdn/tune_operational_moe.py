"""Resumable, validation-only Optuna search for the operational MoE."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import optuna

from wdn.train_operational_moe import TrainingConfig, train


def study_signature(cfg):
    manifest = json.loads((Path(cfg.data_dir)/"manifest.json").read_text())
    versions = {}
    root = Path(__file__).parent
    for name in ("train_operational_moe.py", "tune_operational_moe.py", "models/mechanism_moe.py",
                 "models/temporal_moe.py", "models/temporal_multitask.py", "models/specialist_loss.py", "temporal_dataset.py"):
        versions[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    return {"dataset": manifest["config_sha256"], "training": asdict(cfg), "code": versions}


def suggest_config(trial, base, *, smoke=False, local=False):
    """Search model/training parameters only; never change data or split seeds."""
    if smoke:
        return replace(base, lr=trial.suggest_float("lr", 5e-4, 2e-3, log=True))
    width = trial.suggest_categorical("hidden_dim", [16, 24, 32] if local else [16, 32, 48])
    cfg = replace(base, hidden_dim=width,
        dropout=trial.suggest_float("dropout", 0., .2 if local else .25),
        lr=trial.suggest_float("lr", 5e-4 if local else 1e-4, 2e-3 if local else 3e-3, log=True),
        anomaly_pos_weight=trial.suggest_float("anomaly_pos_weight", 5., 20. if local else 30., log=True),
        replay_weight=trial.suggest_float("replay_weight", 1., 3. if local else 4.),
        long_window=trial.suggest_categorical("long_window", [12, 16] if local else [8, 12, 16]))
    if base.model in ("mechanism", "uniform"):
        cfg = replace(cfg,
            short_window=trial.suggest_categorical("short_window", [4] if local else [2, 4]),
            medium_window=trial.suggest_categorical("medium_window", [8, 12] if local else [4, 8]),
            replay_hidden_dim=int(width*trial.suggest_categorical("replay_width_multiplier", [1., 1.5, 2.])),
            drift_hidden_dim=int(width*trial.suggest_categorical("drift_width_multiplier", [1., 1.5, 2.])))
    if base.model != "single":
        cfg = replace(cfg, lambda_expert=trial.suggest_float("lambda_expert", .5 if local else .1, 2. if local else 1.))
    if base.expert_objective == "balanced":
        cfg = replace(cfg, hard_negative_fraction=trial.suggest_categorical(
            "hard_negative_fraction", [.1, .25, .5, 1.]))
    if base.model not in ("single", "uniform"):
        cfg = replace(cfg, lambda_router=trial.suggest_float("lambda_router", 0., .15 if local else .3))
    if base.model == "mechanism":
        cfg = replace(cfg, router_temperature=trial.suggest_float(
            "router_temperature", .75 if local else .5, 1.5 if local else 2.))
    return cfg


def base_parameters(base):
    """All sampled parameters, so an enqueued pilot is reproduced exactly."""
    params = {name: getattr(base, name) for name in (
        "hidden_dim", "dropout", "lr", "anomaly_pos_weight", "replay_weight", "long_window")}
    if base.model in ("mechanism", "uniform"):
        params.update(short_window=base.short_window, medium_window=base.medium_window,
                      replay_width_multiplier=base.replay_hidden_dim/base.hidden_dim,
                      drift_width_multiplier=base.drift_hidden_dim/base.hidden_dim)
    if base.model != "single":
        params["lambda_expert"] = base.lambda_expert
    if base.expert_objective == "balanced":
        params["hard_negative_fraction"] = base.hard_negative_fraction
    if base.model not in ("single", "uniform"):
        params["lambda_router"] = base.lambda_router
    if base.model == "mechanism":
        params["router_temperature"] = base.router_temperature
    return params


def remaining_trials(study, budget):
    # Enqueued WAITING trials are still part of the requested work. Completed,
    # pruned, failed and interrupted RUNNING trials consume the total budget.
    attempted = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    return max(0, budget-attempted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", help="TrainingConfig JSON; explicit CLI values override it")
    parser.add_argument("--trials", type=int, default=30, help="Total trial budget, including previous attempts")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expert_version", type=int, choices=[1, 2])
    parser.add_argument("--model", choices=["mechanism", "homogeneous", "single", "uniform"])
    parser.add_argument("--local", action="store_true", help="Bounded search around the successful V2 pilot")
    parser.add_argument("--enqueue_base", action="store_true", help="Re-run the supplied recipe as the first trial")
    parser.add_argument("--startup_trials", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Tiny network/one epoch; software check only")
    args = parser.parse_args()
    if args.trials < 1 or args.startup_trials < 1:
        parser.error("trials and startup_trials must be positive")
    overrides = json.loads(Path(args.config).read_text()) if args.config else {}
    if overrides.get("evaluate_test", False):
        parser.error("Optuna trials must never evaluate test")
    overrides.update({name: getattr(args, name) for name in ("seed", "model", "epochs")
                      if getattr(args, name) is not None})
    overrides.setdefault("epochs", 20)
    output = Path(args.output_dir).resolve()
    overrides.update(data_dir=str(Path(args.data_dir).resolve()), output_dir=str(output))
    base = TrainingConfig(**overrides)
    if args.smoke:
        base = replace(base, epochs=1)
    expert_version = args.expert_version or (None if args.config else 2)
    if expert_version == 2:
        base = replace(base, feature_version=2 if base.model in ("mechanism", "uniform") else 1,
                       expert_objective="balanced",
                       expert_warmup_epochs=0 if args.smoke or base.model == "single" else min(2, base.epochs-1))
    elif expert_version == 1:
        base = replace(base, feature_version=1, expert_objective="legacy", expert_warmup_epochs=0)
    if args.smoke:
        base = replace(base, hidden_dim=8, replay_hidden_dim=8, drift_hidden_dim=8,
                       router_hidden_dim=8, num_layers=1, short_window=2, medium_window=4,
                       long_window=8, batch_size=16, expert_warmup_epochs=0)
    signature = study_signature(base)
    signature["search"] = {"local": args.local, "startup_trials": args.startup_trials,
                           "enqueue_base": args.enqueue_base}
    if args.enqueue_base:
        if args.smoke:
            parser.error("enqueue_base is intended for full training, not smoke tests")
        # Reject incompatible categorical/range values rather than silently
        # substituting a recipe different from the supplied pilot.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            suggest_config(optuna.trial.FixedTrial(base_parameters(base)), base, local=args.local)
    output.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name="operational_f1_replay", direction="maximize",
        storage=f"sqlite:///{output/'study.sqlite3'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base.seed, n_startup_trials=args.startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))
    previous = study.user_attrs.get("signature")
    if previous is not None and previous != signature:
        raise SystemExit("Study code/data/config changed: use a new output directory")
    study.set_user_attr("signature", signature)
    study.set_user_attr("software_smoke_only", args.smoke)
    if args.enqueue_base and not study.trials:
        study.enqueue_trial(base_parameters(base))

    def objective(trial):
        cfg = suggest_config(trial, base, smoke=args.smoke, local=args.local)
        cfg = replace(cfg, output_dir=str(output/f"trial_{trial.number:04d}"))
        trial.set_user_attr("run_dir", cfg.output_dir)
        result = train(cfg, trial=trial)
        trial.set_user_attr("seconds", result["seconds"])
        return result["validation"]["target_score"]

    study.optimize(objective, n_trials=remaining_trials(study, args.trials))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    frontier = [t for t in completed if not any(
        u.user_attrs["overall_f1"] >= t.user_attrs["overall_f1"] and
        u.user_attrs["replay_f1"] >= t.user_attrs["replay_f1"] and
        (u.user_attrs["overall_f1"] > t.user_attrs["overall_f1"] or
         u.user_attrs["replay_f1"] > t.user_attrs["replay_f1"]) for u in completed)]
    report = {"test_evaluated": False, "software_smoke_only": args.smoke,
              "trial_budget": args.trials, "attempted_trials": len(study.trials),
              "completed_trials": len(completed),
              "best_trial": study.best_trial.number if completed else None,
              "pareto_trials": [{"number": t.number, "score": t.value, **t.user_attrs} for t in frontier]}
    (output/"study_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
