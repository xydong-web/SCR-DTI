from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scrdti.calibration import apply_temperature, fit_temperature
from scrdti.config import load_config
from scrdti.data import load_feature_npz, make_synthetic_npz
from scrdti.evaluate import (
    binary_metrics,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from scrdti.precompute import DEFAULT_ENCODERS, encode_texts
from scrdti.preprocessing import canonicalize_smiles, normalize_protein_sequence
from scrdti.training import (
    load_release_checkpoint,
    predict_arrays,
    result_to_json,
    train_from_npz,
)


def _cmd_make_synthetic(args: argparse.Namespace) -> None:
    path = make_synthetic_npz(args.output, n=args.n, seed=args.seed)
    print(path)


def _cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = train_from_npz(config, args.data, args.output, device=args.device)
    print(result_to_json(result))


def _cmd_predict(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model, _ = load_release_checkpoint(args.checkpoint, map_location=args.device)
    if model.config != config.model:
        raise SystemExit("checkpoint model configuration does not match --config")
    arrays = load_feature_npz(args.data)
    predictions = predict_arrays(
        model,
        arrays,
        split_id=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    frame = pd.DataFrame(predictions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(args.output)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.predictions)
    metrics = binary_metrics(frame[args.label_column].to_numpy(), frame[args.score_column].to_numpy())
    metrics["brier"] = brier_score(
        frame[args.label_column].to_numpy(), frame[args.score_column].to_numpy()
    )
    metrics["nll"] = negative_log_likelihood(
        frame[args.label_column].to_numpy(), frame[args.score_column].to_numpy()
    )
    metrics["ece10"] = expected_calibration_error(
        frame[args.label_column].to_numpy(), frame[args.score_column].to_numpy(), bins=10
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _cmd_calibrate(args: argparse.Namespace) -> None:
    valid = pd.read_csv(args.validation)
    test = pd.read_csv(args.test)
    temperature = fit_temperature(
        valid[args.logit_column].to_numpy(), valid[args.label_column].to_numpy()
    )
    test = test.copy()
    test[args.output_column] = apply_temperature(test[args.logit_column].to_numpy(), temperature)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.output, index=False)
    print(json.dumps({"temperature": temperature, "output": args.output}, indent=2))


def _cmd_precompute(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input)
    required = {args.smiles_column, args.protein_column}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"missing input columns: {sorted(missing)}")
    raw_smiles = frame[args.smiles_column].astype(str).tolist()
    smiles = [canonicalize_smiles(value) for value in raw_smiles]
    text = [f"SMILES={value}" for value in smiles]
    proteins = [normalize_protein_sequence(value) for value in frame[args.protein_column].astype(str)]
    smiles_emb = encode_texts(
        smiles,
        model_name=args.smiles_model,
        max_length=256,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=True,
    )
    text_emb = encode_texts(
        text,
        model_name=args.text_model,
        max_length=256,
        batch_size=args.batch_size,
        device=args.device,
    )
    protein_emb = encode_texts(
        proteins,
        model_name=args.protein_model,
        max_length=1024,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.label_column in frame:
        labels = frame[args.label_column].to_numpy(dtype=np.float32)
    else:
        labels = np.zeros(len(frame), dtype=np.float32)
    if args.split_column in frame:
        split = frame[args.split_column].to_numpy(dtype=np.int64)
    else:
        split = np.zeros(len(frame), dtype=np.int64)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        smiles=smiles_emb,
        text=text_emb,
        protein=protein_emb,
        labels=labels,
        split=split,
        has_text=np.ones(len(frame), dtype=np.float32),
    )
    print(args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scrdti")
    sub = parser.add_subparsers(dest="command", required=True)

    synthetic = sub.add_parser("make-synthetic", help="create a tiny NPZ smoke-test dataset")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--n", type=int, default=96)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.set_defaults(func=_cmd_make_synthetic)

    train = sub.add_parser("train", help="train SCR-DTI from precomputed feature NPZ")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cpu")
    train.set_defaults(func=_cmd_train)

    predict = sub.add_parser("predict", help="predict a split from precomputed feature NPZ")
    predict.add_argument("--config", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--split", type=int, default=2)
    predict.add_argument("--batch-size", type=int, default=256)
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(func=_cmd_predict)

    evaluate = sub.add_parser("evaluate", help="evaluate a CSV containing labels/probabilities")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--label-column", default="label")
    evaluate.add_argument("--score-column", default="probability")
    evaluate.set_defaults(func=_cmd_evaluate)

    calibrate = sub.add_parser("calibrate", help="fit validation-only temperature scaling")
    calibrate.add_argument("--validation", required=True)
    calibrate.add_argument("--test", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--label-column", default="label")
    calibrate.add_argument("--logit-column", default="logit")
    calibrate.add_argument("--output-column", default="calibrated_probability")
    calibrate.set_defaults(func=_cmd_calibrate)

    precompute = sub.add_parser("precompute", help="extract pooled public encoder features")
    precompute.add_argument("--input", required=True, help="CSV containing SMILES/protein sequence")
    precompute.add_argument("--output", required=True, help="output feature NPZ")
    precompute.add_argument("--smiles-column", default="smiles")
    precompute.add_argument("--protein-column", default="protein_sequence")
    precompute.add_argument("--label-column", default="label")
    precompute.add_argument("--split-column", default="split")
    precompute.add_argument("--smiles-model", default=DEFAULT_ENCODERS["smiles"])
    precompute.add_argument("--text-model", default=DEFAULT_ENCODERS["text"])
    precompute.add_argument("--protein-model", default=DEFAULT_ENCODERS["protein"])
    precompute.add_argument("--batch-size", type=int, default=8)
    precompute.add_argument("--device", default=None)
    precompute.set_defaults(func=_cmd_precompute)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

