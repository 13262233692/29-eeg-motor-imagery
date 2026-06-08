import argparse
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    LeaveOneOut,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_pipeline
from csp import CSP


def build_csp_svm_pipeline(
    n_components: int = 4,
    svm_kernel: str = "rbf",
    svm_c: float = 1.0,
    svm_gamma: str = "scale",
) -> Pipeline:
    return Pipeline([
        ("csp", CSP(n_components=n_components, log=True)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=svm_kernel, C=svm_c, gamma=svm_gamma)),
    ])


def evaluate_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: str = "5fold",
    n_folds: int = 5,
) -> dict:
    if cv_strategy == "loso":
        cv = LeaveOneOut()
    elif cv_strategy == "5fold":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_strategy == "10fold":
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    elif cv_strategy == "kfold":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    print(f"\n[CV] Strategy: {cv_strategy}"
          + (f" (n_splits={n_folds})" if cv_strategy == "kfold" else ""))
    print(f"[CV] Pipeline: {' -> '.join(name for name, _ in pipeline.steps)}")
    print(f"[CV] Data: {X.shape[0]} trials, {X.shape[1]} channels, "
          f"{X.shape[2]} samples/trial")
    print(f"[CV] Classes: {np.unique(y)[0]} (n={np.sum(y == np.unique(y)[0])}), "
          f"{np.unique(y)[1]} (n={np.sum(y == np.unique(y)[1])})")

    scoring = ["accuracy"]
    if cv_strategy != "loso":
        scoring.extend(["f1", "roc_auc"])

    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = results[key]
        summary[metric] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores,
        }
        print(f"[CV] {metric}: {scores.mean():.4f} ± {scores.std():.4f} "
              f"(range: {scores.min():.4f} - {scores.max():.4f})")

    return summary


def run_pipeline(args):
    event_id = None
    if args.event_labels and args.event_codes:
        labels = [l.strip() for l in args.event_labels.split(",")]
        codes = [int(c.strip()) for c in args.event_codes.split(",")]
        if len(labels) != len(codes):
            raise ValueError("Number of event labels must match number of event codes")
        event_id = dict(zip(labels, codes))
    elif args.event_labels or args.event_codes:
        raise ValueError("Both --event-labels and --event-codes must be provided together")

    epochs = preprocess_pipeline(
        filepath=args.input,
        event_id=event_id,
        stim_channel=args.stim_channel,
        notch_freq=args.notch_freq,
        notch_harmonics=tuple(int(h) for h in args.notch_harmonics.split(",")),
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(None, 0),
        reject_threshold=args.reject_threshold,
        eog_channels=None,
    )

    X = epochs.get_data()
    y = epochs.events[:, 2]

    pipeline = build_csp_svm_pipeline(
        n_components=args.n_components,
        svm_kernel=args.svm_kernel,
        svm_c=args.svm_c,
        svm_gamma=args.svm_gamma,
    )

    cv_summary = evaluate_pipeline(
        pipeline, X, y,
        cv_strategy=args.cv,
        n_folds=args.n_folds,
    )

    print("\n" + "=" * 60)
    print("Leak-Free Evaluation Summary")
    print("=" * 60)
    print(f"  Pipeline: CSP({args.n_components}) -> StandardScaler -> "
          f"SVC(kernel={args.svm_kernel}, C={args.svm_c})")
    print(f"  CV strategy: {args.cv}")
    for metric, vals in cv_summary.items():
        print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    print("=" * 60)

    pipeline.fit(X, y)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        csp_step = pipeline.named_steps["csp"]
        np.savez(
            str(output_path),
            cv_accuracy_mean=cv_summary["accuracy"]["mean"],
            cv_accuracy_std=cv_summary["accuracy"]["std"],
            cv_accuracy_scores=cv_summary["accuracy"]["scores"],
            csp_filters=csp_step.get_spatial_filters(),
            csp_patterns=csp_step.get_spatial_patterns(),
            csp_eigenvalues=csp_step.eigenvalues_,
            csp_eigenvalues_picked=csp_step.eigenvalues_picked_,
            labels=y,
            info_sfreq=epochs.info["sfreq"],
            info_ch_names=np.array(epochs.ch_names),
        )
        print(f"\nResults saved to: {output_path}")

    return pipeline, cv_summary


def main():
    parser = argparse.ArgumentParser(
        description="Motor Imagery EEG Decoding Pipeline: Preprocessing + CSP-SVM (Leak-Free)"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to raw EEG data file (.bdf or .fif)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output .npz file with CV results and CSP filters",
    )

    filt_group = parser.add_argument_group("Filtering")
    filt_group.add_argument(
        "--notch-freq", type=float, default=50,
        help="Notch filter frequency in Hz (default: 50)",
    )
    filt_group.add_argument(
        "--notch-harmonics", type=str, default="2,3,4",
        help="Comma-separated notch harmonics (default: 2,3,4)",
    )
    filt_group.add_argument(
        "--l-freq", type=float, default=8.0,
        help="Bandpass low cutoff frequency in Hz (default: 8.0)",
    )
    filt_group.add_argument(
        "--h-freq", type=float, default=30.0,
        help="Bandpass high cutoff frequency in Hz (default: 30.0)",
    )

    epoch_group = parser.add_argument_group("Epoching")
    epoch_group.add_argument(
        "--event-labels", type=str, default=None,
        help="Comma-separated event labels, e.g. 'left_hand,right_hand'",
    )
    epoch_group.add_argument(
        "--event-codes", type=str, default=None,
        help="Comma-separated event codes, e.g. '1,2'",
    )
    epoch_group.add_argument(
        "--stim-channel", type=str, default=None,
        help="Name of the stimulus channel for event extraction",
    )
    epoch_group.add_argument(
        "--tmin", type=float, default=-0.5,
        help="Epoch start time relative to event (seconds, default: -0.5)",
    )
    epoch_group.add_argument(
        "--tmax", type=float, default=3.5,
        help="Epoch end time relative to event (seconds, default: 3.5)",
    )
    epoch_group.add_argument(
        "--reject-threshold", type=float, default=150e-6,
        help="Peak-to-peak rejection threshold in Volts (default: 150e-6)",
    )

    csp_group = parser.add_argument_group("CSP")
    csp_group.add_argument(
        "--n-components", type=int, default=4,
        help="Number of CSP components to extract (default: 4)",
    )

    svm_group = parser.add_argument_group("SVM Classifier")
    svm_group.add_argument(
        "--svm-kernel", type=str, default="rbf",
        help="SVM kernel type (default: rbf)",
    )
    svm_group.add_argument(
        "--svm-c", type=float, default=1.0,
        help="SVM regularization parameter C (default: 1.0)",
    )
    svm_group.add_argument(
        "--svm-gamma", type=str, default="scale",
        help="SVM gamma parameter (default: scale)",
    )

    cv_group = parser.add_argument_group("Cross-Validation")
    cv_group.add_argument(
        "--cv", type=str, default="5fold",
        choices=["5fold", "10fold", "loso", "kfold"],
        help="Cross-validation strategy (default: 5fold)",
    )
    cv_group.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of folds when --cv=kfold (default: 5)",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
