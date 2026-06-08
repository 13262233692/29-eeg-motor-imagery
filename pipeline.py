import argparse
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_pipeline
from csp import CSP
from riemannian import (
    SPDCovariance,
    TangentSpace,
    frechet_mean,
    compute_subject_means,
    cross_subject_align,
    tangent_space_project,
    riemannian_distance,
)


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


def build_riemannian_pipeline(
    hidden_layer_sizes: tuple = (128, 64),
    max_iter_mlp: int = 500,
    tangent_max_iter: int = 50,
    tangent_tol: float = 1e-6,
) -> Pipeline:
    return Pipeline([
        ("cov", SPDCovariance()),
        ("ts", TangentSpace(max_iter=tangent_max_iter, tol=tangent_tol)),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter_mlp,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
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
        print(f"[CV] {metric}: {scores.mean():.4f} +/- {scores.std():.4f} "
              f"(range: {scores.min():.4f} - {scores.max():.4f})")

    return summary


def evaluate_cross_subject_riemannian(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    hidden_layer_sizes: tuple = (128, 64),
    max_iter_mlp: int = 500,
    frechet_max_iter: int = 50,
    frechet_tol: float = 1e-6,
) -> dict:
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    print(f"\n[LOSO-CV] Leave-One-Subject-Out Cross-Validation")
    print(f"[LOSO-CV] {n_subjects} subjects, {X.shape[0]} total trials")
    print(f"[LOSO-CV] Pipeline: SPDCovariance -> CrossSubjectAlign -> "
          f"TangentSpace -> StandardScaler -> MLPClassifier")

    fold_accuracies = []
    fold_details = []

    for fold_idx, test_subject in enumerate(unique_subjects):
        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        train_subjects = subject_ids[train_mask]
        test_subjects_arr = subject_ids[test_mask]

        n_train_classes = len(np.unique(y_train))
        n_test_classes = len(np.unique(y_test))
        if n_train_classes < 2 or n_test_classes < 2:
            print(f"  Fold {fold_idx+1}/{n_subjects} (subject={test_subject}): "
                  f"SKIPPED (single class in train or test)")
            continue

        cov_estimator = SPDCovariance()
        cov_train = cov_estimator.fit_transform(X_train)
        cov_test = cov_estimator.fit_transform(X_test)

        train_means = compute_subject_means(
            cov_train, train_subjects,
            max_iter=frechet_max_iter, tol=frechet_tol
        )

        unique_train_subjects = np.unique(train_subjects)
        mean_covs = np.array([train_means[s] for s in unique_train_subjects])
        reference_mean = frechet_mean(
            mean_covs, max_iter=frechet_max_iter, tol=frechet_tol
        )

        cov_train_aligned = cross_subject_align(
            cov_train, train_subjects, train_means, reference_mean
        )

        test_means = compute_subject_means(
            cov_test, test_subjects_arr,
            max_iter=frechet_max_iter, tol=frechet_tol
        )
        combined_means = {**train_means, **test_means}

        cov_test_aligned = cross_subject_align(
            cov_test, test_subjects_arr, combined_means, reference_mean
        )

        ts = TangentSpace(max_iter=frechet_max_iter, tol=frechet_tol)
        feat_train = ts.fit_transform(cov_train_aligned)
        feat_test = ts.transform(cov_test_aligned)

        scaler = StandardScaler()
        feat_train = scaler.fit_transform(feat_train)
        feat_test = scaler.transform(feat_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter_mlp,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(feat_train, y_train)
        acc = mlp.score(feat_test, y_test)
        fold_accuracies.append(acc)
        fold_details.append({
            "subject": test_subject,
            "accuracy": acc,
            "n_train": len(y_train),
            "n_test": len(y_test),
        })

        print(f"  Fold {fold_idx+1}/{n_subjects} (subject={test_subject}): "
              f"acc={acc:.4f} (train={len(y_train)}, test={len(y_test)})")

    scores = np.array(fold_accuracies)
    summary = {
        "accuracy": {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores,
        },
        "fold_details": fold_details,
        "n_subjects": n_subjects,
        "n_folds_evaluated": len(fold_accuracies),
    }

    print(f"\n[LOSO-CV] Mean accuracy: {scores.mean():.4f} +/- {scores.std():.4f} "
          f"({len(fold_accuracies)}/{n_subjects} folds evaluated)")

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

    if args.pipeline == "csp":
        pipeline = build_csp_svm_pipeline(
            n_components=args.n_components,
            svm_kernel=args.svm_kernel,
            svm_c=args.svm_c,
            svm_gamma=args.svm_gamma,
        )
        cv_summary = evaluate_pipeline(pipeline, X, y, cv_strategy=args.cv, n_folds=args.n_folds)

        print("\n" + "=" * 60)
        print("CSP-SVM Evaluation Summary (Leak-Free)")
        print("=" * 60)
        print(f"  Pipeline: CSP({args.n_components}) -> StandardScaler -> "
              f"SVC(kernel={args.svm_kernel}, C={args.svm_c})")
        print(f"  CV strategy: {args.cv}")
        for metric, vals in cv_summary.items():
            print(f"  {metric}: {vals['mean']:.4f} +/- {vals['std']:.4f}")
        print("=" * 60)

        pipeline.fit(X, y)

    elif args.pipeline == "riemannian":
        if args.subject_ids_file:
            subject_ids = np.load(args.subject_ids_file)
            if len(subject_ids) != len(X):
                raise ValueError(
                    f"subject_ids length ({len(subject_ids)}) != "
                    f"epochs count ({len(X)})"
                )
            cv_summary = evaluate_cross_subject_riemannian(
                X, y, subject_ids,
                hidden_layer_sizes=tuple(int(h) for h in args.mlp_hidden.split(",")),
                max_iter_mlp=args.mlp_max_iter,
                frechet_max_iter=args.frechet_max_iter,
                frechet_tol=args.frechet_tol,
            )
            print("\n" + "=" * 60)
            print("Riemannian Cross-Subject Evaluation Summary")
            print("=" * 60)
            print(f"  Pipeline: SPDCovariance -> CrossSubjectAlign -> "
                  f"TangentSpace -> MLP({args.mlp_hidden})")
            print(f"  CV strategy: Leave-One-Subject-Out "
                  f"({cv_summary['n_subjects']} subjects)")
            print(f"  Accuracy: {cv_summary['accuracy']['mean']:.4f} +/- "
                  f"{cv_summary['accuracy']['std']:.4f}")
            print("=" * 60)
        else:
            pipeline = build_riemannian_pipeline(
                hidden_layer_sizes=tuple(int(h) for h in args.mlp_hidden.split(",")),
                max_iter_mlp=args.mlp_max_iter,
            )
            cv_summary = evaluate_pipeline(pipeline, X, y, cv_strategy=args.cv, n_folds=args.n_folds)

            print("\n" + "=" * 60)
            print("Riemannian Within-Subject Evaluation Summary")
            print("=" * 60)
            print(f"  Pipeline: SPDCovariance -> TangentSpace -> "
                  f"MLP({args.mlp_hidden})")
            print(f"  CV strategy: {args.cv}")
            for metric, vals in cv_summary.items():
                print(f"  {metric}: {vals['mean']:.4f} +/- {vals['std']:.4f}")
            print("=" * 60)

            pipeline.fit(X, y)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "cv_accuracy_mean": cv_summary["accuracy"]["mean"],
            "cv_accuracy_std": cv_summary["accuracy"]["std"],
            "cv_accuracy_scores": cv_summary["accuracy"]["scores"],
            "labels": y,
            "pipeline_type": args.pipeline,
            "info_sfreq": epochs.info["sfreq"],
            "info_ch_names": np.array(epochs.ch_names),
        }
        if "fold_details" in cv_summary:
            save_dict["fold_details"] = cv_summary["fold_details"]
        np.savez(str(output_path), **save_dict)
        print(f"\nResults saved to: {output_path}")

    return cv_summary


def main():
    parser = argparse.ArgumentParser(
        description="Motor Imagery EEG Decoding Pipeline: CSP-SVM / Riemannian-MLP"
    )

    parser.add_argument("input", type=str, help="Path to raw EEG data file (.bdf or .fif)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save output .npz file")
    parser.add_argument("--pipeline", type=str, default="riemannian",
                        choices=["csp", "riemannian"],
                        help="Pipeline type: csp or riemannian (default: riemannian)")

    filt_group = parser.add_argument_group("Filtering")
    filt_group.add_argument("--notch-freq", type=float, default=50)
    filt_group.add_argument("--notch-harmonics", type=str, default="2,3,4")
    filt_group.add_argument("--l-freq", type=float, default=8.0)
    filt_group.add_argument("--h-freq", type=float, default=30.0)

    epoch_group = parser.add_argument_group("Epoching")
    epoch_group.add_argument("--event-labels", type=str, default=None)
    epoch_group.add_argument("--event-codes", type=str, default=None)
    epoch_group.add_argument("--stim-channel", type=str, default=None)
    epoch_group.add_argument("--tmin", type=float, default=-0.5)
    epoch_group.add_argument("--tmax", type=float, default=3.5)
    epoch_group.add_argument("--reject-threshold", type=float, default=150e-6)

    csp_group = parser.add_argument_group("CSP (only for --pipeline csp)")
    csp_group.add_argument("--n-components", type=int, default=4)
    csp_group.add_argument("--svm-kernel", type=str, default="rbf")
    csp_group.add_argument("--svm-c", type=float, default=1.0)
    csp_group.add_argument("--svm-gamma", type=str, default="scale")

    riem_group = parser.add_argument_group("Riemannian (only for --pipeline riemannian)")
    riem_group.add_argument("--mlp-hidden", type=str, default="128,64",
                            help="Comma-separated MLP hidden layer sizes (default: 128,64)")
    riem_group.add_argument("--mlp-max-iter", type=int, default=500)
    riem_group.add_argument("--frechet-max-iter", type=int, default=50)
    riem_group.add_argument("--frechet-tol", type=float, default=1e-6)
    riem_group.add_argument("--subject-ids-file", type=str, default=None,
                            help="Path to .npy file with subject IDs for cross-subject LOSO")

    cv_group = parser.add_argument_group("Cross-Validation (within-subject)")
    cv_group.add_argument("--cv", type=str, default="5fold",
                          choices=["5fold", "10fold", "loso", "kfold"])
    cv_group.add_argument("--n-folds", type=int, default=5)

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
