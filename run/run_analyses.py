#!/usr/bin/env python
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models.logistic_regression import train_logistic_regression
from run.utils.utils_analyse import (
    get_lengths_questions_answers,
    compute_all_token_lengths,
    get_results,
    get_data,
    get_entropy,
    get_token_probs,
    get_token_entropy,
    get_model_confidence,
    get_attn_score,
    get_hidden_state_score,
    get_lr,
    get_icr,
)
from run.utils.utils_plots import (
    get_no_heads_layers,
    make_top_low_histograms,

)


def test_generalization(data_path, semantic_section, label_metric, dataset_nickname, model_he, model_lr, model_hs, model_icr, data_frame, subset=None, skip_truncated=False, compute_ci=False):
    print(f"\nTesting generalization on {dataset_nickname}")
    data_math = get_data(data_path, subset=subset)
    corrected_data_math = data_math   
    X_test_math, y_test_math, y_f1_test_math, skipped_idx_val_math = get_entropy(corrected_data_math, section=semantic_section, correct=True, label_metric= label_metric, skip_truncated=skip_truncated, math=(dataset_nickname == "math"))
    print(f"y_test_math: {len(y_test_math)}, skipped: {len(skipped_idx_val_math)}, total: {len(data_math)}")
    assert len(y_test_math) + len(skipped_idx_val_math) == len(data_math), "Index mismatch!"
    transformed_X_test_math = model_he.named_steps["scaler"].transform(X_test_math)
    y_scores_test_math = model_he.named_steps["clf"].predict_proba(transformed_X_test_math)[:, 1]

    row = {"name": f"logistic_regression_{dataset_nickname}", **get_results(
        y_test_math,
        y_scores_test_math,
        y_test_math,
        y_scores_test_math,
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    
    
    transformed_probs_val_lr = model_lr.named_steps["scaler"].transform(get_lr(corrected_data_math, skip_idxs=skipped_idx_val_math))
    y_scores_test_math_lr = model_lr.named_steps["clf"].predict_proba(transformed_probs_val_lr)[:, 1]

    row = {"name": f"logistic_regression_{dataset_nickname}_lr", **get_results(
        y_test_math,
        y_scores_test_math_lr,
        y_test_math,
        y_scores_test_math_lr,
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    
    transformed_probs_val_hs = model_hs.named_steps["scaler"].transform(get_hidden_state_score(corrected_data_math, skip_idxs=skipped_idx_val_math))
    y_scores_test_math_hs = model_hs.named_steps["clf"].predict_proba(transformed_probs_val_hs)[:, 1]
    row = {"name": f"logistic_regression_{dataset_nickname}_hs", **get_results(
        y_test_math,
        y_scores_test_math_hs,
        y_test_math,
        y_scores_test_math_hs,
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)

    transformed_probs_val_icr = model_icr.named_steps["scaler"].transform(get_icr(corrected_data_math, skip_idxs=skipped_idx_val_math))
    y_scores_test_math_icr = model_icr.named_steps["clf"].predict_proba(transformed_probs_val_icr)[:, 1]
    row = {"name": f"logistic_regression_{dataset_nickname}_icr", **get_results(
        y_test_math,
        y_scores_test_math_icr,
        y_test_math,
        y_scores_test_math_icr,
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    return data_frame

def test_unsupervised(data_path, semantic_section, label_metric, dataset_nickname, data_frame, subset=None, skip_truncated=False, compute_ci=False):
    print(f"\nTesting generalization on {dataset_nickname}")
    if dataset_nickname == "he":
        he_name, lr_name, hs_name, icr_name = "mean_he", "mean_lr", "mean_hs", "mean_icr"
    else:
        he_name, lr_name, hs_name, icr_name = (
            f"mean_{dataset_nickname}", f"mean_{dataset_nickname}_lr",
            f"mean_{dataset_nickname}_hs", f"mean_{dataset_nickname}_icr",
        )
    data_math = get_data(data_path, subset=subset)
    corrected_data_math = data_math   
    X_test_math, y_test_math, y_f1_test_math, skipped_idx_val_math = get_entropy(corrected_data_math, section=semantic_section, correct=True, label_metric= label_metric, skip_truncated=skip_truncated, math=(dataset_nickname == "math"))
    print(f"y_test_math: {len(y_test_math)}, skipped: {len(skipped_idx_val_math)}, total: {len(data_math)}")
    assert len(y_test_math) + len(skipped_idx_val_math) == len(data_math), "Index mismatch!"
    row = {"name": he_name, **get_results(
        y_test_math,
        X_test_math.mean(dim=1) *-1,
        y_test_math,
        X_test_math.mean(dim=1) *-1,
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    
    X_test_math_lr = get_lr(corrected_data_math, skip_idxs=skipped_idx_val_math)
    row = {"name": lr_name, **get_results(
        y_test_math,
        X_test_math_lr.mean(dim=1),
        y_test_math,
        X_test_math_lr.mean(dim=1),
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    
    X_test_math_hs = get_hidden_state_score(corrected_data_math, skip_idxs=skipped_idx_val_math)
    row = {"name": hs_name, **get_results(
        y_test_math,
        X_test_math_hs.mean(dim=1),
        y_test_math,
        X_test_math_hs.mean(dim=1),
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)

    X_test_math_icr = get_icr(corrected_data_math, skip_idxs=skipped_idx_val_math)
    row = {"name": icr_name, **get_results(
        y_test_math,
        X_test_math_icr.mean(dim=1),
        y_test_math,
        X_test_math_icr.mean(dim=1),
        kind="prob",
        compute_ci=compute_ci,
        )}
    data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    
    for func in [get_token_probs, get_model_confidence, get_attn_score, get_token_entropy]:
        score_name = func.__name__.replace("get_", "")
        try:
            out_val = func(corrected_data_math, skip_idxs=skipped_idx_val_math)
        except Exception as e:
            # older generation runs predate attn_eig_prod / logprobs; skip the row
            # rather than aborting the whole analysis, the notebook reads NaN
            print(f"[warn] {score_name} unavailable on {dataset_nickname}: {e}")
            continue
        row = {"name": score_name if dataset_nickname == "he" else f"{score_name}_{dataset_nickname}", **get_results(
            y_test_math,
            out_val,
            y_test_math,
            out_val,
            kind="prob",
            compute_ci=compute_ci,
            )}
        data_frame = pd.concat([data_frame, pd.DataFrame([row])], axis=0)
    return data_frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate QA outputs with Exact Match + BLEU"
    )
    ap.add_argument(
        "--out_prefix_train",
        type=str,
        default="tqa_eval",
        help="Prefix of the training results files",
    )
    ap.add_argument(
        "--compute_ci",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to compute confidence intervals",
    )
    ap.add_argument(
        "--out_prefix_val", type=str, default=None, help="Prefix of the validation results files"
    )
    ap.add_argument(
        "--model_name", type=str, help="Model name, used as the output subdirectory"
    )
    ap.add_argument(
        "--med_generalization", type=str,  default=None, help="Prefix for med generalization test set"
    )
    ap.add_argument(
        "--correct_data",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to correct the data",
    )

    ap.add_argument(
        "--debug",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use a 200-item subset instead of 6000",
    )
    ap.add_argument(
        "--hot_generalization",
        type=str,
        default=None,
        help="Prefix for the HotpotQA generalization test set",
    )
    ap.add_argument(
        "--math_generalization",
        type=str,
        default=None,
        help="Prefix for the MATH generalization test set",
    )

    ap.add_argument(
        "--fever_generalization",
        type=str,
        default=None,
        help="Prefix for the FEVER generalization test set",
    )

    ap.add_argument(
        "--temp_generalization",
        type=str,
        default=None,
        help="Prefix for the temperature generalization test set",
    )
    ap.add_argument(
        "--semantic_section",
        type=str,
        default="trace_jacobian_answer_end",
        help="Semantic section to use",
    )
    ap.add_argument(
        "--semantic_section_emb",
        type=str,
        default="embedding_answer_end",
        help="Semantic section to use for embeddings",
    )
    ap.add_argument(
        "--regularization",
        type=float,
        default=1,
        help="Regularization strength",
    )
    ap.add_argument(
        "--label_metric",
        type=str,
        default="exact_match",
        help="Label metric to use",
    )
    ap.add_argument(
        "--do_data_ablation",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to do data ablation",
    )
    ap.add_argument(
        "--no_scaling",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to skip feature scaling",
    )
    ap.add_argument(
        "--do_c_ablation",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to do c ablation",
    )
    ap.add_argument(
        "--do_layer_ablation",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to do layer ablation",
    )
    ap.add_argument(
        "--do_length_analysis",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to do length analysis",
    )
    ap.add_argument(
        "--thinking",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether the generations used thinking mode",
    )
    ap.add_argument(
        "--exclude_truncated_train",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Drop generations that used the whole token budget from the training set",
    )
    ap.add_argument(
        "--exclude_truncated_eval",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Drop them from the val and generalization sets too. Note this changes "
             "the population being scored, and on MATH it removes most of it",
    )



    args = ap.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name
    print(f"Model name: {model_name}")
    random_model_bool = bool("random" in model_name)

    subset = None
    if args.debug:
        subset = 200
    else:
        subset = 6000

    dataset_suffix = args.out_prefix_train.split("_")[1].split("/")[0]
    if dataset_suffix == "vqa":
        dataset_suffix = "triv"
    print(f"Dataset suffix: {dataset_suffix}")
    section_suffix = args.semantic_section.replace("entropy_","").replace("_end","")
    if section_suffix == "answer":
        section_suffix = ""
    else:
        section_suffix = f"_{section_suffix}"
    if args.label_metric != "exact_match":
        section_suffix += f"_{args.label_metric}"
    
    if not bool(args.thinking):
        section_suffix += f"_no_thinking"
        args.out_prefix_train = args.out_prefix_train.replace("1.3", "0.0") + "_no_thinking"
        args.out_prefix_val = args.out_prefix_val.replace("1.3", "0.0") + "_no_thinking"

    
    if args.no_scaling:
        section_suffix += "_no_scaling"

    if "double" in args.out_prefix_val:
        section_suffix += "_double"

    analyses_path = Path(f"tables_{dataset_suffix}{section_suffix}")

    if args.debug:
        analyses_path = Path(f"tables_{dataset_suffix}{section_suffix}_debug")
    analyses_path.mkdir(parents=True, exist_ok=True)
    analyses_path_model = analyses_path / model_name
    analyses_path_model.mkdir(parents=True, exist_ok=True)     


    results_data_frame = pd.DataFrame(columns=["name","overall_test","roc_auc_test","pr_auc_test", "roc_auc_train" , "pr_auc_train"])


    data = get_data(args.out_prefix_train, subset=subset)

    if bool(args.out_prefix_val):
        data_val = get_data(args.out_prefix_val, subset=subset)
        print("saving test ids to", args.out_prefix_val)
    else:
        print("Splitting data into train and validation, 80/20")
        data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
        data = data_train
        data_val = data_val
        print("Length of validation data: ", len(data_val))
        print("Length of training data: ", len(data_train))
    
    # guarded like the test_generalization calls below: an unset path reaches
    # get_entropy with no records, and its empty-input branch drops into ipdb, which
    # hangs a batch run on an interactive prompt instead of failing
    if bool(args.math_generalization):
        results_data_frame  = test_unsupervised(
            args.math_generalization,
            args.semantic_section,
            args.label_metric,
            "math",
            results_data_frame,
            subset=subset,
            skip_truncated=args.exclude_truncated_eval,
            compute_ci=args.compute_ci,
        )

    if bool(args.fever_generalization):
        results_data_frame  = test_unsupervised(
            args.fever_generalization,
            args.semantic_section,
            args.label_metric,
            "fever",
            results_data_frame,
            subset=subset,
            skip_truncated=args.exclude_truncated_eval,
            compute_ci=args.compute_ci,
        )

    if bool(args.out_prefix_val):
        results_data_frame = test_unsupervised(
            args.out_prefix_val,
            args.semantic_section,
            args.label_metric,
            "he",
            results_data_frame,
            subset=subset,
            skip_truncated=args.exclude_truncated_eval,
            compute_ci=args.compute_ci,
        )


    print(results_data_frame)

    path = f"{analyses_path_model}/latex_results_{timestamp}.csv"
    print(results_data_frame.to_csv(path,index=False))
    text_path = f"{analyses_path_model}/text_results.txt"
    with open(text_path, "w") as f:
        f.write(results_data_frame.to_string(index=True))
    print("Saved results to ", path)

    print("Compute average token lengths")
    token_lengths = compute_all_token_lengths(data_val)
    print("Token lengths: ", token_lengths)
    with open(analyses_path_model / "average_token_lengths_val.json", "w") as f:
        json.dump(token_lengths, f)

    print(f"Using section: {args.semantic_section}")
    X_train, y_train, y_f1_train, skipped_idx_trian  = get_entropy(data, section=args.semantic_section, correct=args.correct_data, label_metric=args.label_metric, skip_truncated=args.exclude_truncated_train)
    X_test, y_test, y_f1_test, skipped_idx_val = get_entropy(data_val, section=args.semantic_section, correct=args.correct_data, label_metric=args.label_metric, skip_truncated=args.exclude_truncated_eval)
    if len(X_train) == len(data):
        print("No skipped indices in train data")
    else:
        print(f"SKIPPPPPPPPPED!!!!!!!!!!!!!!!!!!! {len(data) - len(X_train)} indices")
        
    if len(X_test) == len(data_val):
        print("No skipped indices in val data")
    else:
        print(f"SKIPPPPPPPPPED VAL !!!!!!!!!!!!!!!!!!! {len(data_val) - len(X_test)} indices")
    print(f"Accuracy test: {y_test.float().mean():.4f}")
    print(f"Accuracy train: {y_train.float().mean():.4f}")

    diff_entropy = X_test[y_test == 0].mean(dim=0) - X_test[y_test == 1].mean(dim=0)  # [2560]

    predictor_dict = {}
    predictor_dict["diff_entropy"] = diff_entropy.numpy().tolist()

    if random_model_bool:
        make_top_low_histograms(X_test, y_test, idxs_high=None, idxs_low=None, out_path=args.out_prefix_val if args.out_prefix_val else args.out_prefix_train)

    print("----------------------------------------------------------")
    print("Compute lookback score")
    probs_lb = get_lr(data, skip_idxs=skipped_idx_trian)
    probs_val_lb = get_lr(data_val, skip_idxs=skipped_idx_val)

    y_scores_train_lr, y_scores_test_lr, model_lr , _ = train_logistic_regression(
        probs_lb, y_train, probs_val_lb, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=None
    )
    row = {"name": "lookback_lens_ratio", **get_results(
        y_test,
        y_scores_test_lr,
        y_train,
        y_scores_train_lr,
        kind="prob",
        compute_ci= args.compute_ci
    )}
    results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)

    print("----------------------------------------------------------")
    print("Compute ICR probe score")
    probs_icr = get_icr(data, skip_idxs=skipped_idx_trian)
    probs_val_icr = get_icr(data_val, skip_idxs=skipped_idx_val)

    y_scores_train_icr, y_scores_test_icr, model_icr, _ = train_logistic_regression(
        probs_icr, y_train, probs_val_icr, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=None
    )
    row = {"name": "icr_score", **get_results(
        y_test,
        y_scores_test_icr,
        y_train,
        y_scores_train_icr,
        kind="prob",
        compute_ci= args.compute_ci
    )}
    results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)

    print("----------------------------------------------------------")
    print("Compute hidden score")
    hidden_probs = get_hidden_state_score(data, semantic_section=args.semantic_section_emb, skip_idxs=skipped_idx_trian)
    hidden_probs_val = get_hidden_state_score(data_val, semantic_section=args.semantic_section_emb, skip_idxs=skipped_idx_val)

    y_scores_train_hidden, y_scores_test_hidden, model_hs, _ = train_logistic_regression(
        hidden_probs, y_train, hidden_probs_val, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=None
    )
    row = {"name": "hidden_state_score", **get_results(
        y_test,
        y_scores_test_hidden,
        y_train,
        y_scores_train_hidden,
        kind="prob",
        compute_ci= args.compute_ci
    )}
    results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)

    print("----------------------------------------------------------")
    print("Training he with Logistic Regression with Lasso")

    y_scores_train, y_scores_test, log_model, explainer = train_logistic_regression(
                X_train, y_train, X_test, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=args.out_prefix_val, regularization=args.regularization
            )    
    no_layers, no_heads = get_no_heads_layers(X_train[0])
    row = {"name": f"he_logistic_regression_{no_layers}", **get_results(
                y_test,
                y_scores_test,
                y_train,
                y_scores_train,
                kind="prob",
                compute_ci= args.compute_ci
            )}
    results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)    


    # reshape to [num_samples, no_layers, no_heads]
    reshaped_X_train = X_train.reshape(len(X_train), no_layers, no_heads)
    reshaped_X_test = X_test.reshape(len(X_test), no_layers, no_heads)

    layers = sorted(set(range(1, no_layers, 5)) | {1, no_layers})
    if args.do_layer_ablation:
        for layer in layers:
            if not args.do_layer_ablation and layer != layers[-1]:
                continue
            print(f"Training layer {layer}")
            reduced_X_train = reshaped_X_train[:, -layer:, :].reshape(X_train.shape[0], -1)
            reduced_X_test = reshaped_X_test[:, -layer:, :].reshape(X_test.shape[0], -1)
            y_scores_train, y_scores_test, log_model, explainer = train_logistic_regression(
                reduced_X_train, y_train, reduced_X_test, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=args.out_prefix_val, regularization=args.regularization
            )
            row = {"name": f"he_logistic_regression_{layer}", **get_results(
                y_test,
                y_scores_test,
                y_train,
                y_scores_train,
                kind="prob",
                compute_ci= args.compute_ci
            )}
            results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)    
    
    if args.do_data_ablation:
        data_fractions = [0.001, 0.01, 0.1, 0.2,0.4,0.6,0.8]
        indices = torch.randperm(X_train.size(0))
        shuffled_X_train = X_train[indices]
        shuffled_y_train = y_train[indices]

        for fraction in data_fractions:
            print(f"Training with fraction {fraction}")
            num_samples = int(fraction * shuffled_X_train.shape[0])
            X_train_fraction = shuffled_X_train[:num_samples, ...]
            y_train_fraction = shuffled_y_train[:num_samples, ...]

            print(f"Training with fraction {fraction}, num samples {num_samples}")
            y_scores_train, y_scores_test, _, _ = train_logistic_regression(
            X_train_fraction, y_train_fraction, X_test, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=args.out_prefix_val, regularization=args.regularization)
            row = {"name": f"he_logistic_regression_{fraction}", **get_results(
                y_test,
                y_scores_test,
                y_train_fraction,
                y_scores_train,
                kind="prob", 
                compute_ci=False
            )}
            results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)
        
    if args.do_c_ablation:
        c_values = [0.0001, 0.001, 0.01, 0.1,0.5,2,4]
        for c in c_values:
            print(f"Training with c {c}")
            y_scores_train, y_scores_test, _, _ = train_logistic_regression(
            X_train, y_train, X_test, y_test, feature_cols_names = None, no_scaling=args.no_scaling, model_name=args.out_prefix_val, regularization=c
            )
            row = {"name": f"he_logistic_regression_c_{c}", **get_results(
                y_test,
                y_scores_test,
                y_train,
                y_scores_train,
                kind="prob",
                compute_ci=False
            )}
            results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)
            
            assert X_test.shape[1] == X_test.shape[1]
            
            transformed_X_test = log_model.named_steps["scaler"].transform(X_test)
            shap_values = explainer.shap_values(transformed_X_test)
            assert len(shap_values) == len(y_scores_test) == len(y_test), f"SHAP values length mismatch, shap {len(shap_values)}, probs {len(y_scores_test)}, data {len(data_val)}"
            predictor_dict[f"logistic_regression_c_{c}"] = log_model.named_steps["clf"].coef_.ravel().tolist()
            predictor_dict[f"shap_values_c_{c}"] = shap_values.mean(axis=0).tolist()
    
    X_test_reshape = X_test.reshape(X_test.shape[0], no_layers, no_heads)

    n_layers_half = no_layers // 2
    row = {"name": f"he_logistic_regression_layer_{n_layers_half}", **get_results(
        y_test,
        X_test_reshape[:, n_layers_half, :].mean(axis=1) * -1,
        y_test,
        X_test_reshape[:, n_layers_half, :].mean(axis=1) * -1,
        kind="prob",
        compute_ci=False
    )}
    results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)

    transformed_X_test = log_model.named_steps["scaler"].transform(X_test)
    shap_values = explainer.shap_values(transformed_X_test)
    assert len(shap_values) == len(y_scores_test) == len(y_test), f"SHAP values length mismatch, shap {len(shap_values)}, probs {len(y_scores_test)}, data {len(data_val)}"

    mask_dict = [{"message": data_val[i]["prompt"], "ground_truth_answers": data_val[i]["ground_truth_answers"], "shap_values": shap_values[i].tolist(), "probability": y_scores_test[i].tolist(), "index": i, "label": y_test[i].tolist(), "exact_match": data_val[i]["exact_match"]} for i in range(len(shap_values))]

    # save in four chunks
    if args.out_prefix_val:
        for i in range(4):
            with open(Path(analyses_path_model) / f"shap_values_{i}.json", "w") as f:
                json.dump(mask_dict[i::4], f, indent=4)

    predictor_dict["logistic_regression"] = log_model.named_steps["clf"].coef_.ravel().tolist()
    predictor_dict["shap_values"] = shap_values.mean(axis=0).tolist()


    with open(analyses_path_model / f"predictor_dict_{timestamp}.json", "w") as f:
        json.dump(predictor_dict, f, indent=4)
    
    path = f"{analyses_path_model}/latex_results_{timestamp}.csv"
    print(results_data_frame.to_csv(path,index=False))

    del data_val, data
    print("----------------------------------------------------------")
    if bool(args.temp_generalization):
        print("\nTesting generalization on temp")
        data_temp = get_data(args.temp_generalization, subset=subset)
        if len(data_temp) == 0:
            print("No data found for temp generalization test set, skipping")
        else:
            X_test_hotpot, y_test_hotpot, y_f1_test_hotpot, skipped_idx_val_hot = get_entropy(data_temp, section=args.semantic_section, correct=True, label_metric=args.label_metric)
            transformed_X_test_hotpot = log_model.named_steps["scaler"].transform(X_test_hotpot)
            y_scores_test_hotpot = log_model.named_steps["clf"].predict_proba(transformed_X_test_hotpot)[:, 1]
            row = {"name": "he_logistic_regression_temp", **get_results(
                y_test_hotpot,
                y_scores_test_hotpot,
                y_train,
                y_scores_train,
                kind="prob"
                )}
            results_data_frame = pd.concat([results_data_frame, pd.DataFrame([row])], axis=0)
    
    if bool(args.hot_generalization):
        results_data_frame = test_generalization(args.hot_generalization, args.semantic_section, args.label_metric, "hotpot", log_model, model_lr, model_hs, model_icr, results_data_frame, subset=subset, skip_truncated=args.exclude_truncated_eval, compute_ci=args.compute_ci)

    if bool(args.med_generalization):
        results_data_frame = test_generalization(args.med_generalization, args.semantic_section, args.label_metric, "med", log_model, model_lr, model_hs, model_icr, results_data_frame, subset=subset, skip_truncated=args.exclude_truncated_eval, compute_ci=args.compute_ci)

    if bool(args.math_generalization):
        results_data_frame = test_generalization(args.math_generalization, args.semantic_section, args.label_metric, "math", log_model, model_lr, model_hs, model_icr, results_data_frame, subset=subset, skip_truncated=args.exclude_truncated_eval, compute_ci=args.compute_ci)

    if bool(args.fever_generalization):
        results_data_frame = test_generalization(args.fever_generalization, args.semantic_section, args.label_metric, "fever", log_model, model_lr, model_hs, model_icr, results_data_frame, subset=subset, skip_truncated=args.exclude_truncated_eval, compute_ci=args.compute_ci)

    if args.do_length_analysis:
        print("saving length vs. performance data")
        length_df = pd.DataFrame()
        length_df["length"] = get_lengths_questions_answers(data_val, skip_idxs=skipped_idx_val)
        length_df["correct"] = y_test.numpy()
        length_df["predicted_prob"] = y_scores_test.numpy()
        length_df.to_csv(analyses_path_model / "lengths_vs_performance.csv", index=False)
        print("saved lengths vs. performance data to ", analyses_path_model / "lengths_vs_performance.csv")

    
    print(results_data_frame)

    path = f"{analyses_path_model}/latex_results_{timestamp}.csv"
    print(results_data_frame.to_csv(path,index=False))
    text_path = f"{analyses_path_model}/text_results.txt"
    with open(text_path, "w") as f:
        f.write(results_data_frame.to_string(index=True))
    print("Saved results to ", path)
