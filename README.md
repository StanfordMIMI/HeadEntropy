# 🧠 HeadEntropy: Gradient-Stable Attention Heads Signal LLM Correctness

[![arXiv](https://img.shields.io/badge/arXiv-2602.13699-b31b1b.svg)](https://arxiv.org/abs/2602.13699)

## 🔍 Overview

HeadEntropy predicts whether an LLM's answer is correct from the spread of its attention
heads over the answer tokens, in the same forward pass. The trace of the softmax Jacobian
equals `1 − exp(−H₂)` with `H₂` the 2-Rényi entropy: sharp heads are ones training has
settled, diffuse heads are ones training would still move. High head entropy at inference
signals an answer not grounded in the training data.

## 📈 Performance Highlights

5 LLMs (Qwen3-1.7B/8B/32B, Llama-3.1-8B, Gemma-4-12B) × 5 datasets (TriviaQA, HotpotQA,
MedMCQA, FEVER, MATH):

- **🎯 Training-free**: 0.736 AUROC, beats every training-free baseline (+8.9% to +75.6%)
- **🔁 Transfer**: matches a hidden-state probe out of distribution without training (0.739)
- **⚡ Cost**: < 1% of inference

## 💻 Implementation Details

```python
out = model(input_ids, output_attentions=True)
A = torch.stack(out.attentions, dim=0)[:, 0]           # [layers, heads, T, T], rows sum to 1
tr_J = 1 - (A ** 2).sum(dim=-1)                        # [layers, heads, T]
H_bar = tr_J[:, :, answer_start:answer_end].mean(-1).flatten()   # one value per head
score = -H_bar.mean()                                  # training-free correctness score
```

Optional ℓ1 logistic regression on `H_bar` for the supervised probe
([models/logistic_regression.py](models/logistic_regression.py)).

## 🚀 Getting Started

```bash
git clone git@github.com:StanfordMIMI/HeadEntropy.git && cd HeadEntropy
bash setup_public.sh
```

[demo.ipynb](demo.ipynb): one question from each dataset through Qwen3-1.7B, scored with HeadEntropy.

Generate answers and features (one GPU):

```bash
./run_tqa.sh    # TriviaQA; also run_hqa.sh, run_indqa.sh, run_mathqa.sh, run_feverqa.sh
```

Score, train the probe, transfer it, and run baselines:

```bash
./eval.sh tqa 8B_q      # TriviaQA, Qwen3-8B; ./eval.sh --list explains the keys
```

Results land in `tables_<dataset>_trace_jacobian_answer/<model>/`. Figures:
[analysis/](analysis/).

## 📝 Citation

```bibtex
@article{ostmeier2026attention,
  title={Attention Head Entropy of LLMs Predicts Answer Correctness},
  author={Ostmeier, Sophie and Axelrod, Brian and Varma, Maya and Aali, Asad and Zhang, Yabin and Paschali, Magdalini and Koyejo, Sanmi and Langlotz, Curtis and Chaudhari, Akshay},
  journal={arXiv preprint arXiv:2602.13699},
  year={2026}
}
```

## 🙏 Acknowledgments

Baselines from [Lookback Lens](https://github.com/voidism/Lookback-Lens), LLM-Check and
ICR Probe. TriviaQA docs: [docs/triviaqa_dataset_README](docs/triviaqa_dataset_README).
