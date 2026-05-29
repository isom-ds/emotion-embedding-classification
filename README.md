# emotion-embedding-classification
> Fine-tuning Llama 3.1 8B with LoRA for multi-label emotion classification under scarce training data.

## Abstract

Multi-label emotion classification from text requires large annotated datasets that are rarely available in domain-specific contexts. This repository systematically evaluates parameter-efficient fine-tuning of Llama 3.1 8B Instruct using LoRA across dataset sizes of 1k, 1.5k, and 5k samples drawn from GoEmotions, to identify optimal training configurations when labelled data is scarce. The work operationalises the "doing more with less" principle by showing that small, well-curated datasets combined with efficient fine-tuning can achieve competitive multi-label emotion classification performance.

## Research Context

- **Thesis:** *Epidemiology of Online Emotions* (Kok-Shun, 2026)
- **Chapter:** Chapter 5 — Emotion Detection Models
- **Contribution type:** Artefact (fine-tuned LLM pipeline)
- **Associated paper:** "Doing More with Less: Tackling Data Limitations in Emotion Detection in Text Using Generative AI," AMCIS 2025

## Methods

- Llama 3.1 8B Instruct (HuggingFace)
- LoRA / QLoRA (Low-Rank Adaptation) via PEFT
- Supervised Fine-Tuning (SFT) via TRL
- Multi-label classification head
- Comparative training across 1k / 1.5k / 5k sample sizes

## Datasets

| Dataset | Description | Access |
|---------|-------------|--------|
| GoEmotions | Google 28-class fine-grained emotion corpus (multi-label) | Open |
| NRC Emotion Lexicon | Word-level emotion associations (used as auxiliary lexicon) | Open |

## Repository Structure

```
emotion-embedding-classification/
├── data/                          # Raw and preprocessed datasets
├── functions/                     # Reusable helper functions
├── models/                        # Saved model checkpoints
├── output/                        # Training outputs and evaluation results
├── 000_prepare_data.ipynb         # Data preparation pipeline
├── 010_llama318BInstruct_1k.ipynb
├── 011_llama318BInstruct_1k5e.ipynb
├── 012_llama318BInstruct_5k.ipynb
├── 100_llama31_8b_instruct_1k.ipynb
├── 101_llama31_8b_instruct_1k5e.ipynb
├── 111_llama31_8b_instruct_5k5e.ipynb
├── 120_llama31_8b_instruct_500s.ipynb
├── 121_llama318BInstruct_500s5e.ipynb
├── 130_llama31_8b_instruct_1ks.ipynb
├── 131_llama31_8b_instruct_1ks5e.ipynb
├── 140_llama31_8b_instruct_5ks.ipynb
├── 141_llama31_8b_instruct_5ks5e.ipynb
├── requirements.txt
└── Dockerfile
```

## Requirements & Setup

Python 3.12.5 (Docker), PyTorch, `transformers`, `peft`, `trl`, `accelerate`.

```bash
docker build -t emotion-clf .
docker run --gpus all emotion-clf
```

## Usage

Run the numbered Jupyter notebooks in ascending order for reproducibility. Start with data preparation, then progress through the fine-tuning experiments by dataset size (e.g., `000_prepare_data.ipynb` → `010_llama318BInstruct_1k.ipynb` → `012_llama318BInstruct_5k.ipynb`). Notebooks named with `5e` suffixes correspond to 5-epoch training runs.

## References

B. V. Kok-Shun, J. Chan, G. Peko, and D. Sundaram, "Doing More with Less: Tackling Data Limitations in Emotion Detection in Text Using Generative AI," in *AMCIS 2025 Proceedings*, 2025.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{P5_kok-shun_doing_2025,
  title     = {Doing {More} with {Less}: {Tackling} {Data} {Limitations} in {Emotion} {Detection} in {Text} {Using} {Generative} {AI}},
  booktitle = {{AMCIS} 2025 {Proceedings}},
  author    = {Kok-Shun, Brice Valentin and Chan, Johnny and Peko, Gabrielle and Sundaram, David},
  year      = {2025},
  url      = {https://aisel.aisnet.org/amcis2025/sig_aiaa/sig_aiaa/5},
}
```

</details>
