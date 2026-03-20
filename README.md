# EEGSchizNet

**AI-assisted schizophrenia detection from resting-state EEG**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-yellow.svg)]()

A multi-branch deep learning system that takes a 5-minute resting-state EEG recording and produces a structured clinical risk report in under 3 minutes. Built on top of — and as a systematic correction of — the Nature 2025 image-encoding approach to EEG schizophrenia classification.

> **This is a research prototype. All outputs require confirmation by a qualified psychiatrist before any clinical action is taken.**

---

## What this is

EEGSchizNet analyses EEG recordings through three parallel branches:

- **Branch 1 (Spectral CNN)** — 2-channel CWT scalogram (magnitude + phase) through a ResNet-style CNN. Detects gamma deficit, alpha suppression, delta elevation.
- **Branch 2 (EEG-Conformer + Microstates)** — Parallel CNN/Transformer for temporal dynamics + k-means microstate analysis. Detects P300 latency shift and microstate C excess.
- **Branch 3 (PLI-GAT)** — Phase Lag Index connectivity graph processed by a 3-layer Graph Attention Network. Detects fronto-parietal disconnection.

All three branch outputs are combined via cross-attention fusion, passed through a 5-layer specificity defence system, and the result is rendered as a 2-page clinical PDF report with a GradCAM topographic brain map and 15 quantified biomarkers.

---

## Why this exists — the problem with the original paper

This project began from the Nature Scientific Reports 2025 paper *"Schizophrenia detection from EEG using image encoding."* The paper proposed converting EEG scalograms into images and applying a CNN — a genuinely good idea. We found five methodological problems that made its results clinically invalid:

| Problem | Effect |
|---|---|
| Epoch-level train/test split (data leakage) | Inflates accuracy 5–8 pp; model memorises subjects, not disease |
| Phase channel discarded (magnitude-only CWT) | Loses all PLI/synchrony information |
| ImageNet pre-trained weights | Domain mismatch — photographs ≠ EEG scalograms |
| No electrode topology (pixels treated as independent) | Fronto-parietal connectivity invisible to the model |
| No uncertainty quantification or specificity protection | Miscalibrated output dangerous for clinical use |

EEGSchizNet fixes all five. The full reasoning, rejected alternatives, and design decisions are documented in [`docs/Methodology.docx`](docs/Methodology.docx).

---

## Repository structure

```
eegschiznet/
├── src/
│   ├── preprocessing.py          # Stage 02: bandpass, ICA, epoching
│   ├── backbone/
│   │   ├── labram_finetune.py    # Stage 03: LaBraM fine-tuning
│   │   └── patch_tokenizer.py
│   ├── branches/
│   │   ├── branch_spectral.py   # Stage 04: CWT + ResNet CNN
│   │   ├── cwt_extractor.py
│   │   ├── resnet_cnn.py
│   │   ├── branch_temporal.py   # Stage 05: EEG-Conformer + microstates
│   │   ├── eeg_conformer.py
│   │   ├── microstate_extractor.py
│   │   ├── branch_graph.py      # Stage 06: PLI-GAT
│   │   ├── pli_computer.py
│   │   └── gat_model.py
│   ├── fusion/
│   │   ├── cross_attention.py   # Stage 07: cross-attention fusion
│   │   └── fusion_layer.py
│   ├── classifier.py            # Stage 08: classifier + asymmetric loss
│   ├── safety/
│   │   ├── asymmetric_loss.py   # L1: FP penalty 4×
│   │   ├── normative_vae.py     # L3: healthy-only VAE
│   │   ├── biomarker_gating.py  # L4: 2+ biomarkers required
│   │   ├── zone_classifier.py   # L2: 3-zone output
│   │   └── temperature_scaling.py
│   ├── training/
│   │   ├── train_loso.py        # Stage 09: LOSO cross-validation
│   │   ├── data_augmentation.py
│   │   ├── early_stopping.py
│   │   └── metrics.py
│   ├── biomarkers/
│   │   ├── biomarker_extractor.py  # Stage 10: all 15 biomarkers
│   │   ├── normative_database.py
│   │   ├── band_power.py
│   │   ├── connectivity_metrics.py
│   │   └── erp_estimator.py
│   ├── explainability/
│   │   ├── gradcam.py           # Stage 11: GradCAM saliency
│   │   └── topomap_renderer.py
│   └── report/
│       ├── report_generator.py  # Stage 12: clinical PDF
│       ├── pdf_templates.py
│       ├── chart_renderer.py
│       └── language_rules.py
├── api/
│   ├── app.py                   # Stage 13: FastAPI endpoints
│   ├── worker.py                # Celery background pipeline
│   ├── schemas.py
│   └── security.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── UploadZone.jsx
│           └── ReportViewer.jsx
├── notebooks/
│   ├── EEGSchizNet_Colab_T4.ipynb   # Full pipeline, T4-optimised
│   └── (more notebooks coming)
├── docs/
│   ├── Methodology.docx             # Design decisions, rejected approaches, future scope
│   ├── Pipeline_Spec.docx           # All 13 stages, code files, pitfalls
│   ├── Complete_Documentation.docx  # Full technical reference
│   └── Presentation.pptx            # 13-slide overview
├── data/
│   ├── raw/          # place .mat files here (not committed)
│   ├── processed/    # cached scalograms — auto-generated
│   └── normative/    # normative stats database — auto-generated
├── eegschiznet_pipeline.py   # single-file end-to-end pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quick start

### Option A — Google Colab (recommended for first run)

1. Open [`notebooks/EEGSchizNet_Colab_T4.ipynb`](notebooks/EEGSchizNet_Colab_T4.ipynb) in Google Colab
2. Set runtime to **GPU → T4** (Runtime → Change runtime type)
3. Run cells top to bottom — Cell 4 generates synthetic data if you don't have the real dataset yet
4. Full pipeline with synthetic data completes in ~30 minutes on T4
5. Real dataset (28 subjects, LOSO) takes ~2–3 hours

### Option B — local install

```bash
git clone https://github.com/yourusername/eegschiznet.git
cd eegschiznet

pip install -r requirements.txt

# place Olejarczyk .mat files in data/raw/
# download from: https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441

python eegschiznet_pipeline.py
```

### Option C — Docker (web interface)

```bash
docker-compose up
# open http://localhost:8000
# upload a .edf file, enter patient age and sex, download PDF report
```

---

## Dataset

**Primary:** Olejarczyk & Jernajczyk, PLOS ONE 2017  
28 subjects (14 healthy, 14 schizophrenia) · 19-channel · 250 Hz · 5 min resting-state  
DOI: [10.18150/repod.0107441](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441)  
Free, public, no registration required. Download all `.mat` files and place in `data/raw/`.

**Secondary (for extended validation):** COBRE dataset, 146 subjects  
Access via [Mind Research Network](http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)

---

## Installation

```bash
# Core ML
pip install torch>=2.0 torchvision torch-geometric torch-scatter torch-sparse
pip install transformers>=4.38 huggingface_hub

# EEG processing
pip install mne==1.6.1 PyWavelets==1.5.0 scipy numpy

# Biomarker analysis
pip install antropy nolds networkx scikit-learn

# Explainability and reporting
pip install captum>=0.7.0 matplotlib Pillow reportlab

# API and web
pip install fastapi uvicorn celery[redis] python-multipart cryptography
```

Or all at once:

```bash
pip install -r requirements.txt
```

torch-geometric requires matching your CUDA version. If the above fails:

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html
```

---

## The two rules that must never be broken

These are documented throughout the codebase, but worth stating at the top:

**Rule 1 — Subject isolation in LOSO**

Every epoch from a subject must be entirely in training or entirely in test. Never split. The assertion below is line 1 of the training loop and will throw if violated:

```python
assert len(set(train_subject_ids) & set(test_subject_ids)) == 0
```

Violating this inflates accuracy by 5–8 percentage points but produces a model that fails on any genuinely new patient.

**Rule 2 — MC Dropout at inference**

```python
model.train()   # NOT model.eval()
with torch.no_grad():
    probs = [softmax(model(x)) for _ in range(50)]
```

`model.eval()` disables dropout. All 50 Monte Carlo passes become identical. The uncertainty interval collapses to zero, producing false certainty on every prediction.

---

## Expected performance (honest LOSO)

| Metric | Target | Notes |
|---|---|---|
| Accuracy | 97–98% | Subject-level, LOSO |
| Specificity | 97%+ | Healthy patients correctly cleared |
| Sensitivity | 93–95% | SCZ correctly detected |
| AUC-ROC | 0.97–0.98 | |
| Calibration error (ECE) | < 3% | After temperature scaling |
| Processing time | < 3 min | T4 GPU |

---

## Architecture overview

```
Raw EEG (.edf)
    │
    ▼
Preprocessing  ──  bandpass · notch · avg-ref · ICA · epoch · reject
    │
    ▼
LaBraM backbone  ──  ICLR 2024 · pre-trained 2500+ hrs · fine-tune last 4 layers
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
Branch 1           Branch 2           Branch 3
Spectral CNN     EEG-Conformer      PLI-GAT Network
2-ch CWT         + Microstates      19×19 PLI graph
ResNet blocks    CNN+Transformer    3-layer GAT
256-dim          256-dim            256-dim
    │                  │                  │
    └──────────────────┴──────────────────┘
                       │
                       ▼
            Cross-attention fusion  ──  Branch 1 = Query, B2+B3 = K/V
                       │
                       ▼
         Classifier + 5-layer defence
           L1: Asymmetric Loss (FP ×4)
           L2: 3-zone output
           L3: Normative VAE
           L4: Biomarker gating
           L5: Demographic norms
                       │
                       ▼
              3-zone risk score
            + 15 biomarkers
            + GradCAM brain map
            + 2-page PDF report
```

---

## Documentation

All docs are in the `docs/` folder.

| File | What it covers |
|---|---|
| `Methodology.docx` | Why this architecture: the original paper's flaws, every design decision with justification, 10 rejected alternatives with reasons, future scope (near/medium/long term) |
| `Pipeline_Spec.docx` | All 13 stages: inputs, outputs, code files, libraries, pitfalls and how to fix them |
| `Complete_Documentation.docx` | Full technical reference: all 15 papers, all 15 biomarkers, clinical workflow, competitive landscape |
| `Presentation.pptx` | 13-slide overview for stakeholders and clinical partners |
| `EEGSchizNet_Colab_T4.ipynb` | Complete working notebook, every line commented, runs on free Colab T4 |

---

## Key papers

The architecture is built on 15 peer-reviewed papers. The most important:

- **LaBraM** (backbone): Jiang, Zhao, Lu · ICLR 2024 · [arXiv 2405.18765](https://arxiv.org/abs/2405.18765)
- **Dataset + PLI biomarkers**: Olejarczyk & Jernajczyk · PLOS ONE 2017 · [DOI 10.1371/journal.pone.0188629](https://doi.org/10.1371/journal.pone.0188629)
- **PLI vs PLV**: Stam, Nolte, Daffertshofer · Clinical Neurophysiology 2007 · [DOI 10.1016/j.clinph.2006.09.020](https://doi.org/10.1016/j.clinph.2006.09.020)
- **EEG-Conformer**: Song et al. · IEEE TNSRE 2023 · [DOI 10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)
- **Microstates in SCZ**: Tomescu et al. · Schizophrenia Bulletin 2014 · [DOI 10.1093/schbul/sbt246](https://doi.org/10.1093/schbul/sbt246)
- **Asymmetric Loss**: Ben-Baruch et al. · ICCV 2021 · [arXiv 2009.14119](https://arxiv.org/abs/2009.14119)
- **Graph Attention Networks**: Velickovic et al. · ICLR 2018 · [arXiv 1710.10903](https://arxiv.org/abs/1710.10903)
- **LOSO validation requirement**: Rahul & Sharma · Frontiers Human Neuroscience 2024 · [DOI 10.3389/fnhum.2024.1347082](https://doi.org/10.3389/fnhum.2024.1347082)
- **MC Dropout**: Gal & Ghahramani · ICML 2016 · [arXiv 1506.02142](https://arxiv.org/abs/1506.02142)
- **GradCAM**: Selvaraju et al. · ICCV 2017 · [arXiv 1610.02391](https://arxiv.org/abs/1610.02391)

Full citation list with annotations in `docs/Complete_Documentation.docx`.

---

## Roadmap

**Now (prototype)**
- [x] Branch 1 (Spectral CNN) — complete, tested on T4
- [x] Branch 2 (EEG-Conformer + Microstates) — complete
- [x] Branch 3 (PLI-GAT) — complete
- [x] Cross-attention fusion — complete
- [x] Asymmetric Loss + 3-zone output — complete
- [x] MC Dropout uncertainty — complete
- [x] GradCAM saliency + clinical report — complete
- [x] Colab T4 notebook — complete
- [ ] Normative VAE training on healthy subjects
- [ ] Temperature scaling calibration
- [ ] Web interface deployment

**Phase 1 (months 3–9) — clinical validation**
- [ ] IRB ethics approval
- [ ] Prospective collection at 1–2 partner hospitals (150–300 subjects)
- [ ] COBRE dataset validation
- [ ] CDSCO regulatory filing preparation
- [ ] Peer-reviewed validation paper submission

**Phase 2 (months 9–18) — regulatory + pilot**
- [ ] CDSCO Class B registration
- [ ] 3–5 centre pilot deployment
- [ ] EMR / HL7 FHIR integration

---

## Limitations

This is a research prototype with known constraints:

- **Training data**: 28 subjects (Olejarczyk). The model has not seen the biological diversity of schizophrenia across all ages, ethnicities, and medications.
- **Single-centre**: All validation is from one research group. Cross-site validation on different EEG hardware is required before clinical deployment.
- **Resting state only**: P300 and MMN are estimated from resting EEG. A proper clinical protocol would add a brief oddball paradigm.
- **Binary diagnosis**: Does not distinguish schizophrenia from other psychotic disorders (bipolar with psychosis, schizoaffective disorder).
- **Medication confound**: The Olejarczyk cohort is medicated. The model detects schizophrenia-on-medication, not drug-naive schizophrenia.

---

## Contributing

Pull requests are welcome. Before submitting:

1. Run the test suite: `pytest tests/`
2. Ensure LOSO subject isolation assertion passes on your changes
3. Check that GradCAM outputs show frontal electrode saliency on SCZ test subjects (biological validation)
4. Do not remove or weaken the 5-layer specificity defence without documented justification

For major changes, open an issue first to discuss.

---

## License

MIT License — see [LICENSE](LICENSE).

The Olejarczyk dataset has its own license — see the [dataset page](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441) before commercial use.

---

## Citation

If you use EEGSchizNet in your research, please cite the foundational dataset and the LaBraM backbone:

```bibtex
@article{olejarczyk2017graph,
  title={Graph-based analysis of brain connectivity in schizophrenia},
  author={Olejarczyk, Elzbieta and Jernajczyk, Wojciech},
  journal={PLOS ONE},
  year={2017},
  doi={10.1371/journal.pone.0188629}
}

@inproceedings{jiang2024large,
  title={Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI},
  author={Jiang, Wei-Bang and Zhao, Li-Ming and Lu, Bao-Liang},
  booktitle={ICLR},
  year={2024}
}
```

---

*EEGSchizNet is not a medical device. It is a research prototype for decision support. No output constitutes a clinical diagnosis.*
