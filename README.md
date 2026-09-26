<div align="center">

# 🎓 Project ASPIRE
### Academic Success Prediction through Intelligent Risk Evaluation

[![R Version](https://img.shields.io/badge/R-%E2%89%A5%204.2-276DC3?style=for-the-badge&logo=r&logoColor=white)](#)
[![Model](https://img.shields.io/badge/Model-Random%20Forest%20Ensemble-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Validation](https://img.shields.io/badge/Validation-10--Fold%20CV-success?style=for-the-badge)](#)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.976-brightgreen?style=for-the-badge)](#)
[![Recall](https://img.shields.io/badge/At--Risk%20Recall-84.04%25-blueviolet?style=for-the-badge)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>Shifting higher education advising from reactive post-exam damage control to proactive, mid-term targeted intervention.</b>
</p>

[System Overview](#-system-overview) •
[Architecture](#-architecture--workflow) •
[Feature Engineering](#-feature-engineering--sei) •
[Model Benchmarks](#-model-benchmarks) •
[Actionable Insights](#-actionable-behavioral-insights) •
[Triage Playbook](#-risk-stratification--alert-playbook) •
[Installation & Usage](#-quick-start)

---

</div>

## 📌 System Overview

**Project ASPIRE** is an end-to-end predictive machine learning framework developed in **R** to identify students vulnerable to academic failure ($\text{Exam Score} \le 65$, lower quartile) weeks before final examinations. 

By analyzing an educational benchmark dataset of **6,607 student records** across **20 multidimensional attributes**, ASPIRE balances missingness imputation, synthesizes cross-domain interaction metrics, and calibrates classification cutoffs to maximize intervention recall.

### 🌟 Key Performance Highlights

| Metric | Baseline Default ($\tau = 0.50$) | ASPIRE Calibrated ($\tau^* = 0.38$) | Advising Impact |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | `0.976` | **`0.976`** | Top-tier class separability across risk spectra |
| **At-Risk Detection Recall** | `77.10%` | **`84.04%`** | **+6.94% reduction** in unnoticed failing students |
| **F1-Score** | `0.8356` | **`0.8658`** | Optimal balance between recall & advisor burnout |
| **Behavioral Dominance** | `> 66%` | **`> 66%`** | Over 2/3 of variance is driven by modifiable habits |

---

## 🏗 Architecture & Workflow

The pipeline executes five automated stages from raw benchmark ingestion to diagnostic alert dispatches:

```mermaid
flowchart TD
    classDef raw fill:#f8f9fa,stroke:#6c757d,stroke-width:1px,color:#212529;
    classDef prep fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1;
    classDef model fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef eval fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef triage fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;

    A[Raw Student Cohort<br/>6,607 Records x 20 Features]:::raw --> B[Preprocessing & Mode Imputation]:::prep
    B --> C[Feature Engineering:<br/>Study Efficiency Index - SEI]:::prep
    
    C --> D[10-Fold Cross-Validation Suite]:::model
    D --> D1[Decision Trees CART]:::model
    D --> D2[Logistic Regression]:::model
    D --> D3[Random Forest Ensemble]:::model
    
    D3 --> E[Threshold Optimization<br/>Shift Cutoff: 0.50 -> 0.38]:::eval
    E --> F[Performance Target Met<br/>Recall: 84.04% | AUC: 0.976]:::eval
    
    F --> G{Automated Risk Stratification}:::triage
    G -->|P >= 0.60| H[🚨 High Risk: 1-on-1 Advising & Tutoring]:::triage
    G -->|0.38 <= P < 0.60| I[⚠️ Moderate Risk: Peer Study Pods]:::triage
    G -->|P < 0.38| J[✅ Low Risk: Standard Self-Paced Track]:::triage
```

---

## 🔬 Feature Engineering & SEI

ASPIRE monitors 20 features across 4 foundational domains:
1. **Academic Foundations:** Historical grade trends, assignment submission rates, continuous assessment marks.
2. **Dynamic Behaviors:** Attendance percentage, weekly self-study volume, library checkouts.
3. **Lifestyle & Circadian Health:** Sleep routine consistency, screen time balance, extracurricular load.
4. **Socio-Environmental Context:** Peer group engagement, resource access tiers.

### Study Efficiency Index (SEI)
Rather than evaluating study hours in a vacuum, ASPIRE synthesizes the **Study Efficiency Index (SEI)** to measure knowledge absorption adjusted for lifestyle stability:

$$\text{SEI} = \frac{\text{Weekly Study Hours} \times \text{Continuous Assessment Performance}}{\text{Sleep Variance Index} + \epsilon}$$

Where $\epsilon$ represents a smoothing factor preventing division by zero for students with completely uniform sleep schedules.

---

## 📊 Model Benchmarks

All models were evaluated via stratified 10-fold cross-validation targeting the lower quartile ($\text{Exam Score} \le 65$):

```
Model Comparison (ROC-AUC & At-Risk Recall)
────────────────────────────────────────────────────────────────────────
Random Forest (τ* = 0.38) [████████████████████] AUC: 0.976 | Recall: 84.04%
Random Forest (τ = 0.50)  [████████████████████] AUC: 0.976 | Recall: 77.10%
Logistic Regression       [██████████████████  ] AUC: 0.923 | Recall: 76.85%
Decision Tree (CART)      [████████████████    ] AUC: 0.891 | Recall: 72.30%
────────────────────────────────────────────────────────────────────────
```

### Comprehensive Results Matrix

| Architecture | Operational $\tau$ | Accuracy | ROC-AUC | Recall | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Trees (CART)** | $0.50$ | 88.1% | 0.891 | 72.30% | 0.7812 | 0.7510 |
| **Logistic Regression** | $0.50$ | 90.4% | 0.923 | 76.85% | 0.8240 | 0.7953 |
| **Random Forest Baseline** | $0.50$ | **94.8%** | **0.976** | 77.10% | **0.9120** | 0.8356 |
| **Random Forest (ASPIRE Optimized)** | $\mathbf{0.38}$ | 93.6% | **0.976** | **84.04%** | 0.8930 | **0.8658** |

> **Why shift to $\tau^* = 0.38$?**  
> In educational retention, **false negatives carry high institutional and human cost** (failing to detect a struggling student), while **false positives are low-cost** (offering supplemental tutoring to a borderline student). Shifting the threshold expands the safety net without swamping advising capacity.

---

## 💡 Actionable Behavioral Insights

Mean Decrease in Gini importance demonstrates that **changeable behavioral habits drive >66% of target variance**, debunking the premise that academic trajectory is predestined by static demographics:

```text
Actionable Behavioral Drivers vs. Static Background (Gini Importance)
────────────────────────────────────────────────────────────────────────
[Dynamic] Attendance Rate              ██████████████████  (36.2%)
[Dynamic] Study Efficiency Index (SEI)  ███████████████     (29.9%)
[Academic] Continuous Assessment Marks  ███████             (14.1%)
[Lifestyle] Sleep Consistency Metric    ████                (9.3%)
[Static] Socio-Demographic Factors      ███                 (6.8%)
[Other] Uncorrelated Environment Data   ██                  (3.7%)
────────────────────────────────────────────────────────────────────────
Actionable Variance Share: 66.1%
```

---

## 🚦 Risk Stratification & Alert Playbook

ASPIRE automatically outputs triage rosters for advisors categorized into actionable risk tiers:

```
 Cohort Risk Distribution (N = 6,607)
 ┌───────────────────────┬────────────────────────┬──────────────────────┐
 │   🔴 High Risk (14%)  │   🟡 Moderate Risk (18%)│   🟢 Low Risk (68%)   │
 └───────────────────────┴────────────────────────┴──────────────────────┘
  P(Risk) >= 0.60         0.38 <= P(Risk) < 0.60   P(Risk) < 0.38
```

| Tier | Probability Range | Action Items & Faculty Intervention Playbook |
| :---: | :---: | :--- |
| **🔴 HIGH RISK** | $P \ge 0.60$ | **Immediate Triage:** Mandatory 1-on-1 advisor consultation within 5 business days, diagnostic learning disability check, and assignment to supervised weekly study halls. |
| **🟡 MODERATE RISK** | $0.38 \le P < 0.60$ | **Targeted Scaffolding:** Automated invitation to peer tutoring pods, supplemental course material access, and bi-weekly attendance monitor checkpoints. |
| **🟢 LOW RISK** | $P < 0.38$ | **Autonomous Progress:** Standard cohort advising cadence, automated milestone encouragement emails, and elective enrichment modules. |

---

## 📁 Repository Structure

```plaintext
project-aspire/
├── data/
│   ├── raw/                      # Raw student benchmark records (6,607 entries)
│   └── processed/                # Imputed, scaled, and split matrices
├── R/
│   ├── 01_data_preprocessing.R   # Mode imputation & categorical encoding
│   ├── 02_feature_engineering.R  # Feature generation (SEI, Sleep Indices)
│   ├── 03_model_benchmarking.R   # CART, Logistic Regression, Random Forest
│   ├── 04_threshold_tuning.R     # Precision-Recall & ROC-AUC calibration
│   └── 05_alert_generator.R      # Automated CSV alert list compilation
├── models/
│   └── rf_aspire_optimized.rds   # Trained serialized model artifact
├── reports/
│   ├── figures/                  # ROC Curves, PR curves, Gini ranking plots
│   └── alerts/                   # Generated risk-tiered advisor rosters
├── config.yml                    # Pipeline parameters, thresholds, and seeds
├── run_pipeline.R                # Master CLI execution script
└── README.md
```

---

## 🚀 Quick Start

### 1. Requirements
- **R** $\ge$ 4.2.0
- Recommended dependencies:
  ```R
  install.packages(c("tidyverse", "caret", "randomForest", "pROC", "ROCR", "yaml", "knitr"))
  ```

### 2. Installation
```bash
git clone https://github.com/your-username/project-aspire.git
cd project-aspire
```

### 3. Execution
Run the complete pipeline end-to-end:
```bash
Rscript run_pipeline.R
```

### 4. Inspect Triage Outputs
Once execution completes, view the prioritized alerts in:
```bash
head -n 20 reports/alerts/advising_action_roster.csv
```

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{project_aspire_2024,
  title   = {Project ASPIRE: Academic Success Prediction through Intelligent Risk Evaluation},
  author  = {ASPIRE Research & Engineering Team},
  journal = {Educational Data Mining & Advising Analytics Repository},
  year    = {2024}
}
```

<div align="center">
  <sub>Built with ❤️ using R and Machine Learning for proactive student retention.</sub>
</div>
