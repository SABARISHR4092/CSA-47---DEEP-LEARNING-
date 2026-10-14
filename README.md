# Poisoned Pixels: Detecting Adversarial and Backdoor Attacks in Vision Models Before They Strike

A research-grade deep learning security research dashboard and evaluation suite for investigating, simulating, detecting, and explaining adversarial evasion attacks and backdoor trojan injections against image classification vision models.

---

## 1. System Architecture & Research Scope

The platform addresses two distinct attack vectors against deep vision architectures (specifically ResNet-18 trained on CIFAR-10):

| Threat Vector | Attack Phase | Target | Defense Mechanism | Scope |
|---|---|---|---|---|
| **Adversarial Evasion** | Test / Inference | Single input tensor | **Mahalanobis Distance Out-of-Distribution Detector** | Input-level runtime packet filtering |
| **Backdoor Trojans** | Training / Fine-tuning | Model weights | **Activation Clustering** | Model-level / Training-set offline auditing |

### Key Modules:
- **Classifier**: ResNet-18 adapted for CIFAR-10 (3×3 conv1, stride 1, no initial maxpooling, 512-dim penultimate feature representations).
- **Adversarial Lab**: PyTorch implementations of Fast Gradient Sign Method (**FGSM**) and Projected Gradient Descent (**PGD** with $L_\infty$ projection), with amplified perturbation visualizer.
- **Backdoor Analysis**: Controlled 3×3 checkerboard and patch trigger injections, 4-way cross-checkpoint verification matrix, and 2D PCA Activation Clustering with silhouette score analysis.
- **Explainability**: PyTorch **Grad-CAM** attention heatmap generation hooked into `layer4` with JET colormap overlays and scientific interpretation disclaimers.
- **Evaluation & Reports**: Empirical accuracy, Attack Success Rate (ASR) before and after defense, Precision, Recall, F1-score, False Positive Rate (FPR), ROC-AUC curves, and confusion matrix.

---

## 2. Technology Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Recharts
- **Backend**: Python 3.13, FastAPI, Uvicorn, Pydantic v2
- **Machine Learning**: PyTorch 2.14 (CPU/CUDA), Torchvision, Scikit-Learn, SciPy, NumPy, Pillow, Matplotlib

---

## 3. Project Directory Structure

```
DL DEMO/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                  # Normalization, constants, trigger geometry
│   │   ├── main.py                    # FastAPI server & route handlers
│   │   ├── models/
│   │   │   └── classifier.py          # ResNet-18 CIFAR architecture & ModelManager
│   │   ├── attacks/
│   │   │   ├── adversarial.py         # FGSM & PGD PyTorch autograd attacks
│   │   │   └── backdoor.py            # Controlled trigger injection
│   │   ├── defense/
│   │   │   ├── mahalanobis.py         # Penultimate Gaussian OOD anomaly detector
│   │   │   └── activation_clustering.py # 2-Means PCA backdoor representation audit
│   │   ├── explainability/
│   │   │   └── gradcam.py             # Layer4 Grad-CAM hooks & JET colormap overlays
│   │   ├── evaluation/
│   │   │   └── benchmark.py           # Quantitative metrics & ROC curve calculations
│   │   └── data/
│   │       └── cifar10_samples.py     # CIFAR-10 gallery generator & tensor converter
│   ├── scripts/
│   │   └── train_or_calibrate.py      # Model training & Mahalanobis calibration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx             # Telemetry & model toggle header
│   │   │   ├── Overview.tsx           # Threat overview & experiment history
│   │   │   ├── ImageAnalysis.tsx      # CIFAR gallery, upload & OOD check
│   │   │   ├── AdversarialLab.tsx     # FGSM/PGD lab & perturbation visualizer
│   │   │   ├── BackdoorAnalysis.tsx   # Trigger detonation & activation clustering
│   │   │   ├── Explainability.tsx     # Grad-CAM attention maps & disclaimer
│   │   │   ├── EvaluationReports.tsx  # Metrics, ASR comparison & ROC curves
│   │   │   └── ModelInfo.tsx          # Technical specifications
│   │   ├── api.ts                     # Typed REST API client
│   │   ├── types.ts                   # TypeScript interfaces
│   │   ├── App.tsx                    # Root application component
│   │   └── index.css                  # Dark navy cybersecurity design system
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

---

## 4. Setup and Running Locally

### Step 1: Backend Setup
```bash
# Navigate to workspace
cd "c:\Users\sabar\Desktop\DL DEMO"

# Install Python requirements
python -m pip install -r backend/requirements.txt

# Start FastAPI backend server
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API documentation is available at:
`http://127.0.0.1:8000/docs`

### Step 2: Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (already initialized)
npm install

# Start Vite development server
npm run dev
```
The web dashboard opens at:
`http://localhost:5173`

---

## 5. Security Distinctions & Research Disclaimers

### Input Alerts vs. Model Findings
1. **Adversarial Alerts (Input-Level)**: Monitored at inference time via **Mahalanobis Distance** ($M(x) > \tau$). Flagged when an incoming image tensor produces out-of-distribution latent activations in `layer4`.
2. **Backdoor Findings (Model-Level)**: Audited at training/validation time via **Activation Clustering**. Flagged when penultimate representations within a single class exhibit bimodal cluster separation (silhouette $> 0.35$). Because backdoor triggers are spatially small (9 pixels), input-level detectors may permit them as in-distribution; model audits are required.

### Explainability Notice
Grad-CAM heatmaps highlight spatial regions in the penultimate feature maps (`layer4`) that positively correlate with the class activation score. High activation localized on non-semantic artifacts (e.g. corner pixels) provides visual evidence of anomalous attention patterns, but heatmaps alone do not mathematically or cryptographically prove the presence of an active backdoor.
