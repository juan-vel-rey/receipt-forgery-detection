# Receipt Forgery Detection — Three-Judge Gemini Ensemble

> **Dataset**: Find It Again Receipt Dataset for Document Forgery Detection  
> **Model**: `gemini-2.0-flash` × 3 specialised judges  
> **Output**: `{ label, confidence (0–100), reasons }`

---

## Project Structure

```
.
├── data/
│   ├── real/          ← genuine receipts (.jpg / .png / .webp …)
│   └── fake/          ← forged  receipts
|── notebooks/
│   ├── receipt_eda_evaluation.ipynb  ← EDA + evaluation notebook
├── results/           ← auto-created by the pipeline
│   ├── predictions.csv
│   ├── predictions.json
│   ├── run.log
│   ├── summary_report.txt
│   └── *.png          ← EDA & evaluation charts
├── receipt_forgery_detector.py   ← main pipeline
└── requirements.txt
```

---

## Architecture
It consists of a three-judge ensemble built on top of Google's `gemini-2.0-flash` model. Each judge is an independent instance of the same model configured with a dedicated system prompt and a distinct temperature 

## The Three Judges

| Judge | Temperature | Responsibility |
|-------|------------|----------------|
| **Visual Anomaly Inspector** | 0.1 | Pixel-level artifacts, font rendering, copy-paste traces, compression noise |
| **Semantic Logic Auditor** | 0.5 | Arithmetic validation (totals, tax, line items), date/merchant coherence |
| **Layout Forensics Analyst** | 0.8 | Column alignment, spacing irregularities, POS template deviations |

**Voting**: Confidence majority-vote.
Decision rules (in priority order):
      · REAL      — at least 2 judges voted REAL
      · FAKE      — at least 2 judges voted FAKE
      · UNCERTAIN — no label reached 2 votes (split across all three labels,
                    or too many UNCERTAIN votes prevented a clear majority)

---

## Output JSON Schema

```json
{
  "label":      "FAKE | REAL | UNCERTAIN",
  "confidence": 87,
  "reasons": [
    "[VISUAL]   Font kerning irregularity in the 'Total' field",
    "[SEMANTIC] Subtotal $24.97 + Tax $3.12 ≠ Total $27.43 (off by $0.34)",
    "[LAYOUT]   Price column misaligned by ~4px from row 3 onward"
  ],
  "evidence_regions": [
          "GST @6% field",
          "Total Amount field",
          "GST Summary"
  ]
}
```

---

## AI tools usage
- Claude (Sonnet 4.6) was used to design the project to work with the Find it again - Receipt Dataset for Document Forgery Detection
- GitHub Copilot in VS Code was mainly used for single-line completions, edit suggestions to full function implementations.
- Claude vs Copilot: When updating or suggesting new requirements, both Cluade and Copilot were used but Claude provided slightly better implementations in this iterative process.
- Claude was used to generate the SVG diagram to summarize the workflow of this project.

---

## Quickstart
### 0. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1. Install dependencies
```bash
pip install -r requirements.txt

# Optional — for OCR-based receipt total extraction in the notebook
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
```

### 2. Set your Gemini API key
```bash
export GEMINI_API_KEY="your_key_here"
```

### 3. Set you Kaggle API credentials
- Generate a token: Log in to your Kaggle account and go to the Account tab in your profile settings.
- In the "API" section, click "Create New API Token" to download a kaggle.json file.
- Place the file in the correct directory: Move the downloaded kaggle.json file to the ~/.kaggle/ directory (or C:\Users\<YourUsername>\.kaggle\ on Windows).
- Secure the file permissions (Linux/Mac only): Run the following command in your terminal for security:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Prepare the dataset
```
data/
├── real/
│   ├── r001.jpg
│   └── ...
└── fake/
    ├── f001.jpg
    └── ...
```

### 5. Run the pipeline
```bash
# Full dataset
python receipt_forgery_detector.py

# Custom path
python receipt_forgery_detector.py --dataset /path/to/dataset

# Quick test (first 10 images only)
python receipt_forgery_detector.py --limit 10

# Pass API key directly
python receipt_forgery_detector.py --api-key YOUR_KEY
```

### 5. Open the EDA + Evaluation notebook
```bash
jupyter notebook notebooks/receipt_eda_evaluation.ipynb
```

---

## Notebook Contents

### Phase 1 — Dataset EDA
- Class distribution (bar chart + donut)
- Image dimension statistics (width, height, file size histograms)
- **Receipt totals histogram** (overlaid histogram, KDE density, box plot, ECDF)
  - Uses `pytesseract` OCR if available; falls back to filename heuristics

### Phase 2 — Model Evaluation
- Overall accuracy & `sklearn` classification report
- Confusion matrix heatmap
- Confidence score distributions (correct vs incorrect)
- Per-judge violin plots (Visual / Semantic / Layout)
- Unanimous vs disagreement case analysis
- Summary report saved to `results/summary_report.txt`

---

## Confidence Score Guide

| Range | Meaning |
|-------|---------|
| 85–100 | Multiple clear, unambiguous indicators |
| 65–84  | Consistent but not overwhelming evidence |
| 40–64  | Mixed or inconclusive evidence |
| 20–39  | Very few or weak signals |
| 0–19   | Cannot determine anything meaningful |
