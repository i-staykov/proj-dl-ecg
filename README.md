# Interpretable Heartbeat Classification from ECG

Detecting life-threatening ventricular arrhythmias in single heartbeats, where **missing one costs more
than a false alarm** — and then showing what the model actually looked at.

Course project for **Deep Learning**, Hasso-Plattner-Institut, Feb–Mar 2026.
Gustav Freitag · Ivaylo Staykov · supervised by Pedro de Macedo Gomes.

| | MLP baseline | **1D-CNN** |
| :--- | ---: | ---: |
| Ventricular **recall** *(the clinical objective)* | 0.957 | **0.976** |
| Ventricular precision | 0.45 | **0.83** |
| Ventricular F1 | 0.62 | **0.90** |
| Overall accuracy | 0.80 | **0.939** |

**[→ Read the notebook](DL_Project_8_ECG.ipynb)**  ·  **[→ Full report (PDF)](DL_Project_8_ECG_Report.pdf)**

---

## The problem

The MIT-BIH Arrhythmia database gives 109,446 heartbeats as 187-step waveforms at 125 Hz, labelled into
five AAMI diagnostic classes. It is severely imbalanced — normal beats dominate, and the dangerous ones are
rare.

That imbalance is the whole problem. **Optimising accuracy produces a model that ignores the class that
kills people.** Class 2 (`V`, ventricular ectopic) is the one that matters clinically: these beats
originate in the pumping chambers and can precede ventricular fibrillation. So the target metric is recall
on class 2, not accuracy — and accuracy is expected to *fall*.

## What we did

**Cost-sensitive learning rather than resampling.** We rejected undersampling (throws away real ECG
variance) and SMOTE (synthetic medical time-series introduces artefacts a model can memorise). Instead the
data distribution is left untouched and the *loss* is reweighted by inverse class frequency, passed into
`CrossEntropyLoss`.

**A clinical severity multiplier on top.** Pure inverse-frequency weighting is clinically wrong: it hands
the largest weight to the *rarest* class (fusion beats), not the most *dangerous* one. We apply an
additional ×5.0 multiplier to class 2 so the penalty for missing a ventricular beat outweighs the
statistical weight of less dangerous arrhythmias. This single decision is what drives the results.

**Two architectures, identical training loop.** An MLP baseline (187 → 128 → 64 → 32 → 5, ReLU, dropout
0.2) treats the beat as a flat vector. The 1D-CNN treats it as a temporal signal: three
`Conv1d → BatchNorm → ReLU → MaxPool1d` blocks (32 → 64 → 128 filters) compressing 187 steps to ~23,
then two FC layers. Same loss, same optimiser, same epochs, so the comparison isolates the architecture.

## What we found

**The MLP hits the recall target and fails on precision** — 0.957 recall at 0.45 precision. It catches the
dangerous beats but cries wolf constantly, which in a real ward means alarm fatigue.

**The CNN fixes precision without giving up recall.** 0.83 precision at 0.976 recall, nearly doubling
precision. The reason is morphological: sliding filters detect *local* structure — how wide the QRS complex
is, whether the T-wave is inverted — while the MLP only ever sees 187 independent amplitudes.

**The saliency maps confirm the mechanism rather than just asserting it.** Backpropagating the ventricular
class score to the input shows the CNN concentrating on the abnormal QRS complex in steps 0–50 and
correctly ignoring the flat resting phase afterwards. The MLP's attention is scattered across the whole
signal, including the empty tail. The architecture argument is visible in the gradients, not only in the
metrics.

## Running it

The notebook is self-contained and was written for Colab (free T4 GPU): `Runtime → Change runtime type →
T4 GPU`. It pulls the dataset via `kagglehub`, and falls back to CUDA → MPS → CPU automatically.

```
numpy · pandas · matplotlib · seaborn · scikit-learn · torch · kagglehub
```

Section 17 saves the trained CNN weights and simulates real-time monitoring, streaming unseen beats through
the model one at a time with predicted versus true diagnosis.

## Data

- **Original:** [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Preprocessed:** [Kaggle ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data)

## Honest limits

- The ×5.0 severity multiplier is **hand-set, not tuned.** It encodes a clinical judgement we made, not one
  we derived; a real deployment would set it from the cost ratio a hospital actually assigns.
- Evaluation uses the standard Kaggle train/test split. **No patient-level separation**, so beats from one
  patient can appear in both splits — the usual optimistic bias on this dataset. Results are comparable to
  published work on the same split, not to prospective clinical performance.
- Saliency maps show input sensitivity, not causation. They are evidence about the model, not about
  cardiology.
