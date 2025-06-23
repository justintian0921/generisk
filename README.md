# GeneRisk – Mutation Pathogenicity Predictor
---
**GeneRisk** is an interactive machine learning app that predicts whether a genetic mutation is likely to be **Pathogenic** or **Benign**, using a pre-trained XGBoost model and 15 of the most impactful genomic features.

Built with **Python**, **Streamlit**, and **SHAP**, this app provides:

- 🧠 Real-time prediction from user-inputted mutation features
- 📊 Clear probability-based confidence score
- 🧾 Text-based SHAP explanation of which features influenced the prediction—and how

---

### 🚀 Features

- ✅ XGBoost classifier trained on ClinVar-derived mutation data  
- 🔍 15 top-ranked features selected via feature importance  
- 🧪 SHAP explainability: shows feature-level contribution to the outcome  
- 🧬 Manual input form for testing hypothetical mutations  
- 👆 Prediction triggered only when **Submit** is pressed (no auto-run)  
- 🎨 Custom dark theme + wide layout for readability  

---

### ▶️ How to Run

Run this model with:

```bash
streamlit run app.py
