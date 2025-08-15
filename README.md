# ClearML Dataset: ML-Based Pathloss Radio Map Predictor

This repository contains the dataset and upload script for a machine learning pipeline that predicts **radio pathloss** using environmental and signal data. The project is designed for seamless tracking and reproducibility via [ClearML](https://clear.ml/).

---

## 📁 Project Structure

```
ClearML/
├── code/               # Scripts for data processing or model training (optional extension)
├── data/               # Raw or processed pathloss-related data
├── data-upload.py      # Script to upload dataset to ClearML
└── README.md           # This documentation
```

---

## 🚀 Getting Started

### 1. Install ClearML
```bash
pip install clearml
```

### 2. Upload Dataset to ClearML
```bash
python data-upload.py
```

This script uses the `clearml.Dataset` API to upload and version your dataset.

---

## 💡 Use Cases

- ML-based wireless pathloss prediction
- Radio map reconstruction
- Coverage and connectivity estimation

---

## 🔗 ClearML Integration

This dataset is versioned and tracked via ClearML for:

- Experiment reproducibility
- Easy team collaboration
- Data version control

```python
from clearml import Dataset

dataset = Dataset.get(dataset_name="pathloss_v1", dataset_project="RadioMap/Pathloss")
local_copy = dataset.get_local_copy()
```

---

## 👥 Contributors

Maintained by the **CCI xG Testbed** research team.  
For questions, reach out via [GitHub Issues](https://github.com/CCI-xGTestbed/).

---

## 📜 License

For academic/research use only. Contact us for broader access or commercial usage.
