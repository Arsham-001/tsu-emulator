# TSU Emulator - Dependencies & Installation Guide

## ✅ All Required Libraries Installed

This document lists all dependencies required to run the TSU Emulator project.

### **Core Dependencies**

| Library | Version | Purpose |
|---------|---------|---------|
| **numpy** | 2.3.4 | Numerical computing & array operations |
| **scipy** | 1.16.3 | Scientific computing (stats, optimization) |
| **matplotlib** | 3.10.7 | Static 2D visualization |
| **plotly** | 6.4.0 | Interactive 2D/3D visualization |
| **kaleido** | 1.2.0 | Static image export for Plotly |
| **pandas** | 2.3.3 | Data manipulation & analysis |
| **scikit-learn** | 1.7.2 | Machine learning utilities |

### **Installation**

All dependencies are listed in `requirements.txt`. To install:

```bash
# Using the venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Or with the venv activated
pip install -r requirements.txt
```

### **Verification**

To verify all packages are installed correctly:

```bash
.\.venv\Scripts\python.exe -c "import numpy, scipy, matplotlib, plotly, pandas, sklearn; print('✓ All libraries OK')"
```

**Current Status:** ✅ All libraries verified and working

```
NumPy:        2.3.4
SciPy:        1.16.3
Matplotlib:   3.10.7
Plotly:       6.4.0
Pandas:       2.3.3
Scikit-learn: 1.7.2
```

### **What Each Library Does**

#### **NumPy (Numerical Computing)**
- Core array operations for the TSU emulator
- Used in: `tsu_core.py` for all mathematical operations
- Essential for: Langevin dynamics, gradients, random sampling

#### **SciPy (Scientific Computing)**
- Statistical tests (Kolmogorov-Smirnov, normal distributions)
- Used in: `tsu_core.py` validation functions
- Essential for: Distribution analysis, hypothesis testing

#### **Matplotlib (Static Visualization)**
- 2D plotting for traditional scientific papers
- Used in: `tsu_proper_demo.py`, `ising_solver.py`
- Output: `.png` files for publication-quality figures

#### **Plotly (Interactive Visualization)**
- Interactive 2D/3D charts for research presentations
- Used in: `tsu_proper_demo.py`
- Output: `.html` files with zoom, pan, hover capabilities
- Ideal for: Conferences, online presentations, digital papers

#### **Kaleido (Export Engine)**
- Enables Plotly to export static images (PNG, SVG, PDF)
- Used in: Plotly visualization functions
- Essential for: Converting interactive plots to static formats

#### **Pandas (Data Analysis)**
- DataFrames for organizing results
- Used in: Future analysis and reporting
- Useful for: Tabular data, CSV export, statistical summaries

#### **Scikit-learn (Machine Learning)**
- Potential use for dimensionality reduction, clustering
- Used in: Future enhancements
- Available for: Advanced statistical analysis

---

## Running the Code

### **Basic TSU Core Tests**
```bash
.\.venv\Scripts\python.exe test_basic.py
```

### **TSU Proper Demo (Full Analysis)**
```bash
.\.venv\Scripts\python.exe tsu_proper_demo.py
```

### **Ising Solver (Optimization)**
```bash
.\.venv\Scripts\python.exe ising_solver.py
```

---

## Output Files Generated

Each demo generates:

- **Plotly Interactive HTML** (4-5 MB each):
  - `tsu_modes_research.html` — 2D mode exploration
  - `tsu_metrics_research.html` — Performance dashboard

- **Matplotlib PNG** (static):
  - `tsu_continuous_sampling_demo.png`
  - `convergence_demo.png`
  - `test_output.png`

---

## Troubleshooting

### "Module not found" error
```bash
# Reinstall from requirements
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --force-reinstall
```

### Plotly not exporting to HTML
```bash
# Ensure kaleido is installed
.\.venv\Scripts\python.exe -m pip install --upgrade kaleido
```

### Performance slow on visualization
- Reduce sample sizes in code (n_samples parameter)
- Plotly handles 500-1000 samples smoothly
- Use matplotlib for faster static plots

---

## Hardware Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 100MB for venv + packages
- **CPU**: Multi-core for parallel TSU sampling
- **Network**: Optional (only for cloud deployment)

---

## Python Version

- **Tested**: Python 3.14
- **Required**: Python 3.8+
- **Recommended**: Python 3.10+

---

## License & Attribution

All dependencies are open-source with permissive licenses (BSD, Apache 2.0, MIT).

For details on each library's license:
- NumPy: BSD
- SciPy: BSD
- Matplotlib: PSF License
- Plotly: MIT
- Kaleido: MIT
- Pandas: BSD
- Scikit-learn: BSD
