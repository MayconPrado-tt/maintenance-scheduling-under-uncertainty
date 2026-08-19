# Maintenance Scheduling under Uncertainty

Computational implementation of a **maintenance scheduling model under uncertainty for small and medium-sized industries (SMEs)**.

This repository contains the source code used in the computational experiments of the associated research paper. The proposed approach combines mixed-integer linear programming (MILP), imperfect preventive maintenance, Monte Carlo simulation, and risk measures to support maintenance scheduling decisions under limited resources and uncertain maintenance effectiveness.

## Overview

The model determines preventive maintenance interventions over a finite planning horizon while considering operational and resource constraints.

The computational framework includes:

- Preventive maintenance scheduling using MILP;
- Budget constraints;
- Labor capacity constraints;
- Minimum and maximum maintenance intervals;
- Imperfect maintenance effectiveness;
- Linear approximation of failure probability;
- Synthetic industrial instance generation;
- Monte Carlo simulation;
- Beta-distributed maintenance effectiveness;
- Correlated uncertainty using a Gaussian copula;
- Value-at-Risk (VaR);
- Conditional Value-at-Risk (CVaR);
- Comparison between linear and exponential failure probability models;
- Export of computational results to Excel.

## Computational Framework

The optimization problem is implemented in Python using the mixed-integer linear programming solver available in `scipy.optimize.milp`.

The main binary decision variable indicates whether preventive maintenance is performed on equipment `i` during period `j`.

The optimization procedure considers:

1. maintenance resource availability;
2. preventive maintenance costs;
3. labor requirements;
4. minimum intervals between interventions;
5. maximum allowable intervals without maintenance;
6. equipment relative age;
7. maintenance effectiveness;
8. failure consequences.

## Uncertainty Analysis

Maintenance effectiveness is treated as an uncertain parameter.

A **Beta distribution** is used because maintenance effectiveness is bounded between 0 and 1. Correlation between maintenance outcomes can also be introduced using a **Gaussian copula**.

The following uncertainty scenarios are evaluated:

| Scenario | Mean effectiveness | Standard deviation | Correlation |
|---|---:|---:|---:|
| Optimistic | 0.95 | 0.03 | 0.10 |
| Baseline | 0.85 | 0.08 | 0.00 |
| High variance | 0.85 | 0.18 | 0.00 |
| Correlated | 0.85 | 0.12 | 0.60 |
| Degraded | 0.65 | 0.10 | 0.30 |
| Severe | 0.50 | 0.15 | 0.60 |

Each scenario is evaluated using **5,000 Monte Carlo replications**.

## Risk Measures

The simulation calculates the following statistics:

- Mean cost;
- Standard deviation;
- 5th percentile;
- Median;
- VaR at 95%;
- CVaR at 95%;
- VaR at 99%;
- CVaR at 99%.

These measures allow the maintenance schedules to be evaluated not only according to expected cost but also according to exposure to unfavorable outcomes.

## Synthetic Instances

The implementation also generates synthetic industrial instances to evaluate the behavior of the model for different problem sizes and operational contexts.

The experiments include:

- Bottling line;
- Pumping station;
- Cold-storage facility;
- Multi-product factory.

These instances vary in the number of equipment items, planning horizon, failure costs, preventive maintenance costs, and labor requirements.

## Repository Structure

```text
maintenance-scheduling-under-uncertainty/
│
├── maintenance_model.py
├── requirements.txt
├── README.md
│
├── results/
│   └── maintenance_model_results.xlsx
│
└── figures/
    └── computational experiment figures
```

## Requirements

The implementation requires Python 3 and the following packages:

```text
numpy
pandas
scipy
tqdm
openpyxl
xlsxwriter
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Model

Clone the repository:

```bash
git clone https://github.com/MayconPrado-tt/maintenance-scheduling-under-uncertainty.git
```

Enter the repository directory:

```bash
cd maintenance-scheduling-under-uncertainty
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the computational experiment:

```bash
python maintenance_model.py
```

## Output

After execution, the script creates a results directory containing:

```text
model_results/
└── maintenance_model_results.xlsx
```

The Excel workbook contains:

- Model parameters;
- Optimized maintenance schedule;
- Detailed equipment-period results;
- Monte Carlo simulation results;
- VaR and CVaR measures;
- Synthetic instance results;
- Maintenance schedules for the synthetic instances.

## Reproducibility

Random seeds are explicitly defined in the implementation to improve computational reproducibility.

The baseline experiments use the same parameters reported in the associated research study. Monte Carlo scenarios and synthetic instances also use predefined random seeds.

## Research Context

The computational framework was developed to support preventive maintenance planning in industrial environments characterized by:

- limited maintenance budgets;
- restricted labor availability;
- imperfect maintenance actions;
- uncertainty in intervention effectiveness;
- significant consequences associated with equipment failure.

The model is particularly intended as a transparent and computationally accessible decision-support framework for small and medium-sized industries.

## Authors

- Natalia Sanchez Sandoval — Universidade Federal de São Paulo (UNIFESP)
- Karen Viviana Pinilla Moreras — Universidade Federal de São Paulo (UNIFESP)
- Maycon Cruz do Prado — ITA / UNIFESP
- Lucas A. Alves Gazale — Universidade Federal de São Paulo (UNIFESP)
- Emily Brito de Oliveira — Suzano
- Leonardo Mito — Suzano
- Filipe Santos — Suzano
- Luiz Leduino Salles-Neto — Universidade Federal de São Paulo (UNIFESP)

## Citation

If you use this code in academic work, please cite the associated research paper.

Full bibliographic information will be added after publication.

## License

This repository is intended for academic and research purposes.
