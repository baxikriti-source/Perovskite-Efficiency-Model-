# Perovskite-Efficiency-Model-
Machine Learning-Assisted $\text{CsPbBr}_3$ Perovskite Photodetector Performance PredictionA data-driven repository implementing a machine learning framework to predict, analyze, and optimize the optoelectronic performance of all-inorganic cesium lead bromide ($\text{CsPbBr}_3$) perovskite photodetectors.

# Overview

Exploring multi-dimensional parameter spaces in perovskite synthesis and device fabrication via traditional trial-and-error is resource-intensive. 

This project demonstrates how regression algorithms (Random Forest) can map complex, non-linear relationships between fabrication/operating parameters (such as stoichiometry, annealing profiles, grain size, bias voltage, and incident wavelength) and core detector outputs like Responsivity ($R$) and Specific Detectivity ($D^*$).

# Key Features

Data Processing & Feature Engineering: Automatically handles dataset scaling, splitting, and physics-informed feature generation.

Model Training: Implements robust ensemble learning via scikit-learn's Random Forest Regressor.

Performance Evaluation: Computes Root Mean Square Error (RMSE) and Coefficient of Determination ($R^2$) accuracy metrics.

Explainable Insights: Generates feature importance visualizations to highlight which fabrication factors (e.g., wavelength and bias voltage) drive photodetector efficiency.

# Tech Stack

Language: Python 3.x

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
