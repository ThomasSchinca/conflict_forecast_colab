# Conflict Forecast Input Flee Model

This repository contains the replication data for the forecasting part of the paper "An agent-based model with machine learning-based conflict forecasting approach for predicting conflict-driven population displacement".

## Overview
The repository includes the necessary scripts and data to replicate the results presented in the paper for the forecasting part. The primary script, `Colab.py`, executes both models, generates output files, and store prediction results.

## Requirements
- WARNING: The PRIO-GRID Yearly Variables for 2007-2014 and PRIO-GRID Static Variables Dataset (available here: https://grid.prio.org/#/download) need to be added in the Input folder.
- **Python version:** 3.8.5
- Required libraries: Install dependencies using:
  ```bash
  pip install -r requirements.txt

## Running the Model
To reproduce the results, execute the following command in your terminal:
```bash
python Colab.py
```
## Expected Runtime
The script should take approximately 15 minutes to complete.

## Cases/countries
- Mali 2012
- Central African Republic 2013
- Burundi 2015
- South Sudan 2013
