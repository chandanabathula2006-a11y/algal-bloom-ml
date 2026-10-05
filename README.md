# Algal Bloom Prediction - Machine Learning

Machine learning model for predicting algal bloom risk using environmental water-quality measurements.

## Dataset

The dataset contains 37,732 observations and 20 columns.

Target variable:

- `Bloom_Risk_Label`

Target classes:

- Low
- Medium
- High

## Features Used

The model uses the following 8 environmental features:

- Temperature
- Specific_Conductivity
- DO_Saturation
- Dissolved_Oxygen
- Chlorophyll
- Turbidity
- Phycocyanin
- pH

Four other environmental columns (`Nitrate`, `Wind_Speed`, `Air_Temperature`, and `Barometric_Pressure`) contain a very high percentage of missing values, so they were excluded from the model.

## Machine Learning Model

A Random Forest Classifier is used.

Preprocessing:

- Median imputation
- Train/test split
- Stratified sampling

Model configuration:

- 200 trees
- Balanced class weights
- Random state: 42

## Performance

Test accuracy:

**99.48%**

The model achieved approximately 0.99 F1-score across all three classes.

## Model Files

- `models/bloom_model.pkl` - trained machine learning pipeline
- `models/features.json` - list of input features

## Training

Activate the virtual environment and run:

```bash
python src/train_model.py