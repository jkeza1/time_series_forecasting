## Beijing Air Quality Forecasting
Time series forecasting of PM2.5 concentrations using advanced LSTM neural networks. This project tackles the critical challenge of predicting air pollution by leveraging deep learning to model complex temporal and meteorological patterns.

# Project Status

Current Score: 19,568.4 RMSE

Target Score: < 3,000 RMSE

Status: 🚧 Under Active Optimization

 ## Project Overview
Air pollution, particularly PM2.5, is a significant public health concern in urban environments like Beijing. This project involves building a robust forecasting system using historical air quality and weather data. The model is designed to understand the underlying environmental dynamics, including local emissions, regional pollution transport, and meteorological conditions

# Model Architecture:
The best performing model is a stacked LSTM designed to learn patterns from short-term to long-term dependencies hierarchically.

Input(shape=(72, 45))
→ LSTM(128, return_sequences=True)
→ LSTM(64, return_sequences=True)
→ LSTM(32)
→ Dense(16, activation='relu')
→ Dense(1)

&bull; Adam

&bull; Loss Function: Mean Squared Error (MSE)

&bull; Key Techniques: Early Stopping, Dropout Regularization

# Key Objectives
* Feature Engineering: 45+ engineered features

* Model Architecture: Bidirectional LSTM implementation

* Performance Optimization: Ongoing improvements

* Target Achievement: Working towards sub-3000 RMSE

# Technical Stack

# Core Technologies

**Python 3.8+ · TensorFlow 2.x · Keras · Scikit-learn · Pandas · NumPy**

# Key Features

## Dataset


Dataset	Samples	Time Period	Features
Training	30,676	2010-2013	45
Testing	13,148	2013-2014	45
# Preprocessing:
Missing values were imputed using a 24-hour rolling median to maintain temporal consistency and handle outliers.

# The notebook guides you through:

&bull; Loading and cleaning the data

&bull; Feature engineering and sequencing

&bull; Model definition and training

&bull; Generating predictions for submission

## Feature Engineering
&bull;Temporal Features

&bull; Cyclical time encoding (hour, day, month)

&bull; Seasonal indicators and weekend flags

&bull; Fourier transformations for periodicity

##  Meteorological Features
&bull; Temperature-dew point differentials

&bull; Wind-pressure interactions

&bull; Rolling statistics (6, 12, 24, 48h windows)

&bull; Lag features (1-48 hour intervals)


# Example: Cyclical encoding

``` df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
```
```
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

# Load data
```train = pd.read_csv('data/train.csv')
```
```
test = pd.read_csv('data/test.csv')
```
# Train model
model = create_bidirectional_lstm(sequence_length=36, n_features=45)

history = model.fit(X_train, y_train, validation_split=0.15, epochs=50)

# Performance Tracking
&middot; Metric	Current	Target	Improvement Needed
RMSE	19,568	< 3,000	85%

&middot; Training Stability	NaN issues	Stable	Critical

# Validation Gap	TBD	< 100	TBD
&middot; Current Focus
&middot; Bug Fixes
&middot; NaN values in training sequences
&middot; Gradient instability issues

# Optimization

&bull; Learning rate scheduling

&bull;Gradient clipping

&bull; Advanced regularization

# Feature Analysis

&bull; Feature importance

&bull ;Dimensionality reduction

&bull; Cross-validation

# Impact
This project supports:

&middot; Urban air quality management

&middot; Public health warnings

&middot; Environmental policy

&middot; Sustainable planning

## Contributing
We welcome contributions in:

&bull;Model architecture

&bull;Feature engineering

&bull;Hyperparameter optimization

## Future Work & Environmental Impact
This model provides a strong foundation for tools that can:

&bull; Offer earlier public health warnings during pollution events.

&bull; Evaluate the potential effectiveness of emission control policies through simulation.

&bull; It can be extended into a spatial-temporal model by incorporating data from neighboring cities.

# License
MIT License - see LICENSE file for details
