## Beijing Air Quality Forecasting
Time series forecasting of PM2.5 concentrations using advanced LSTM neural networks.

## Project Status
Current Score: 19,568.4 RMSE
Target Score: < 3,000 RMSE
Status: 🚧 Under Active Optimization

# Key Objectives
* Feature Engineering: 45+ engineered features

* Model Architecture: Bidirectional LSTM implementation

* Performance Optimization: Ongoing improvements

* Target Achievement: Working towards sub-3000 RMSE

# Technical Stack

# Core Technologies

" Python 3.8+ · TensorFlow 2.x · Keras · Scikit-learn · Pandas · NumPy "

# Key Features
Bidirectional LSTMs · Advanced Feature Engineering · Time Series Analysis
" Python 3.8+ · TensorFlow 2.x · Keras · Scikit-learn · Pandas · NumPy " 

# Key Features
Bidirectional LSTMs · Advanced Feature Engineering · Time Series Analysis
🏗️ Model Architecture
Bidirectional LSTM with multiple layers, batch normalization, and dropout regularization

🔧 Feature Engineering
🕐 Temporal Features
⏰ Cyclical time encoding (hour, day, month)

📅 Seasonal indicators and weekend flags

🔁 Fourier transformations for periodicity

🌡️ Meteorological Features
🌡️ Temperature-dew point differentials

💨 Wind-pressure interactions

📊 Rolling statistics (6, 12, 24, 48h windows)

📈 Lag features (1-48 hour intervals)

python
# Example: Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
📊 Dataset
Dataset	Samples	Time Period	Features
Training	30,676	2010-2013	45
Testing	13,148	2013-2014	45
🚀 Quick Start
Installation
bash
pip install tensorflow scikit-learn pandas numpy matplotlib
Basic Usage
python
import pandas as pd
from models import create_bidirectional_lstm

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Train model
model = create_bidirectional_lstm(sequence_length=36, n_features=45)
history = model.fit(X_train, y_train, validation_split=0.15, epochs=50)
📈 Performance Tracking
Metric	Current	Target	Improvement Needed
RMSE	19,568	< 3,000	85%
Training Stability	NaN issues	Stable	Critical
Validation Gap	TBD	< 100	TBD
🔧 Current Focus
🐛 Bug Fixes

NaN values in training sequences

Gradient instability issues

⚡ Optimization

Learning rate scheduling

Gradient clipping

Advanced regularization

📊 Feature Analysis

Feature importance

Dimensionality reduction

Cross-validation

🌍 Impact
This project supports:

🏙️ Urban air quality management

🏥 Public health warnings

📋 Environmental policy

🌱 Sustainable planning

🤝 Contributing
We welcome contributions in:

Model architecture

Feature engineering

Hyperparameter optimization

## Future Work & Environmental Impact
This model provides a strong foundation for tools that can:

&bull; Offer earlier public health warnings during pollution events.

Evaluate the potential effectiveness of emission control policies through simulation.

Be extended into a spatial-temporal model by incorporating data from neighboring cities.

📝 License
MIT License - see LICENSE file for details
