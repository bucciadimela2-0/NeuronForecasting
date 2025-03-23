# Hybrid Neural-Physical Model for Chaotic Systems

This project implements a hybrid approach combining traditional physical models with deep learning for improved prediction of chaotic neural systems. It features a unique integration of physical models (FitzHugh-Nagumo and LIF) with GRU-based error correction mechanisms.

## Features

- **Hybrid Model Architecture**
  - Physical Models (FitzHugh-Nagumo and LIF)
  - GRU-based error correction
  - Data assimilation techniques

- **Realistic Noise Simulation**
  - Measurement noise
  - Biological noise (pink noise)
  - Synaptic noise
  - Periodic artifacts
  - Systematic errors

- **Advanced Visualization**
  - Phase space plots
  - Chimera state analysis
  - Synchronization maps
  - Time series comparisons

## Installation

```bash
# Clone the repository
git clone https://github.com/bucciadimela2-0/NeuronForecasting

# Install dependencies
pip install numpy torch scikit-learn matplotlib scipy
```

## Usage

### Running the Models

```python
# Run the main simulation
python main.py
```

### Model Components

1. **FitzHugh-Nagumo Model**
   - Classical neuron model implementation
   - Chimera state analysis
   - Coupling strength adjustments

2. **LIF (Leaky Integrate-and-Fire) Model**
   - Basic spiking neuron implementation
   - Network dynamics simulation

3. **Hybrid Model**
   - GRU-based error correction
   - Data assimilation window
   - Autonomous forecasting

## Project Structure

```
├── main.py                 # Main execution script
├── neuronModels/           # Neural model implementations
│   ├── FitzhughNagumoModel.py
│   ├── LifModel.py
│   ├── HybridModel.py
│   └── GRUNetwork.py
├── utils/                  # Utility functions
│   ├── DataHandler.py
│   ├── NoiseGenerator.py
│   └── Plotter.py
└── figures/                # Generated visualizations
```

## Model Architecture

The hybrid model combines physical models with machine learning:

1. **Physical Model Layer**
   - Implements exact physics equations
   - Provides base predictions

2. **Error Correction Layer**
   - GRU network for error prediction
   - Learns systematic deviations

3. **Data Assimilation**
   - Combines predictions and observations
   - Improves forecast accuracy

## Visualization

The project includes comprehensive visualization tools:

- Phase space trajectories
- Chimera state analysis
- Synchronization patterns
- Forecast comparisons

## Results

The hybrid approach demonstrates significant improvements:

- Reduced prediction error compared to pure physical models
- Better handling of noise and systematic errors
- Improved long-term forecasting capability

