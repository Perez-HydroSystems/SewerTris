# Sewertris: Tetris-based Sewer Network Generator

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Sewertris** is a novel synthetic benchmarking dataset generator and Python package designed to support reproducible model testing and innovation in sewer system analysis. Inspired by the iconic game Tetris, it assembles modular, spatially coherent "block units" to build synthetic sewer networks that mimic the layout and heterogeneity of real urban systems.

## 🌟 Overview

Sanitary sewer systems are critical infrastructure for public health, environmental protection, and wastewater treatment. However, aging infrastructure, rapid population growth, changing land use, and evolving climate patterns place increasing stress on these systems. The development and validation of modeling strategies for infiltration and inflow (I&I) estimation, network design optimization, and monitoring strategies are often hindered by limited data availability and site-specific, proprietary datasets.

**Sewertris addresses this gap** by providing a comprehensive synthetic benchmarking platform that enables robust testing of flow prediction models, anomaly detection algorithms, and design tools across a wide range of system configurations.

## 🎯 Key Capabilities

### 🔧 Modular Network Generation
- **Tetris-inspired Block System**: Assembles modular "block units" representing different urban infrastructure types (residential, commercial, industrial, parks)
- **Scalable Networks**: Generate synthetic sewer networks ranging from 100 to 10,000+ pipes
- **Realistic Topology**: Creates spatially coherent networks that reflect real urban sewer system layouts
- **Configurable Complexity**: Adjustable network density, connectivity patterns, and hierarchical structures

### 🏙️ Urban Distribution Modeling
- **Heterogeneous Land Use**: Simulate diverse urban environments with varying development patterns
- **Demographic Variability**: Model different population densities and usage patterns
- **Infrastructure Age**: Incorporate varying pipe ages and conditions across the network
- **Geographic Constraints**: Account for topographical and geological factors in network design

### 🌊 Hydrologic Flow Components
The package generates realistic hydrologic inputs including:

#### **Base Wastewater Flow (BWF)**
- Diurnal and seasonal patterns
- Population-based flow generation
- Commercial and industrial contributions
- Weekend and holiday variations

#### **Groundwater Infiltration (GWI)**
- Steady-state and seasonal infiltration patterns
- Pipe condition-dependent infiltration rates
- Soil type and groundwater level influences
- Age-related deterioration effects

#### **Rainfall-Derived Infiltration and Inflow (RDII)**
- Event-based inflow patterns
- Various rainfall intensity scenarios
- Soil moisture and antecedent conditions
- Defect-specific inflow characteristics

### 🔄 EPA SWMM Integration
- **Seamless Coupling**: Direct integration with EPA Storm Water Management Model (SWMM)
- **Long-term Simulation**: Support for multi-year simulations (20+ years of synthetic forcing)
- **Pollutant Tracing**: Advanced flow component disaggregation using pollutant tracers
- **Performance Metrics**: Automated extraction of hydraulic performance indicators

## 📊 Applications & Use Cases

### 🔬 Research & Development
- **Model Validation**: Test and validate new I&I estimation algorithms
- **Algorithm Benchmarking**: Compare performance of different modeling approaches
- **Sensitivity Analysis**: Evaluate model robustness across diverse system configurations
- **Method Development**: Develop new approaches for sewer system analysis

### 🎓 Educational Applications
- **Urban Hydrology Courses**: Hands-on learning with realistic sewer networks
- **Infrastructure Design**: Teach principles of sewer network design and optimization
- **Data Analysis Training**: Practice with large-scale infrastructure datasets
- **Research Training**: Provide students with accessible, comprehensive datasets

### 🏗️ Engineering Practice
- **Design Optimization**: Test design alternatives before implementation
- **Monitoring Strategy**: Optimize sensor placement and monitoring frequencies
- **System Assessment**: Evaluate existing system performance and upgrade needs
- **Risk Analysis**: Assess system vulnerability under various scenarios

## 🚀 Getting Started

### Installation

```bash
pip install sewertris
```

### Quick Example

```python
import sewertris as st

# Generate a medium-scale sewer network
# Example here

# Configure hydrologic inputs
# Example here

# Export to EPA SWMM
# Example here

# Run simulation and analyze results
# Example here
```

## 📈 Dataset Characteristics

- **Network Sizes**: 100 to 10,000+ pipes per network
- **Pipe Attributes**: Diameter, material, age, slope, depth
- **Flow Components**: BWF, GWI, RDII with realistic temporal patterns
- **Simulation Period**: Up to 20+ years of synthetic forcing data
- **Spatial Resolution**: Block-level to individual pipe resolution
- **Temporal Resolution**: Sub-hourly to annual patterns

## 🤝 Contributing

We welcome contributions to the Sewertris project! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit bug reports, feature requests, and code contributions.

## 📚 Citation

If you use Sewertris in your research, please cite our work:

```bibtex
@conference{blanco-silva2024tetris,
  title={Tetris-based Sewer Network Generator: A Virtual Benchmarking Dataset for Evaluating I\&I, Design, and Monitoring Models},
  author={Blanco-Silva, Kevin and Perez, Gabriel},
  institution={School of Civil and Environmental Engineering, Oklahoma State University},
  address={Stillwater, Oklahoma, USA},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Kevin Blanco-Silva** - School of Civil and Environmental Engineering, Oklahoma State University
- **Gabriel Perez** - School of Civil and Environmental Engineering, Oklahoma State University

---

*Sewertris: Building better sewer networks, one block at a time* 🧱🌊

