# Cross-Environment Cooperation Enables Zero-shot Multi-agent Coordination
Repository with environment and training scripts for paper *Cross-Environment Cooperation Enables Zero-shot Multi-agent Coordination* (Jha et. al, 2025)

Environment Code and Scripts based on [JaxMARL](https://github.com/FLAIROx/JaxMARL) repository.

## Installation

```bash
pip install -e .
```

## Usage

All scripts are in the `baselines/CEC/` directory.
Customize the config file in `baselines/CEC/config/ippo_final.yaml` and run the following command to get started.

```bash
python baselines/CEC/ippo_general.py
```

## Environment Code

The Dual Destination and Overcooked procedurally generated environments can be found in the `jaxmarl/environments/` directory under the `ToyCoop` and `overcooked` subdirectories respectively.