# MCDM Schedulers

Advanced schedulers based on Multi-Criteria Decision Making (MCDM).

## Algorithms

- `ARAS` (`aras_scheduler.py`, `aras_strategy.py`)
- `EDAS` (`edas_scheduler.py`, `edas_strategy.py`)
- `MABAC` (`mabac_scheduler.py`, `mabac_strategy.py`)
- `WRR` (`wrr_scheduler.py`, `wrr_strategy.py`)

Each scheduler adapts to the shared scheduling interface through `base_mcdm.py` and uses strategy implementations from `base_strategy.py`.

## Core files

- `base_strategy.py`: common strategy contract and optional dynamic weighting support.
- `base_mcdm.py`: bridges worker/task data to strategy scoring.
- `config_manager.py`: default and persisted config handling.

## Criteria model

MCDM scoring uses weighted criteria such as:
- CPU cores/frequency
- RAM availability
- success rate
- average task duration
- battery/power context (depending on selected config)

`criteria_types` convention:
- `1` means benefit criterion (higher is better)
- `-1` means cost criterion (lower is better)

## Configuration sources

- Default configs are defined in `scheduling/factory.py`.
- Async factory can load scheduler configs from DB (`scheduler_config` table/model path).

## When to use MCDM

Use MCDM schedulers when worker heterogeneity is high and assignment quality matters more than scheduler simplicity.
