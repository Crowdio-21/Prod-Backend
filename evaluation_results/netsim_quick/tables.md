# Network-awareness sweep: tables (generated)

Median paired makespan gain over FIFO per seed, with a 95% bootstrap CI of the median. Positive = shorter than FIFO. `*` marks a CI that excludes zero. Rows with fewer than 30 seeds say so.


## transfer/compute ratio

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 4 | 9.5 | +0.9% [-1.7, +3.2] | +0.0% [+0.0, +0.0] | +6.0% [+1.1, +8.9]* | +6.0% [+1.1, +8.9]* | +4.7% [+0.1, +7.8]* | +4.7% [+0.1, +7.7]* | +6.7% [+1.2, +9.2]* |
| 0.1 | 4 | 10.3 | +1.2% [-0.8, +4.7] | +0.0% [+0.0, +0.0] | +5.4% [+3.1, +9.2]* | +5.4% [+3.1, +9.2]* | +2.1% [+0.8, +3.3]* | +2.5% [+0.8, +4.6]* | +5.1% [+2.1, +11.3]* |
| 0.25 | 4 | 12.4 | -1.0% [-10.7, +0.2] | +0.0% [+0.0, +0.0] | +4.7% [+2.3, +6.8]* | +4.7% [+2.3, +6.8]* | +2.1% [+0.7, +5.4]* | +2.2% [+0.7, +5.4]* | +4.9% [+3.1, +7.3]* |
| 0.5 | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |
| 1 | 4 | 23.7 | -2.6% [-7.9, +6.8] | +0.0% [+0.0, +0.0] | +5.8% [+1.6, +11.9]* | +5.8% [+1.6, +11.9]* | +2.6% [-1.4, +7.6] | +2.6% [-3.3, +7.6] | +7.2% [+4.9, +13.8]* |
| 2 | 4 | 39.9 | -5.1% [-22.7, +15.8] | +0.0% [+0.0, +0.0] | +6.6% [+0.6, +21.9]* | +6.6% [+0.6, +21.9]* | +8.5% [-2.2, +16.7] | +10.6% [-7.4, +16.3] | +13.7% [+4.5, +20.5]* |

## heterogeneity

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moderate | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |

## mobility

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walking 1.4 m/s | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |

## bandwidth report freshness

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| every 10 s | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |

## contention from other devices

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |

## backhaul cap (median)

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 Mbit/s | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |

## network events

| condition | seeds | FIFO makespan (s) | round_robin | least_loaded | bandwidth_aware | adaptive | rssi_aware | rssi_aware_cal | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 4 | 16.5 | +0.2% [-4.4, +1.4] | +0.0% [+0.0, +0.0] | +4.2% [+0.5, +12.7]* | +4.2% [+0.5, +12.7]* | +1.1% [-3.4, +10.3] | +1.1% [-4.7, +10.3] | +6.4% [+1.3, +11.9]* |

## RSSI vs bandwidth-only (paired, per condition)

Median paired gain of the RSSI-aware scheduler over bandwidth-aware.

| axis | condition | default curve | calibrated curve | bandwidth est. error | RSSI-cal est. error |
| --- | --- | --- | --- | --- | --- |
| transfer/compute ratio | 0.05 | -1.1% [-2.2, -0.5]* | -1.1% [-2.2, -0.5]* | 0.35 | 0.35 |
| transfer/compute ratio | 0.1 | -3.9% [-6.5, -1.6]* | -2.7% [-6.5, -1.6]* | 0.37 | 0.36 |
| transfer/compute ratio | 0.25 | -1.5% [-5.6, +0.0] | -1.5% [-5.3, +0.0] | 0.38 | 0.40 |
| transfer/compute ratio | 0.5 | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
| transfer/compute ratio | 1 | -2.9% [-9.1, -0.3]* | -3.8% [-9.1, -0.3]* | 0.42 | 0.42 |
| transfer/compute ratio | 2 | -0.9% [-6.7, +3.1] | -3.1% [-8.0, +7.3] | 0.44 | 0.44 |
| heterogeneity | moderate | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
| mobility | walking 1.4 m/s | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
| bandwidth report freshness | every 10 s | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
| contention from other devices | none | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
| backhaul cap (median) | 20 Mbit/s | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
| network events | none | -3.0% [-4.4, -2.7]* | -3.6% [-5.3, -1.9]* | 0.41 | 0.42 |
