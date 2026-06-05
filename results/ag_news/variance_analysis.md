# Variance Analysis

This is a lightweight analysis for the reproducibility study.

## Descriptive Variation

- `optimizer` eta-squared: `0.4636`
- `budget` eta-squared: `0.1287`
- `seed` eta-squared: `0.0001`

## Group Summary

```text
 index      budget optimizer     mean      std  count
     0     default      adam 0.907658 0.002561      5
     1     default     adamw 0.909184 0.002985      5
     2     default   rmsprop 0.904500 0.009394      5
     3     default       sgd 0.250000 0.000000      5
     4 small_tuned      adam 0.910658 0.002091      5
     5 small_tuned     adamw 0.905500 0.003408      5
     6 small_tuned   rmsprop 0.892711 0.012306      5
     7 small_tuned       sgd 0.881711 0.002315      5
```
