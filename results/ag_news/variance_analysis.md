# Variance Analysis

This is a lightweight analysis for the reproducibility study.

## Descriptive Variation

- `optimizer` eta-squared: `0.4539`
- `budget` eta-squared: `0.1345`
- `seed` eta-squared: `0.0001`

## Group Summary

```text
 index      budget optimizer     mean      std  count
     0     default      adam 0.907658 0.002561      5
     1     default     adamw 0.909184 0.002985      5
     2     default   rmsprop 0.904500 0.009394      5
     3     default       sgd 0.250000 0.000000      5
     4 small_tuned      adam 0.904368 0.004765      5
     5 small_tuned     adamw 0.903289 0.004721      5
     6 small_tuned   rmsprop 0.909132 0.003026      5
     7 small_tuned       sgd 0.889342 0.003308      5
```
