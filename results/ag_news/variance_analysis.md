# Variance Analysis

This is a lightweight analysis for the reproducibility study.

## Descriptive Variation

- `optimizer` eta-squared: `0.4581`
- `budget` eta-squared: `0.1337`
- `seed` eta-squared: `0.0000`

## Group Summary

```text
 index      budget optimizer     mean      std  count
     0     default      adam 0.909237 0.002206      5
     1     default     adamw 0.909526 0.002953      5
     2     default   rmsprop 0.910474 0.001997      5
     3     default       sgd 0.250000 0.000000      5
     4 small_tuned      adam 0.905132 0.002176      5
     5 small_tuned     adamw 0.910316 0.002506      5
     6 small_tuned   rmsprop 0.909632 0.002776      5
     7 small_tuned       sgd 0.889342 0.003308      5
```
