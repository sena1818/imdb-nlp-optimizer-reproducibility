# Variance Analysis

This is a lightweight analysis for the reproducibility study.

## Descriptive Variation

- `optimizer` eta-squared: `0.5540`
- `budget` eta-squared: `0.1117`
- `seed` eta-squared: `0.0006`

## Group Summary

```text
 index      budget optimizer     mean      std  count
     0     default      adam 0.876536 0.007682      5
     1     default     adamw 0.876560 0.008142      5
     2     default   rmsprop 0.863664 0.009296      5
     3     default       sgd 0.500000 0.000000      5
     4 small_tuned      adam 0.878368 0.004490      5
     5 small_tuned     adamw 0.873472 0.007125      5
     6 small_tuned   rmsprop 0.866128 0.004892      5
     7 small_tuned       sgd 0.825408 0.008749      5
```
