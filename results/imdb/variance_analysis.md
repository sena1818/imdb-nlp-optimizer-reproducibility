# Variance Analysis

This is a lightweight analysis for the reproducibility study.

## Descriptive Variation

- `optimizer` eta-squared: `0.7283`
- `budget` eta-squared: `0.0531`
- `seed` eta-squared: `0.0038`

## Group Summary

```text
 index      budget optimizer     mean      std  count
     0     default      adam 0.876536 0.007682      5
     1     default     adamw 0.876560 0.008142      5
     2     default   rmsprop 0.861424 0.012878      5
     3     default       sgd 0.500000 0.000000      5
     4 small_tuned      adam 0.876152 0.005074      5
     5 small_tuned     adamw 0.867728 0.003903      5
     6 small_tuned   rmsprop 0.863488 0.009740      5
     7 small_tuned       sgd 0.741016 0.083409      5
```
