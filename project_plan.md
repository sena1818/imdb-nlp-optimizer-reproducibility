# Reproducibility Study Project Plan

Course: Reproducible Machine Learning  
Paper basis: Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers  
Working title: Optimizer-Level Nondeterminism on an NLP Text Classification Task

## 1. Core Idea

The original paper studies how optimizer comparisons can be unstable under different tasks, tuning budgets, learning-rate schedules, and random seeds. This project transfers the same idea to a clearer NLP setting.

The central question is:

> Do optimizer rankings remain stable on an NLP text classification task, or do they change with tuning budget and random seed?

The project should not try to reproduce the full scale of the original benchmark. Instead, it should reproduce one important mechanism from the paper:

- default optimizer settings versus tuned settings
- repeated runs across random seeds
- ranking stability across optimizers
- optionally, a small variance component analysis

## 2. Recommended Experimental Setup

### Dataset

Primary choice: AG News

Reason:

- simple 4-class topic classification task
- standard NLP benchmark
- easy to explain in the final presentation
- large enough to be meaningful, but still manageable
- accuracy is an intuitive metric

Backup choice: SST-2

Reason:

- binary sentiment classification
- also easy to explain
- often used in NLP

AG News is currently the better choice because it is slightly more substantial and does not require dealing with the full GLUE setup.

### Model Architecture

Primary choice: TextCNN

Pipeline:

```text
text -> tokens -> embeddings -> 1D convolutions -> max pooling -> classifier -> label
```

Reason:

- lightweight neural model
- fast enough for repeated optimizer experiments
- still clearly a deep learning setup
- easier than transformer fine-tuning
- suitable for optimizer comparison

Backup choice: BiLSTM

Reason:

- more sequence-oriented than TextCNN
- still manageable

TextCNN is currently the safer first choice. BiLSTM can be used as an extension if time allows.

### Optimizers

Recommended set:

- SGD
- Adam
- RMSProp
- AdamW

Reason:

- small enough to finish
- representative mix of classical and adaptive optimizers
- directly connected to the paper's optimizer-comparison theme
- Adam/AdamW provide strong modern baselines

Optional extension:

- add NAdam or Adagrad if the first experiment is cheap enough

## 3. Experimental Conditions

### Condition A: One-shot / default setting

For each optimizer, use a reasonable default configuration.

Example:

- SGD: learning rate 0.1 or 0.01, momentum optional
- Adam: learning rate 1e-3
- AdamW: learning rate 1e-3, weight decay default/small
- RMSProp: learning rate 1e-3 or framework default

Goal:

Measure out-of-the-box performance.

This matches the paper's question:

> How well do optimizers work without tuning?

### Condition B: Small tuning budget

For each optimizer, sample a small number of hyperparameter configurations by random search.

Recommended budget:

- minimum: 5 random configurations per optimizer
- better: 10 random configurations per optimizer

Suggested hyperparameters:

- learning rate
- weight decay
- optimizer-specific parameters only if manageable

Suggested distributions:

- learning rate: log-uniform, for example 1e-4 to 1e-1
- weight decay: log-uniform or include zero plus log-uniform, for example 0 and 1e-6 to 1e-2
- momentum for SGD: uniform or discrete choices, for example 0, 0.5, 0.9

Goal:

Test whether tuning changes optimizer ranking.

### Condition C: Seed reruns

After selecting the best configuration for each optimizer and condition, rerun it with multiple random seeds.

Recommended:

- minimum: 3 seeds
- better: 5 seeds
- ideal if GPU time allows: 10 seeds

For this course project, 5 seeds is a good balance.

Goal:

Estimate stability with mean and standard deviation.

Important point:

If tuning is done with a single seed and only the selected configuration is rerun across multiple seeds, lucky tuning can still happen. This is not a mistake; it is one of the reproducibility issues being studied.

## 4. Minimal Viable Study

This is the smallest version that still makes a good reproducibility study.

Use:

- AG News
- TextCNN
- 4 optimizers: SGD, Adam, RMSProp, AdamW
- default setting
- small random-search tuning with 5 configurations per optimizer
- final rerun with 5 seeds

Approximate run count:

```text
default phase:
4 optimizers x 5 seeds = 20 runs

tuning phase:
4 optimizers x 5 configs x 1 tuning seed = 20 runs

rerun tuned best:
4 optimizers x 5 seeds = 20 runs

total = about 60 training runs
```

This should be enough to answer:

- Does tuning change the ranking?
- Does Adam/AdamW remain strong?
- How large is seed variance?
- Do some optimizers look better under default settings than after controlled reruns?

## 5. Recommended Study

This is the version to aim for if time and GPU access are okay.

Use:

- AG News
- TextCNN
- 4 optimizers
- default setting
- small tuning budget with 10 configurations per optimizer
- final rerun with 5 seeds

Approximate run count:

```text
default phase:
4 optimizers x 5 seeds = 20 runs

tuning phase:
4 optimizers x 10 configs x 1 tuning seed = 40 runs

rerun tuned best:
4 optimizers x 5 seeds = 20 runs

total = about 80 training runs
```

This is the recommended balance between scientific value and feasibility.

## 6. Optional Extension

If the basic experiment runs quickly, add one of the following:

### Option 1: Add a second model

Add BiLSTM on AG News.

This would let the study ask whether optimizer ranking depends on model architecture.

### Option 2: Add a second dataset

Add SST-2.

This would let the study ask whether optimizer ranking depends on task/data.

### Option 3: Add a simple learning-rate schedule

Compare constant learning rate versus cosine schedule.

This directly connects to the paper's schedule x optimizer interaction.

### Option 4: Add LMEM analysis

Fit a simple linear mixed effects model on the collected accuracies.

Example formula:

```text
accuracy ~ optimizer + budget + optimizer:budget + (1 | seed)
```

If there are multiple datasets or models:

```text
accuracy ~ optimizer + budget + optimizer:budget + (1 | seed) + (1 | task)
```

This would help estimate how much variance is attributable to optimizer choice, tuning budget, seed, and residual noise.

## 7. Metrics and Outputs

Primary metric:

- test accuracy

Secondary metrics:

- validation accuracy during tuning
- mean and standard deviation across seeds
- ranking of optimizers under each condition
- difference between default ranking and tuned ranking

Useful plots:

- bar plot: mean accuracy by optimizer and condition
- error bars: standard deviation across seeds
- ranking table: default versus tuned
- line plot: optimizer rank changes from default to tuned
- optional heatmap: pairwise performance differences, similar in spirit to Figure 2 of the paper

## 8. Expected Time Cost

### Human work time

Estimated human work:

- project setup and data loading: 2-4 hours
- implement TextCNN and training loop: 4-6 hours
- implement optimizer configs and random search: 3-5 hours
- run pilot experiments: 2-4 hours
- run full experiments: mostly waiting, plus monitoring
- analyze results and make plots: 4-6 hours
- prepare final presentation: 1-2 days

Total active work:

```text
about 2-4 focused working days
```

More realistic calendar time:

```text
about 1 week, because experiments need iteration and checking
```

### GPU time estimate

This depends strongly on hardware, batch size, sequence length, and epochs.

For AG News + TextCNN:

- rough per-run time on a reasonable GPU: 2-10 minutes
- rough per-run time on CPU: 10-40 minutes

Minimal study, about 60 runs:

```text
GPU estimate: 2-10 hours
CPU estimate: 10-40 hours
```

Recommended study, about 80 runs:

```text
GPU estimate: 3-14 hours
CPU estimate: 15-55 hours
```

Safer planning number:

```text
Reserve about 10-15 GPU hours for the main experiment.
Reserve another 3-5 GPU hours for debugging and pilot runs.
Total recommended GPU budget: about 15-20 GPU hours.
```

The project should first run a very small pilot to measure actual time per run before launching all experiments.

## 9. Practical Execution Plan

### Step 1: Pilot

Run one small TextCNN training on AG News with Adam.

Goal:

- verify data loading
- verify tokenization
- verify model trains
- measure runtime per epoch
- choose number of epochs

### Step 2: Default baseline

Run all four optimizers with default settings for 3 seeds.

Goal:

- get a first ranking
- check whether any optimizer fails or diverges
- decide whether defaults are reasonable

### Step 3: Random search tuning

Run 5-10 sampled configurations per optimizer on one tuning seed.

Goal:

- select best validation configuration for each optimizer
- record all sampled hyperparameters and validation scores

### Step 4: Multi-seed rerun

Rerun the selected best configuration for each optimizer with 5 seeds.

Goal:

- compute mean and standard deviation
- test whether selected configurations are stable

### Step 5: Analysis

Compare:

- default versus tuned performance
- optimizer rankings before and after tuning
- seed variance per optimizer
- whether Adam/AdamW remain strong baselines

Optional:

- fit a simple LMEM if the result table is rich enough

### Step 6: Final presentation

Tell the story as:

1. Original paper: optimizer ranking is conditional and noisy.
2. This project: test the same idea on NLP text classification.
3. Method: AG News + TextCNN + optimizers + tuning/seeds.
4. Results: rankings, variance, and tuning effects.
5. Interpretation: what this says about reproducibility.
6. Limitations: small task, small model, limited tuning budget.

## 10. Main Risks

### Risk 1: Too many runs

Mitigation:

- start with 3 seeds and 5 tuning configs
- increase only if runs are fast

### Risk 2: SGD performs badly

Mitigation:

- include momentum
- tune learning rate carefully
- treat bad default performance as an interesting finding

### Risk 3: TextCNN is too stable and optimizer differences are small

Mitigation:

- report that differences are small relative to seed variance
- this is still a reproducibility result

### Risk 4: Tuning distributions introduce bias

Mitigation:

- explicitly discuss this as part of the study
- report search ranges clearly
- avoid claiming universal optimizer superiority

## 11. Final Scope Recommendation

The best first version is:

```text
AG News + TextCNN
4 optimizers: SGD, Adam, RMSProp, AdamW
one-shot default condition
small random search condition
5 seeds for final reruns
mean/std + ranking analysis
optional LMEM if time allows
```

This scope is realistic, close to the paper, and clearly connected to reproducible machine learning.
