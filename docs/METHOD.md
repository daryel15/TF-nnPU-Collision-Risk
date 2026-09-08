# Implemented method

The 5-dimensional TF input, Transformer architecture, positional buffer, attention pooling and classification head are preserved from the supplied notebooks. The final checkpoint is unchanged. Full trajectories are used for unsupervised reconstruction in the baseline reruns; supervised evaluation preserves the author's random prefix split. It measures agreement with endpoint PU labels; the validation trajectories also contribute training prefixes.

## Historical negative-risk correction

Let R+ = pi * E_P[softplus(-logit)] and R- = E_U[softplus(logit)] - pi * E_P[softplus(logit)]. The supplied code returns:

```
R+ - gamma * R-    if R- < -beta
R+ + R-            otherwise
```

At the released defaults gamma=1 and beta=0 this equals R+ + abs(R-). This is the historical implementation retained by `tf_nnpu.losses.nnpu_loss` and the comparison runners. It differs from directly minimizing the clipped nnPU estimator R+ + max(0,R-). It also differs from the original authors' correction update, which backpropagates through -gamma*R- alone in that branch. See the [original authors' implementation](https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py).

Manuscript wording matching this release: “The implementation uses a negative-risk correction in which the training loss is R+ + R- when R- is non-negative and R+ - R- otherwise (gamma=1, beta=0). This retains the positive-risk term during the corrective update.” Do not describe this as identical to Equation 16's clipping rule. This release does not change the optimizer objective of previously reported experiments or claim that the variant has the original estimator's guarantees.

## Rerun settings

- TF: input=5, width=64, heads=4, layers=2, FFN=128, dropout=0.1, positional length=500; 71,746 trainable parameters.
- Stage 1 reruns: AdamW, learning rate 0.001, weight decay 0.0001, batch 16, 20 epochs, reconstruction MSE. These are the supplied executable research defaults, which differ from the 30-epoch/batch-32 statement in one manuscript version.
- Stage 2: AdamW, learning rate 0.0001, weight decay 0.0001, batch 32, at most 200 epochs; validation F1 early stopping, patience=8, minimum improvement=0.001; threshold=0.5, prior=0.1.
- Sensitivity: depth 1/2/3, heads 1/2/4/8, context 5/10/15, learning rate 1e-5/5e-5/1e-4/5e-4/1e-3. The release's isolated reruns pretrain afresh for each configuration and use saved main split labels. They do not inherit hidden state or label corruption from earlier interactive cells. Their seed is explicitly 42.
- Cost-sensitive: WBCE positive weight=3; focal alpha=0.25, gamma=2. Masks use original seed 123. New model seeds are explicit and differ from the undocumented historical interactive RNG sequence; outputs are labeled new seeded reruns.
- Su-LSTM, GAT, CMPA: model definitions and optimizer settings are extracted from the supplied notebooks, with saved split/mask loading and output recording added. GAT/CMPA are adapted baselines on processed ego-relative features, not claims of reproducing their originating papers' full datasets or sensing systems. Their source attribution and adaptation notes are retained in `archive/`.

Full reruns write their own checkpoints and prediction keys. Small GPU/platform differences and unavailable original random states mean exact historical baseline scores are not guaranteed.
