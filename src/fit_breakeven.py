"""Reviewer point 4: give the break-even curve a closed form.

The manuscript reports the break-even detection quality at three minority
fractions. A practitioner whose dataset sits between them needs to interpolate,
so we fit a two-parameter curve through the three crossings and report it.

The threshold is a probability, so it is fitted on the logit scale against the
log of the minority fraction:

    logit(t*) = a + b * ln(m)

which keeps the prediction inside (0, 1) for every m and is linear in the two
quantities the rule is stated in. We weight each crossing by the inverse width
of its bootstrap interval, since the 2% crossing is the least precisely located.
"""
import numpy as np, pandas as pd

T = pd.read_csv('/home/claude/fmclean/results/decouple_thresholds.csv')
m = T.minority.values
t = T.crossing.values
w = 1.0 / (T.hi.values - T.lo.values)          # precision of each crossing

logit = lambda p: np.log(p / (1 - p))
sigmoid = lambda z: 1 / (1 + np.exp(-z))

X = np.column_stack([np.ones_like(m), np.log(m)])
y = logit(t)
W = np.diag(w)
beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
a, b = beta

pred = sigmoid(X @ beta)
resid = t - pred
ss_res = float((resid ** 2).sum())
ss_tot = float(((t - t.mean()) ** 2).sum())

print('fitted curve:  logit(t*) = %.4f %+.4f * ln(m)' % (a, b))
print('equivalently:  t* = 1 / (1 + exp(-(%.3f %+.3f ln m)))' % (a, b))
print()
print(' minority   observed   fitted    error')
for mi, ti, pi in zip(m, t, pred):
    print('   %5.2f      %.3f     %.3f   %+.4f' % (mi, ti, pi, pi - ti))
print()
print('max |error| = %.4f over the fitted range; R^2 = %.4f'
      % (np.abs(resid).max(), 1 - ss_res / ss_tot))
print()
print('interpolated thresholds a reader might want:')
for mi in (0.01, 0.05, 0.15, 0.20, 0.25, 0.50):
    z = a + b * np.log(mi)
    flag = '  (extrapolated)' if not (0.02 <= mi <= 0.35) else ''
    print('   m = %4.0f%%  ->  t* = %.3f%s' % (100 * mi, sigmoid(z), flag))
