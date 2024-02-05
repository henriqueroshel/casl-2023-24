import numpy as np
from scipy.stats import t

CONFIDENCE = 0.98

data = np.array([12.0, 1.0, 4.0, 5.0, 0.0, 6.0])
n = len(data)
m = data.mean()
s = data.std(ddof=1)
s2 = sum([ (x-m)**2 for x in data ]) / (n-1)
s1 = s2**.5

print(f"Average: {m}")
print(f"Standard deviation: {s}, ({s1})")
print(f"Degrees of freedom: {n-1}")

# Compute lower and upper bound of confidence interval
ci = t.interval(
    CONFIDENCE,
    n-1,
    m,
    np.sqrt(s)
)

# Compute the accuracy
def acc(ci, m):
    ci_lower, ci_upper = ci
    eps = (ci_upper-ci_lower) / (2*m) # relative error
    acc = 1 - eps
    return acc

accuracy = acc(ci,m)
print(f"Relative error: \u0395 = {1-accuracy:.2f}")
print(f"Accuracy: 1-\u0395 = {accuracy:.2f}")