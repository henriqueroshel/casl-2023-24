import numpy as np
from tqdm import tqdm

GRADES_DIST = { 18:87,  19:62,  20:74,  21:55,  22:99, 23:94,  24:117, 
                25:117, 26:136, 27:160, 28:215, 29:160, 30:473 }
MIN_GRADE, MAX_GRADE = 18,30
# dict with cumulative probability for each possible grade
GRADES_CUMULAT_PROB = dict()
GRADES_PROB = dict()
n_samples = sum([GRADES_DIST[grd] for grd in GRADES_DIST])
cumulat_prob = 0
for grd in range(MIN_GRADE,MAX_GRADE+1):
    GRADES_PROB[grd] = GRADES_DIST[grd] / n_samples
    cumulat_prob += GRADES_PROB[grd]
    GRADES_CUMULAT_PROB[grd] = cumulat_prob

# generate a grade, based on the cumulative distribution
def grade_generator():
    u = np.random.uniform(0,1)
    for grd in range(MIN_GRADE,MAX_GRADE+1):
        if u < GRADES_CUMULAT_PROB[grd]:
            return grd


avg = 0
for grd in range(MIN_GRADE,MAX_GRADE+1):
    avg += grd*GRADES_PROB[grd]
print(avg)