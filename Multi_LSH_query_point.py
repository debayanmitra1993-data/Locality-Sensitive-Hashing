import numpy as np

MAX_ITER = 100
x_q = np.array([6.5,-39])

set_points = set()
NUM_NEIGHBOURS = 5

for i in range(MAX_ITER):
  mylsh = LSH()
  mylsh.fit(X)
  preds = mylsh.predict(x_q)
  if len(set_points) == 0:
    set_points = set_points.union(preds)
  else:
    if len(set_points.intersection(preds)) < NUM_NEIGHBOURS:
      break
    else:
      set_points = set_points.intersection(preds)
