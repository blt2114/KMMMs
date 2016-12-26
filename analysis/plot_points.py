import matplotlib.pyplot as plt
import sys

points = []
for l in sys.stdin:
    points.append(float(l))

plt.plot(points)
plt.title("Log Probability of Joint Distribution by Iteration")
plt.xlabel("Iteration")
plt.ylabel("log_p joint")
plt.show()
