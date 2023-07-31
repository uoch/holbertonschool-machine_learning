import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3
plt.subplot(321)
plt.plot(y0, color='red')


mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
plt.subplot(322)
plt.scatter(x1, y1, c='m', s=10)
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title('Scatter Plot', fontsize='x-small')


x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)
plt.subplot(323)
plt.plot(x2, y2, 'b-')
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')
plt.yscale('log')


x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)


plt.subplot(324)
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, c='green', label='Ra-226')
plt.xlim(x3[0], x3[-1])
plt.ylim(min(y31[-1], y32[-1]), y31[0])
plt.legend()
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
plt.subplot2grid((3, 2), (2, 0), colspan=2)
plt.hist(x=student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
plt.ylim(0, 30)
plt.xlim(0, 100)
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')
plt.suptitle('All in One', fontsize='x-small')
plt.tight_layout()
plt.text(11000, y31[-1], 'C-14', color='red', fontsize='x-small')

plt.show()
