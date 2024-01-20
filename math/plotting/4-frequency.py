#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""plt.hist(
    x=student_grades,
    bins=10,
    range=(0, 100),
    density=False,
    cumulative=False,
    histtype='bar',
    align='mid',
    orientation='vertical',
    rwidth=None,
    log=False,
    color='blue',
    label='Grades Distribution',
    stacked=False
)"""
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
plt.hist(x=student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
plt.ylim(0, 30) 
plt.xlim(0,100)
plt.xticks(np.arange(0, 101, 10)) 
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
