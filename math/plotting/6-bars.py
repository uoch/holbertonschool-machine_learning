#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
# Sample data
categories = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create the stacked bar graph
for i, fruit_name in enumerate(fruits):
    fruit_values = fruit[i, :]
    if i == 0:
        plt.bar(categories, fruit_values, color=colors[i], label=fruit_name)
    else:
        bottom_fruit_values = np.sum(fruit[:i, :], axis=0)
        plt.bar(categories, fruit_values, bottom=bottom_fruit_values, color=colors[i], label=fruit_name)
"""

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
print(fruit)

# Sample data
categories = ['Farrah', 'Fred', 'Felicia']
apples = fruit[0, :]
bananas = fruit[1, :]
oranges = fruit[2, :]
peaches = fruit[3, :]


plt.bar(categories, apples, color='red', label='apples')
plt.bar(categories, bananas, bottom=apples, color='yellow', label='bananas')
plt.bar(categories, oranges, bottom=np.add(
    apples, bananas), color='#ff8000', label='oranges')
plt.bar(categories, peaches, bottom=np.add(
    np.add(apples, bananas), oranges), color='#ffe5b4', label='peaches')


plt.ylabel('Quantity of Fruit')
plt.ylim([0, 80])
plt.title('Number of Fruit per Person')
plt.legend()

plt.show()
