#!/usr/bin/env python3
"""6-bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Define bars function"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    names = ['Farrah', 'Fred', 'Felicia']

    plt.bar(names, apples, width=0.5, color='red', label='apples')
    plt.bar(names, bananas, width=0.5, color='yellow', label='bananas',
            bottom=apples)
    plt.bar(names, oranges, width=0.5, color='#ff8000', label='oranges',
            bottom=apples + bananas)
    plt.bar(names, peaches, width=0.5, color='#ffe5b4', label='peaches',
            bottom=apples + bananas + oranges)

    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
