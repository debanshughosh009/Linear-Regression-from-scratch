
# y = m.x+b
# E = 1/n . sum(Yi - (m.Xi + b))**2    - MSE
# dE/dm = -2/n . sum(Xi.(Yi - (m.Xi + b)))
# dE/db = -2/n . sum(Yi - (m.Xi + b))
# m = m - L.dE/dm       , L = learning rate, lower is better but slower
# b = b - L.dE/db

import pandas as pd
import matplotlib.pyplot as plt

data = pd. read_csv('Salary_Data.csv')
print(data.head())

plt.scatter(data.YearsExperience, data.Salary)
plt.show()

def loss_function(m, b, points):
  total_error = 0
  for i in range(len(points)):
    x=points.iloc[i].YearsExperience
    y=points.iloc[i].Salary
    total_error += (y - (x*m + b))**2
  total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
  m_gradient=0
  b_gradient=0
  n=len(points)

  for i in range(n):
    x=points.iloc[i].YearsExperience
    y=points.iloc[i].Salary
    m_gradient += -(2/n)*x*(y-(m_now*x+b_now))
    b_gradient += -(2/n)*y-(m_now*x+b_now)

  m = m_now - m_gradient * L
  b = b_now - b_gradient * L
  return m, b

m=0
b=0
L=0.0001
epochs=200

for i in range(epochs):
  if i%50 == 0:
    print(f"Epoch: {i}")
  m, b = gradient_descent(m, b, data, L)

print(m,b)

plt.scatter(data.YearsExperience, data.Salary, color="black")
plt.plot(list(range(0,15)), [m*x +b for x in range(0,15)], color="red")
plt.show()


# Calculate and print the R2 score
from sklearn.metrics import r2_score
y_true = data.Salary
y_pred = m * data.YearsExperience + b
print('R2:', r2_score(y_true, y_pred))
