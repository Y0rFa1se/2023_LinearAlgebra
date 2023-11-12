from Matrix_Vector import *
import matplotlib.pyplot as plt

points = Matrix([[-2.8, 32.4],
          [-2.1, 19.7],
          [-0.8, 5.7],
          [1.1, 2.1],
          [0.1, 1.2],
          [1.9, 8.9],
          [3.1, 25.7],
          [4.0, 41.5]])

selected = points.random_select(6)
selected2 = points.random_select(6)

A = Matrix([[_ ** 2, _, 1] for _ in selected.column(0)])
b = Vector(selected.column(1))
x = (A.transpose().mat_mul(A)).inverse().mat_mul(A.transpose()).vector_mul(b)
graph_x = Vector([_ * 0.1 for _ in range(-50, 55)])
graph_y = (graph_x ** 2) * x[0] + graph_x * x[1] + Vector([x[2] for i in range(len(graph_x))])
y = Vector([(_ ** 2) * x[0] + _ * x[1] + x[2] for _ in selected.column(0)])

A2 = Matrix([[_ ** 2, _, 1] for _ in selected2.column(0)])
b2 = Vector(selected2.column(1))
x2 = (A2.transpose().mat_mul(A2)).inverse().mat_mul(A2.transpose()).vector_mul(b2)
graph_y2 = (graph_x ** 2) * x2[0] + graph_x * x2[1] + Vector([x2[2] for i in range(len(graph_x))])
y2 = Vector([(_ ** 2) * x2[0] + _ * x2[1] + x2[2] for _ in selected2.column(0)])

print(f"Error of Graph 1: {squared_error(y, selected.column(1))}")
print(f"Error of Graph 2: {squared_error(y2, selected2.column(1))}")

plt.scatter(selected.column(0), selected.column(1), color="gray", label="sample")
plt.scatter(selected2.column(0), selected2.column(1), color="red", label="sample2")
plt.plot(graph_x, graph_y, color="green", label="graph")
plt.plot(graph_x, graph_y2, color="blue", label="graph2")
plt.legend()
plt.grid(True)

plt.show()