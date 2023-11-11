from Matrix_Vector import *
import matplotlib.pyplot as plt

points = Matrix([[-6, -115],
           [-5, -66],
           [-4, -36],
           [-3, -15],
           [-2, -2.7],
           [-1, 2],
           [1, -1.4],
           [2, -4],
           [3, -4.8],
           [4, 0],
           [5, 11],
           [6, 31]])

A = Matrix([[_ ** 2, _, 1] for _ in points.column(0)])
b = Vector(points.column(1))
x = (A.transpose().mat_mul(A)).inverse().mat_mul(A.transpose()).vector_mul(b)
graph_x = Vector([_ * 0.1 for _ in range(-100, 100)])
graph_y = (graph_x ** 2) * x[0] + graph_x * x[1] + Vector([x[2] for i in range(len(graph_x))])
y = Vector([(_ ** 2) * x[0] + _ * x[1] + x[2] for _ in points.column(0)])

A2 = Matrix([[_ ** 3, _ ** 2, _, 1] for _ in points.column(0)])
b2 = Vector(points.column(1))
x2 = (A2.transpose().mat_mul(A2)).inverse().mat_mul(A2.transpose()).vector_mul(b2)
graph_y2 = (graph_x ** 3) * x2[0] + (graph_x ** 2) * x2[1] + graph_x * x2[2] + Vector([x2[3] for i in range(len(graph_x))])
y2 = Vector([(_ ** 3) * x2[0] + (_ ** 2) * x2[1] + _ * x2[2] + x2[3] for _ in points.column(0)])

A3 = Matrix([[_ ** 4, _ ** 3, _ ** 2, _, 1] for _ in points.column(0)])
b3 = Vector(points.column(1))
x3 = (A3.transpose().mat_mul(A3)).inverse().mat_mul(A3.transpose()).vector_mul(b3)
graph_y3 = (graph_x ** 4) * x3[0] + (graph_x ** 3) * x3[1] + (graph_x ** 2) * x3[2] + graph_x * x3[3] + Vector([x3[4] for i in range(len(graph_x))])
y3 = Vector([(_ ** 4) * x3[0] + (_ ** 3) * x3[1] + (_ ** 2) * x3[2] + _ * x3[3] + x3[4] for _ in points.column(0)])

print(f"Error of Graph 1: {squared_error(y, points.column(1))}")
print(f"Error of Graph 2: {squared_error(y2, points.column(1))}")
print(f"Error of Graph 3: {squared_error(y3, points.column(1))}")

plt.scatter(points.column(0), points.column(1), color="gray", label="sample")
plt.plot(graph_x, graph_y, color="green", label="graph")
plt.plot(graph_x, graph_y2, color="blue", label="graph2")
plt.plot(graph_x, graph_y3, color="red", label="graph3")
plt.legend()
plt.grid(True)

plt.show()