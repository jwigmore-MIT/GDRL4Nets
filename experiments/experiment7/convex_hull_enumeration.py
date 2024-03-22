import numpy as np
from scipy.spatial import ConvexHull
from diversipy import polytope
import json


## EXAMPLE FOR SIMPLE VERTICES
# Example: Define vertices of a 3D convex polytope
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])

# Step 2: Compute the convex hull
hull = ConvexHull(vertices)

# Step 3: Extract hyperplane equations
# The equations attribute contains the coefficients of the hyperplane equations
equations = hull.equations

# Print the inequalities
# print("The linear inequality constraints (A*x <= b) are:")
# for eq in equations:
#     # eq[:-1] are the coefficients of the variables (normal vector), eq[-1] is the offset
#     print(f"{eq[:-1]} * x <= {-eq[-1]}")

# Each printed inequality corresponds to one face of the convex hull

lower = np.zeros(vertices.shape[1])
upper = np.ones(vertices.shape[1])*10000
A = equations[:,:-1]
b = -equations[:,-1]

samples = polytope.sample(10, lower, upper, A1 =A, b1 = b)


## EXAMPLE FOR POINTS FROM CONVEX_SPACE_DICTIONARY
context_space_dict = json.load(open("SH2u_lf1.32_context_space-nondominated.json", 'rb'))
vertex_arrival_rates = np.array([context_space_dict["context_dicts"][str(i)]["arrival_rates"] for i in  range(context_space_dict["num_envs"])])
# add the zero vector to the vertex_arrival_rates
vertex_arrival_rates = np.vstack((vertex_arrival_rates, np.zeros(vertex_arrival_rates.shape[1])))

hull = ConvexHull(vertex_arrival_rates)
equations = hull.equations
lower = np.zeros(vertex_arrival_rates.shape[1])
upper = np.ones(vertex_arrival_rates.shape[1])*10000
A = equations[:,:-1]
b = -equations[:,-1]

samples = polytope.sample(100, lower, upper, A1 =A, b1 = b)