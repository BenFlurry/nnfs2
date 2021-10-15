# inputs (x), weights (w), bias (b)
x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# FORWARD PASS
# multiply inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# add bias
z = xw0 + xw1 + xw2 + b

# relu activation function
y = max(z, 0)
print(y)

# BACKWARD PASS

'''
dReLU/dx 
= (ReLU(sum(mult(x[0], w[0]), mult(x[1], w[1]), mult(x[2], w[2])))
= dReLU/dsum * dsum/dmul * dmul/dx
= dReLU/dsum * dsum/dxw * dxw/dx

dReLU/dw 
= (ReLU(sum(mult(x[0], w[0]), mult(x[1], w[1]), mult(x[2], w[2])))
= dReLU/dsum * dsum/dmul * dmul/dw
= dReLU/dsum * dsum/dxw * dxw/dw
'''

# derivative from the previous layer (assume 1 for this case)
dvalue = 1.0

# derivative of relu
drelu_dz = dvalue * (1.0 if z > 0 else 0.0)

# partial derivatives of the sum of weights and biases
# derivative of d/dx(x) = 1
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
# chain rule of relu function: dR/dxw = dR/dz * dz/dxw
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2

# partial derivatives of the multiplication of weights and biases
# partial derivative of d/dx(f(w,x)) = x, w
# so d/dw(w) = x and d/dx(x)
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
# chain rule of relu function dR/dx = dR/dxw * dx/dw
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
# chain rule of relu function dR/dw = dR/dxw * dw/dx
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2

# OPTIMISED VERSION:
# wrt x
drelu_dx0 = dvalue * (1.0 if z > 0 else 0.0) * w[0]
drelu_dx1 = dvalue * (1.0 if z > 0 else 0.0) * w[1]
drelu_dx2 = dvalue * (1.0 if z > 0 else 0.0) * w[2]
# wrt w
drelu_dw0 = dvalue * (1.0 if z > 0 else 0.0) * x[0]
drelu_dw1 = dvalue * (1.0 if z > 0 else 0.0) * x[1]
drelu_dw2 = dvalue * (1.0 if z > 0 else 0.0) * x[2]
# combine into a vector:
dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = dsum_db

# use the backprop to optimise (minimise loss function dx, db, dw)
print(w, b)
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += - 0.001 * db
print(w, b)
# and forward pass again to see values
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
z = xw0 + xw1 + xw2 + b
y = max(z, 0)
print(y)


# NUMPY VERSION
