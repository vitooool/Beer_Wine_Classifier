import math

def cost(activation_value,truth):
    return (truth-activation_value)**2

def activation(x,y,w1,w2,b):
    return sigmoid((x*w1)+(y*w2)+b)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def der_sigmoid_w1(S,x,truth):
    return 2*(S-truth)*S*(1-S)*x

def der_sigmoid_w2(S,y,truth):
    return 2*(S-truth)*S*(1-S)*y

def der_sigmoid_b(S,b,truth):
    return 2*(S-truth)*S*(1-S)

x = float(input("Enter x:"))
y = float(input("Enter y:"))
truth = float(input("Enter truth:"))

Learning_Rate = float(0.001)

w1 = float(1)
w2 = float(1)
b = float(1)

A1 = activation(x,y,w1,w2,b)
print("Value of Cost function before steppest ascent",cost(A1,truth))

for i in range(10000000):
    S = activation(x,y,w1,w2,b)
    w1 = w1 - Learning_Rate*der_sigmoid_w1(S,x,truth)
    w2 = w2 - Learning_Rate*der_sigmoid_w2(S,x,truth)
    b = b - Learning_Rate*der_sigmoid_b(S,b,truth)

A1 = activation(x,y,w1,w2,b)

print("w1:",w1)
print("w2:",w2)
print("b:",b)
print("Value of Cost function after steppest ascent",cost(A1,truth))
