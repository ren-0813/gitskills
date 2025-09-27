import torch

x_date = [1.0,2.0,3.0]
y_date = [2.0,4.0,6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return w * x
def loss():
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_date,y_date):
        l = loss()
        l.backward()
        print("\t grad:",x,y,w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        print("process :",epoch,l.item())

print("predict (after training)",4,forward(4).item())