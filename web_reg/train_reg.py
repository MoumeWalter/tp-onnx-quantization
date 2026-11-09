import torch, torch.nn as nn

torch.manual_seed(0)
x = torch.rand(200, 1) * 10
y = 3 * x + 2 + torch.randn(200, 1) * 0.5

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(500):
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()
print('poids', model.weight.item(), 'biais', model.bias.item())

torch.onnx.export(model, torch.randn(1, 1), 'linreg.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=13, dynamo=False)
