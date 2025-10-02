import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from autograd import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    print(f"Your engine - Forward: {ymg.data}, Gradient: {xmg.grad}")
    print(f"PyTorch    - Forward: {ypt.data.item()}, Gradient: {xpt.grad.item()}")
    
    # forward pass went well
    assert abs(ymg.data - ypt.data.item()) < 1e-6, f"Forward pass failed: {ymg.data} vs {ypt.data.item()}"
    # backward pass went well
    assert abs(xmg.grad - xpt.grad.item()) < 1e-6, f"Backward pass failed: {xmg.grad} vs {xpt.grad.item()}"
    
    print("✅ All tests passed! Your autograd engine matches PyTorch results.")


if __name__ == "__main__":
    test_sanity_check()