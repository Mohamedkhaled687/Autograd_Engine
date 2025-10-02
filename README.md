# 🚀 Autograd Engine

A lightweight automatic differentiation engine built from scratch in Python, inspired by PyTorch's autograd system. This project demonstrates the core concepts behind automatic differentiation and neural network computation graphs.

## ✨ Features

- **Automatic Differentiation**: Forward and backward propagation with gradient computation
- **Neural Network Components**: Fully implemented `Neuron`, `Layer`, and `MLP` classes
- **PyTorch Compatibility**: Tested against PyTorch for numerical accuracy
- **Clean API**: Intuitive interface similar to popular deep learning frameworks
- **Educational**: Well-documented code perfect for learning autograd concepts

## 🏗️ Architecture

### Core Components

- **`Value`**: The fundamental building block that tracks computation graphs
- **`Neuron`**: Single neuron with weights, bias, and optional ReLU activation
- **`Layer`**: Collection of neurons forming a layer
- **`MLP`**: Multi-layer perceptron for building neural networks

### Computation Graph

Each `Value` object maintains:
- `data`: The actual numerical value
- `grad`: The gradient with respect to this value
- `_prev`: Children nodes in the computation graph
- `_op`: Operation that created this value
- `_backward`: Function to compute gradients

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Autograd_Engine.git
cd Autograd_Engine
```

### Basic Usage

```python
from autograd import Value, Neuron, Layer, MLP

# Create a simple computation
x = Value(2.0)
y = Value(3.0)
z = x * y + 5
z.backward()

print(f"Result: {z.data}")      # 11.0
print(f"dx/dz: {x.grad}")       # 3.0
print(f"dy/dz: {y.grad}")       # 2.0
```

### Neural Network Example

```python
from autograd import MLP

# Create a 3-layer neural network
model = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers of 4 neurons each, 1 output

# Forward pass
x = [1.0, -2.0, 3.0]
output = model(x)
print(f"Network output: {output.data}")

# Backward pass
output.backward()
print(f"Gradients computed for {len(model.parameters())} parameters")
```

## 🧪 Testing

The engine is thoroughly tested against PyTorch to ensure numerical accuracy:

```bash
# Activate your PyTorch environment
conda activate pytorch_env

# Run tests
python test/test.py
```

**Test Results:**
```
Your engine - Forward: -20.0, Gradient: 46.0
PyTorch    - Forward: -20.0, Gradient: 46.0
✅ All tests passed! Your autograd engine matches PyTorch results.
```

## 📚 Supported Operations

| Operation | Symbol | Example |
|-----------|--------|---------|
| Addition | `+` | `z = x + y` |
| Multiplication | `*` | `z = x * y` |
| Power | `**` | `z = x ** 2` |
| ReLU | `.relu()` | `z = x.relu()` |
| Division | `/` | `z = x / y` |
| Negation | `-` | `z = -x` |

## 🎯 Use Cases

- **Learning**: Understand how automatic differentiation works under the hood
- **Prototyping**: Quick neural network experiments without heavy dependencies
- **Education**: Perfect for teaching deep learning fundamentals
- **Research**: Lightweight base for custom autograd implementations

## 🔬 How It Works

### Forward Pass
1. Operations create new `Value` objects
2. Each `Value` tracks its parents in the computation graph
3. Gradients are initialized to zero

### Backward Pass
1. Topological sort of the computation graph
2. Gradient flows backward through the graph
3. Chain rule is applied automatically

### Example Computation Graph
```
x = Value(2.0)     ──┐
                     ├── * ──→ z = Value(6.0)
y = Value(3.0)     ──┘
```

## 🙏 Acknowledgments

- Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
- PyTorch for providing reference implementations
- The deep learning community for educational resources

