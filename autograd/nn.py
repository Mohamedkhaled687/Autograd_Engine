import random
from .engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron :
    def __init__(self , nin , nonlin = True):
        self.w = [Value(random.uniform(-1 , 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1 , 1))
        self.nonlin = nonlin

    def __call__(self , x):
        # w * x + b
        act = sum((w * x for w , x in zip(self.w , x)) , self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer : 
    def __init__(self , nin , nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self , x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(repr(n) for n in self.neurons)}]"
    

class MLP :
    def __init__(self , nin , nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i] , sz[i + 1]) for i in range(len(sz) - 1)]        

    def __call__(self , x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(repr(l) for l in self.layers)}]"