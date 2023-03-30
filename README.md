# Swifty micrograd 
This is a port of Karpathy's micrograd in Swift. 

### What is micrograd?
Micrograd is a simple and tiny implementation of backpropagation opearting over scalar values. This autograd engine is enough to train neural networks. The rest is making it fast (e.g. by vectorization).

### Example usage

```
let a = Value(-4.0)
let b = Value(2.0)
let c = a + b
let d = a * b + b**3
let c_1 = c + c + 1
let c_2 = c_1 + 1 + c_1 - a
let d_1 = d + d * 2 + relu(b + a)
let d_2 = d_1 + 3 * d_1 + relu(b - a)
let e = c_2 - d_2
let f = e**2
let g = f / 2.0
let g_1 = g + (10.0) / f

assert(abs(g_1.data - 24.7041) < 1e-4)

g_1.backward()
assert(abs(a.grad - 138.8338) < 1e-4)
assert(abs(b.grad - 645.5773) < 1e-4)
```

### 3 layer network

A simple 3 layer network can be explored in the Swift playground `MLP.playground`.
```
let mlp = MLP(dim_in: 2, dim_outs: [16, 16, 1], activation: .relu)
```

**Hinge loss (for the dataset `sklearn.datasets.make_moons`)**
```
Step 0 loss 2.333378857315669, accuracy 50.0%
Step 1 loss 0.9342276431697976, accuracy 50.0%
Step 2 loss 1.3801763192976588, accuracy 77.0%
Step 3 loss 0.4883205083530057, accuracy 79.0%
Step 4 loss 0.49335226400935034, accuracy 82.0%
Step 5 loss 0.30098028854311853, accuracy 86.0%
Step 6 loss 0.24895841027538348, accuracy 87.0%
Step 7 loss 0.2190228586034104, accuracy 88.0%
Step 8 loss 0.20418502548934328, accuracy 92.0%
Step 9 loss 0.19425285709158457, accuracy 94.0%
Step 10 loss 0.1753767475031612, accuracy 95.0%
Step 11 loss 0.18618003544586537, accuracy 92.0%
Step 12 loss 0.1410612708254436, accuracy 95.0%
Step 13 loss 0.13477688997747586, accuracy 96.0%
Step 14 loss 0.13464042057525094, accuracy 95.0%
Step 15 loss 0.11904167240578056, accuracy 96.0%
Step 16 loss 0.10872678309483241, accuracy 95.0%
Step 17 loss 0.13948166384550217, accuracy 95.0%
Step 18 loss 0.1972398662849428, accuracy 93.0%
Step 19 loss 0.24020840110182842, accuracy 89.0%
Step 20 loss 0.14478743131254226, accuracy 92.0%
...
Step 99 loss 0.0, accuracy 100.0%


### License 
MIT
