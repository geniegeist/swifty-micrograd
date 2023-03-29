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

### License 
MIT
