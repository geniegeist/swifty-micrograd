import SwiftyMicrograd

// Initialize nn with initial weights between -1.0 and 1.0
// Use tanh as an activation function
let n = MLP(dim_in: 3, dim_outs: [4,4,1], activation: .tanh)
let xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
// desired targets
let ys = [1.0, -1.0, -1.0, 1.0]
let ypred = xs.map{ n($0)[0] }

// It is likely that the loss function explodes,
// just run this playbook again until the loss converges
// Alternativly you can decrease the learning rate or set
// the initial weights of the MLP between -0.1 and 0.1
for i in 0..<20 {
    let ypred = xs.map{ n($0)[0] }
    let loss = zip(ypred, ys).reduce(Value(0)) { $0 + ($1.0 - Value($1.1))**2 }
    
    // backprop
    n.flushGrad()
    loss.backward()
    
    //    let learning_rate = 1.0 - 0.9 * Double(i) / 100
    let learning_rate = 0.1
    n.parameters.forEach{
        $0.data -= learning_rate * $0.grad
    }
    
    print("Step \(i) loss \(loss.data)")
}
