import SwiftyMicrograd

let _x = [0.0, 0.0, 0.0]
let _n = MLP(dim_in: 3, dim_outs: [4,4,1])
print(_n(_x))


let l = Layer(dim_in: 2, dim_out: 2)
let x = [2.0, -2.0]
let n1 = l.neurons[0]
let n2 = l.neurons[1]

n1.b.data = 100.0
n1.w[0].data = 2.0
n1.w[1].data = 3.0

n2.b.data = 200.0
n2.w[0].data = -5.0
n2.w[1].data = 10.0

let actual_out = l(x.map{ Value($0) })
print(actual_out)
let expected_out = [
    n1.w[0].data * x[0] + n1.w[1].data * x[1] + n1.b.data,
    n2.w[0].data * x[0] + n2.w[1].data * x[1] + n2.b.data
]
print(expected_out)
