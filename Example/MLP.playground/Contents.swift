import SwiftUI
import PlaygroundSupport
import Charts
import SwiftyMicrograd

// Initialize an MLP
let mlp = MLP(dim_in: 2, dim_outs: [16, 16, 1], activation: .relu)
print(mlp)
print(mlp.parameters.count)

// Load some toy data
let m = makeMoonsFromCSV()
let X = m.0
let y = m.1.map{ $0 * 2 - 1 }

struct MoonPoint: Identifiable {
    var id = UUID()
    let x1: Double;
    let x2: Double;
    let family: String;
}

let data = zip(X, y).map { MoonPoint(x1: $0.0[0], x2: $0.0[1], family: $0.1 == 1 ? "A" : "B") }

// See: https://github.com/jordibruin/Swift-Charts-Examples/blob/main/Swift%20Charts%20Examples/Charts/PointCharts/ScatterChart.swift
struct Notebook: View {
    var body: some View {
        Chart {
            ForEach(data) { p in
                PointMark(
                    x: .value("x1", p.x1),
                    y: .value("x2", p.x2)
                )
                .foregroundStyle(by: .value("Family", p.family))
            }
        }
        .frame(width: 400, height: 400)
    }
}

PlaygroundPage.current.setLiveView(Notebook())

func hingeLoss(_ x: Value, _ y: Value) -> Value {
    let out = relu(Value(1.0) + (x * (-y)))
    return out
}

func loss() -> (Value, Double) {
    // forward pass
    let scores = X.map { mlp($0) }.map { ($0[0]) }
    
    // compute SVM loss
    let scores_and_y = zip(scores, y)
    let sum_losses = scores_and_y.reduce(Value(0)) { $0 + hingeLoss($1.0, Value(Double($1.1))) }
    let data_loss = sum_losses * 1.0 / Double(scores.count)
    
    // L2 regularization
    let alpha = 1e-4
    let reg_loss = alpha * mlp.parameters.reduce(Value(0.0)) { $0 + $1**2 }
    
    let total_loss = data_loss
    let accuracy = Double(scores_and_y.filter{ ($0.0.data > 0) == ($0.1 > 0) }.count) / Double(scores.count)
    return (total_loss, accuracy)
}

for i in 0..<100 {
    let res = loss()
    let total_loss = res.0
    let accuracy = res.1
    
    // backprop
    mlp.flushGrad()
    total_loss.backward()
    
    let learning_rate = (1.0 - 0.9 * Double(i) / 100) 
    // let learning_rate = 0.01
    mlp.parameters.forEach{
        $0.data -= learning_rate * $0.grad
    }
    
    print("Step \(i) loss \(total_loss.data), accuracy \(accuracy * 100)%")
}
