import Foundation

public func generateMoonsData(_ n: Int) -> ([[Double]], [Int]) {
    var labels: [Int] = []
    var data: [[Double]] = []
    
    for i in stride(from: 0, to: Double.pi, by: (Double.pi * 2)/Double(n)) {
        let p = [
            cos(i) + Double.random(in: -0.1...0.1),
            sin(i) + Double.random(in: -0.1...0.1)
        ]
        data.append(p)
        labels.append(-1)
        
        let q = [
            1 - cos(i) + Double.random(in: -0.1...0.1),
            1 - sin(i) + Double.random(in: -0.1...0.1) - 0.5
        ]
        data.append(q)
        labels.append(1)
    }
    
    return (data, labels)
}
