// A multi layer perception is a collection of layers

public class MLP: CustomStringConvertible {
    public let layers: [Layer]
    public var parameters: [Value] {
        get { layers.reduce([]) { $0 + $1.parameters } }
    }
    
    public var description: String { return "MLP of \(layers)" }
    
    public init(dim_in: Int, dim_outs: [Int], activation: ActivationFunction = .relu) {
        let dims = zip([dim_in] + dim_outs, dim_outs)
        self.layers = dims.enumerated().map{ Layer(dim_in: $1.0, dim_out: $1.1, nonlinear: $0 < dim_outs.count - 1, activation: activation) }
    }
    
    public func callAsFunction(_ x: [Value]) -> [Value] {
        var out = x
        for l in layers {
            out = l(out)
        }
        return out
    }
    
    public func flushGrad() {
        parameters.forEach{ $0.grad = 0.0 }
    }
}

// MARK: Convenience methods

public extension MLP {
    func callAsFunction(_ x: [Double]) -> [Value] {
        return self(x.map{ Value($0) })
    }
}

