// A neuron computes a nonlinear function with scalar values

public class Neuron: CustomStringConvertible {
    // w and b will be initialized to random values
    public var w: [Value]
    public var b: Value
    // if it is a nonlinear neuron, then a ReLu function is applied
    public let nonlinear: Bool
    public var parameters: [Value] {
        get { w + [b] }
    }
    
    public var description: String { return nonlinear ? "ReLUNeuron(\(w.count))" : "LinearNeuron(\(w.count))" }
    
    public init(dim_in: Int, nonlinear: Bool = true) {
        self.w = (0..<dim_in).map{ _ in Value(Double.random(in: -1.0 ... 1.0)) }
        self.b = Value(0.0)
        self.nonlinear = nonlinear
    }
    
    public func callAsFunction(_ x: [Value]) -> Value {
        var out = zip(w,x).reduce(Value(0.0)) { $0 + $1.0 * $1.1 }
        out = out + b
        return nonlinear ? relu(out) : out
    }
}

// MARK: Convenience methods

public extension Neuron {
    func callAsFunction(_ x: [Double]) -> Value {
        return self(x.map{ Value($0) })
    }
}
