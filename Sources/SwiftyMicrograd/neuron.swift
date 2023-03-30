// A neuron computes a nonlinear function with scalar values

public enum ActivationFunction {
    case relu
    case tanh
}

public class Neuron: CustomStringConvertible {
    // w and b will be initialized to random values
    public var w: [Value]
    public var b: Value
    // if it is a nonlinear neuron, then a ReLu function is applied
    public let nonlinear: Bool
    public let activation: ActivationFunction
    public var parameters: [Value] {
        get { w + [b] }
    }
    
    public var description: String { return nonlinear ? "ReLUNeuron(\(w.count))" : "LinearNeuron(\(w.count))" }
    
    public init(dim_in: Int, nonlinear: Bool = true, activation: ActivationFunction = .relu) {
        self.w = (0..<dim_in).map{ _ in Value(Double.random(in: -1.0 ... 1.0)) }
        self.b = Value(0)
        self.nonlinear = nonlinear
        self.activation = activation
    }
    
    public func callAsFunction(_ x: [Value]) -> Value {
        var out = zip(w,x).reduce(Value(0.0)) { $0 + ($1.0 * $1.1) }
        out = out + b
        var act: ((Value) -> Value)!
        switch (activation) {
        case .relu:
            act = relu
            break;
        case .tanh:
            act = tanh
            break
        }
       
        return nonlinear ? act(out) : out
    }
    
    public func flushGrad() {
        parameters.forEach{ $0.grad = 0.0 }
    }
}

// MARK: Convenience methods

public extension Neuron {
    func callAsFunction(_ x: [Double]) -> Value {
        return self(x.map{ Value($0) })
    }
}
