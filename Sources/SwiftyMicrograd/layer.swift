// A layer is a collection of neurons

public class Layer: CustomStringConvertible {
    public let neurons: [Neuron]
    public let nonlinear: Bool
    public var parameters: [Value] {
        get { neurons.reduce([]) { $0 + $1.parameters } }
    }
    
    public var description: String { return "Layer of \(neurons)" }
    
    public init(dim_in: Int, dim_out: Int, nonlinear: Bool = true) {
        self.neurons = (0..<dim_out).map{ _ in Neuron(dim_in: dim_in, nonlinear: nonlinear)}
        self.nonlinear = nonlinear
    }
    
    public func callAsFunction(_ x: [Value]) -> [Value] {
        return neurons.map { $0(x) }
    }
}
