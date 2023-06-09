// A simple Autograd engine in Swift heavily inspired by micrograd (see: https://github.com/karpathy/micrograd)
import Darwin

// Swift doesn't know the operator ** for exponentiation yet
// let's define it
precedencegroup ExponentiationPrecedence {
    associativity: right
    higherThan: MultiplicationPrecedence
}

infix operator ** : ExponentiationPrecedence

// Some useful enums
public enum Op {
    case plus
    case minus
    case mult
    case div
    case pow
    case relu
}

// The interesting part begins here
public class Value: CustomStringConvertible {
    public var data: Double
    let children: [Value]
    let op: Op?
    public internal(set) var grad: Double = 0.0
    var _backward: (() -> Void) = {}
    
    public var description: String { return "Value(data=\(data), grad=\(grad))" }
    
    // MARK: Init
    
    public init(_ data: Double, op: Op? = nil, children: [Value] = []) {
        self.data = data
        self.op = op
        self.children = children
    }
    
    // MARK: Backward
    
    public func backward() {
        var topo: [Value] = []
        var visited: Set<Value> = []
        
        func topologicalSort(node: Value) {
            if (!visited.contains(node)) {
                visited.insert(node)
                for child in node.children {
                    topologicalSort(node: child)
                }
                topo.append(node)
            }
        }
        topologicalSort(node: self)
        
        // Inititialize grad of root node to one
        self.grad = 1.0;

        // backpropagate the grad
        for n in topo.reversed() {
            n._backward()
        }
    }
    
    // MARK: Other
    
    public func flushGrad() {
        self.grad = 0.0
    }
}

// MARK: Overloading operators
public extension Value {
    static func + (left: Value, right: Value) -> Value {
        let out = Value(left.data + right.data, op: .plus, children: [left, right])
        let _backward = {
            left.grad += out.grad
            right.grad += out.grad
        }
        out._backward = _backward
        return out
    }
    
    static func - (left: Value, right: Value) -> Value {
        let out = Value(left.data - right.data, op: .minus, children: [left, right])
        let _backward = {
            left.grad += out.grad
            right.grad -= out.grad
        }
        out._backward = _backward
        return out
    }
    
    static func * (left: Value, right: Value) -> Value {
        let out = Value(left.data * right.data, op: .mult, children: [left, right])
        let _backward = {
            left.grad += right.data * out.grad
            right.grad += left.data * out.grad
        }
        out._backward = _backward
        return out
    }
    
    static func / (left: Value, right: Value) -> Value {
        let out = Value(left.data / right.data, op: .div, children: [left, right])
        let _backward = {
            left.grad += (1.0/right.data) * out.grad
            right.grad += left.data * (-1.0 / pow(right.data, 2)) * out.grad
        }
        out._backward = _backward
        return out
    }
    
    static func ** (left: Value, right: Int) -> Value {
        let x = pow(left.data, Double(right))
        let out = Value(x, op: .pow, children: [left])
        let _backward = {
            left.grad += (Double(right) * pow(left.data, Double(right - 1))) * out.grad
        }
        out._backward = _backward
        return out
    }
    
    prefix static func - (_ m: Value) -> Value {
        return m * Value(-1)
    }
}

// MARK: More Operations
public func relu(_ left: Value) -> Value {
    let out = Value(left.data < 0 ? 0 : left.data, op: .relu, children: [left])
    let _backward = {
        left.grad += (out.data > 0 ? 1 : 0) * out.grad
    }
    out._backward = _backward
    return out
}

public func tanh(_ left: Value) -> Value {
    let x = left.data
    let t = (exp(2 * x) - 1.0)/(exp(2 * x) + 1)
    let out = Value(t, children: [left])
    let _backward = {
        left.grad += (1 - pow(t,2)) * out.grad
    }
    out._backward = _backward
    return out
}

// MARK: Making operations more convenient
public extension Value {
    static func + (left: Double, right: Value) -> Value {
        return Value(left) + right
    }
    
    static func + (left: Value, right: Double) -> Value {
        return left + Value(right)
    }
    
    static func * (left: Double, right: Value) -> Value {
        return Value(left) * right
    }
    
    static func * (left: Value, right: Double) -> Value {
        return left * Value(right)
    }
    
    static func / (left: Value, right: Double) -> Value {
        return left / Value(Double(right))
    }
    
    static func / (left: Double, right: Value) -> Value {
        return Value(left) / right
    }
}

// MARK: Hashable
extension Value: Hashable {
    public static func == (lhs: Value, rhs: Value) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }
    
    public func hash(into hasher: inout Hasher) {
        return hasher.combine(ObjectIdentifier(self))
    }
}
