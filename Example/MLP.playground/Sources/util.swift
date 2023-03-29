import Foundation

func load(file named: String) -> String? {
    guard let fileUrl = Bundle.main.url(forResource: named, withExtension: "csv") else {
        return nil
    }
    
    guard let content = try? String(contentsOf: fileUrl, encoding: .utf8) else {
        return nil
    }
    
    return content
}

func csv(_ data: String) -> [[Double]] {
    var result: [[Double]] = []
    let d = data.components(separatedBy: "\n")
    let rows = d.prefix(d.count - 1)
    for row in rows {
        let columns = row.components(separatedBy: ",")
        result.append(columns.map({ Double($0)! }))
    }
    return result
}

func csv_y(_ data: String) -> [Double] {
    var result: [Double] = []
    let d = data.components(separatedBy: "\n")
    let rows = d.prefix(d.count - 1)
    for row in rows {
        result.append(Double(row)!)
    }
    return result
}

public func makeMoons() -> ([[Double]], [Double]) {
    let X = csv(load(file: "moons_X")!)
    let y = csv_y(load(file: "moons_y")!)
    return (X,y)
}
