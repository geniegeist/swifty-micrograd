import SwiftUI
import PlaygroundSupport
import Charts
import SwiftyMicrograd

let mlp = MLP(dim_in: 2, dim_outs: [16, 16 ,1])
print(mlp)
print(mlp.parameters.count)

let m = makeMoons()
let X = m.0
let y = m.1

struct MoonPoint: Identifiable {
    var id = UUID()
    let x1: Double;
    let x2: Double;
    let family: String;
}

let data = zip(X, y).map { MoonPoint(x1: $0.0[0], x2: $0.0[1], family: $0.1 == 1 ? "A" : "B") }

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
