// swift-tools-version: 5.9
import PackageDescription
let package = Package(
    name: "NeuralRazzTrainer",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "NeuralRazzTrainer", path: ".")
    ]
)
