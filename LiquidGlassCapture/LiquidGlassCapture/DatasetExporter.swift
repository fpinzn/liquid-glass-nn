import UIKit
import Foundation

class DatasetExporter {
    let outputDir: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        outputDir = docs.appendingPathComponent("liquid_glass_dataset", isDirectory: true)
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    }

    func clear() throws {
        if FileManager.default.fileExists(atPath: outputDir.path) {
            try FileManager.default.removeItem(at: outputDir)
        }
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    }

    func saveInput(_ image: CGImage, index: Int) throws {
        let url = outputDir.appendingPathComponent(String(format: "input_%05d.png", index))
        guard let data = UIImage(cgImage: image).pngData() else { return }
        try data.write(to: url)
    }

    func saveComposited(_ image: UIImage, index: Int) throws {
        let url = outputDir.appendingPathComponent(String(format: "composited_%05d.png", index))
        guard let data = image.pngData() else { return }
        try data.write(to: url)
    }

    func saveManifest(frameCount: Int, size: CGSize, glassFraction: CGFloat) throws {
        let manifest: [String: Any] = [
            "frame_count": frameCount,
            "width": Int(size.width),
            "height": Int(size.height),
            "glass_shape": "circle",
            "glass_fraction": glassFraction,
            "format": "input_XXXXX.png paired with composited_XXXXX.png"
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outputDir.appendingPathComponent("manifest.json"))
    }

    var shareURL: URL { outputDir }
}
