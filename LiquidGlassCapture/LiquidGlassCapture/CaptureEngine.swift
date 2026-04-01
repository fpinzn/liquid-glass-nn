import SwiftUI
import UIKit

@Observable
class CaptureEngine {
    var status: Status = .idle
    var progress: Float = 0
    var statusText: String = "Ready"
    var datasetURL: URL?

    enum Status {
        case idle, extractingFrames, capturingFrames, done, error
    }

    private let captureSize = CGSize(width: 256, height: 256)
    private let glassFraction: CGFloat = 0.8
    private let viewCapture = ViewCapture()
    private let exporter = DatasetExporter()

    @MainActor
    func process(videoURL: URL) async {
        do {
            status = .extractingFrames
            statusText = "Extracting frames..."
            progress = 0

            let frames = try await FrameExtractor.extractFrames(
                from: videoURL,
                targetSize: captureSize
            ) { [weak self] p in
                Task { @MainActor in
                    self?.progress = Float(p.current) / Float(max(p.total, 1))
                    self?.statusText = "Extracting: \(p.current)/\(p.total)"
                }
            }

            guard !frames.isEmpty else {
                status = .error
                statusText = "No frames extracted"
                return
            }

            try exporter.clear()
            viewCapture.setup(size: captureSize)
            defer { viewCapture.teardown() }

            status = .capturingFrames
            let total = frames.count

            for (i, frame) in frames.enumerated() {
                // Save original input
                try exporter.saveInput(frame, index: i)

                // Capture through Apple's Liquid Glass
                let inputImage = UIImage(cgImage: frame)
                let captureView = CaptureView(
                    backgroundImage: inputImage,
                    size: captureSize,
                    glassFraction: glassFraction
                )

                if let composited = await viewCapture.capture(captureView, size: captureSize) {
                    try exporter.saveComposited(composited, index: i)
                }

                progress = Float(i + 1) / Float(total)
                if (i + 1) % 50 == 0 || i == total - 1 {
                    statusText = "Capturing: \(i + 1)/\(total)"
                }
            }

            try exporter.saveManifest(
                frameCount: frames.count,
                size: captureSize,
                glassFraction: glassFraction
            )

            status = .done
            statusText = "Done! \(frames.count) frame pairs captured."
            datasetURL = exporter.shareURL

        } catch {
            status = .error
            statusText = "Error: \(error.localizedDescription)"
        }
    }
}
