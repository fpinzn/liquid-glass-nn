import AVFoundation
import CoreGraphics
import UIKit

class FrameExtractor {
    struct ExtractionProgress {
        let current: Int
        let total: Int
    }

    static func extractFrames(
        from url: URL,
        targetSize: CGSize = CGSize(width: 256, height: 256),
        progress: @escaping (ExtractionProgress) -> Void
    ) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let track = try await asset.loadTracks(withMediaType: .video).first!
        let duration = try await asset.load(.duration)
        let nominalFrameRate = try await track.load(.nominalFrameRate)
        let totalFrames = Int(CMTimeGetSeconds(duration) * Double(nominalFrameRate))

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(trackOutput)
        reader.startReading()

        var frames: [CGImage] = []
        frames.reserveCapacity(totalFrames)
        var count = 0

        let context = CIContext()

        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let scaleX = targetSize.width / ciImage.extent.width
            let scaleY = targetSize.height / ciImage.extent.height
            let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

            if let cgImage = context.createCGImage(scaled, from: CGRect(origin: .zero, size: targetSize)) {
                frames.append(cgImage)
                count += 1
                if count % 100 == 0 {
                    progress(ExtractionProgress(current: count, total: totalFrames))
                }
            }
        }

        progress(ExtractionProgress(current: frames.count, total: frames.count))
        return frames
    }
}
