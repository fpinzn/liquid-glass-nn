# Liquid Glass Dataset Capture iOS App

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an iOS app that captures Apple's real Liquid Glass effect as ground truth training data (displacement maps + composited frames) from video input.

**Architecture:** SwiftUI app that loads a video, extracts frames, renders each behind a `.glassEffect()` overlay, and captures the result via `drawHierarchy`. Uses a UV-gradient trick to extract the displacement field. Exports dataset via share sheet / AirDrop.

**Tech Stack:** SwiftUI, AVFoundation, UIKit (drawHierarchy capture), CoreGraphics, Accelerate (EXR export)

---

## Key Technical Constraints

1. **Liquid Glass cannot be captured offscreen.** `ImageRenderer` produces blank results. Must use `UIView.drawHierarchy(in:afterScreenUpdates:true)` with the view in a live `UIWindow`.
2. **Displacement field is constant for a static glass shape.** We capture it once via the UV gradient trick, then only capture composited results per-frame.
3. **The UV trick:** Render a gradient image (R=x_normalized, G=y_normalized, B=0) behind the glass. Capture the result. Under the glass region, each pixel's RGB tells us where it sampled from. `displacement = captured_uv - original_uv`.
4. **Specular extraction:** Render a solid black image behind the glass. Anything visible in the capture is specular/reflection/caustic from the glass itself.

## File Structure

```
LiquidGlassCapture/
├── LiquidGlassCapture.xcodeproj
├── LiquidGlassCapture/
│   ├── LiquidGlassCaptureApp.swift      # App entry point
│   ├── ContentView.swift                 # Main UI: video picker + progress
│   ├── CaptureView.swift                 # The glass-over-image view for capture
│   ├── CaptureEngine.swift               # Frame extraction + capture orchestration
│   ├── FrameExtractor.swift              # AVAsset → [CGImage] extraction
│   ├── DisplacementExtractor.swift       # UV trick + specular extraction
│   ├── DatasetExporter.swift             # Save to disk + share sheet
│   ├── Assets.xcassets/
│   └── Info.plist
```

---

### Task 1: Xcode Project Setup

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture.xcodeproj`
- Create: `LiquidGlassCapture/LiquidGlassCapture/LiquidGlassCaptureApp.swift`
- Create: `LiquidGlassCapture/LiquidGlassCapture/ContentView.swift`

- [ ] **Step 1: Create Xcode project**

Create a new SwiftUI iOS app project targeting iOS 26.0, deployment target iPhone. App name: `LiquidGlassCapture`. Bundle ID: `com.liquidglass.capture`.

- [ ] **Step 2: Set up LiquidGlassCaptureApp.swift**

```swift
import SwiftUI

@main
struct LiquidGlassCaptureApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

- [ ] **Step 3: Set up placeholder ContentView.swift**

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Text("Liquid Glass Capture")
                .font(.largeTitle)
            Text("Ready")
                .foregroundStyle(.secondary)
        }
    }
}
```

- [ ] **Step 4: Build and run on device to verify setup**

Run: Xcode Build & Run on iPhone Pro 16
Expected: App launches showing "Liquid Glass Capture" / "Ready"

- [ ] **Step 5: Commit**

```bash
git add LiquidGlassCapture/
git commit -m "feat: initial Xcode project setup for dataset capture app"
```

---

### Task 2: Frame Extractor (Video → CGImages)

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture/FrameExtractor.swift`

- [ ] **Step 1: Implement FrameExtractor**

```swift
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

        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()

            // Scale to target size
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
```

- [ ] **Step 2: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/FrameExtractor.swift
git commit -m "feat: add video frame extraction with AVAssetReader"
```

---

### Task 3: Capture View (Glass Effect Overlay)

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture/CaptureView.swift`

This is the view that renders an image behind a Liquid Glass overlay. It will be placed in a live UIWindow for `drawHierarchy` capture.

- [ ] **Step 1: Implement CaptureView**

```swift
import SwiftUI

struct CaptureView: View {
    let backgroundImage: UIImage
    let glassShape: GlassShape
    let size: CGSize

    enum GlassShape {
        case circle
        case roundedRect(cornerRadius: CGFloat)
        case capsule
    }

    var body: some View {
        ZStack {
            Image(uiImage: backgroundImage)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: size.width, height: size.height)
                .clipped()

            glassOverlay
        }
        .frame(width: size.width, height: size.height)
    }

    @ViewBuilder
    private var glassOverlay: some View {
        // Glass covers the full frame — we want displacement for the entire image
        switch glassShape {
        case .circle:
            Color.clear
                .frame(width: size.width, height: size.height)
                .glassEffect(.regular, in: .circle)
        case .roundedRect(let radius):
            Color.clear
                .frame(width: size.width, height: size.height)
                .glassEffect(.regular, in: .rect(cornerRadius: radius))
        case .capsule:
            Color.clear
                .frame(width: size.width, height: size.height)
                .glassEffect(.regular, in: .capsule)
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/CaptureView.swift
git commit -m "feat: add CaptureView with glass effect overlay"
```

---

### Task 4: Screenshot Capture via drawHierarchy

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture/ViewCapture.swift`

This is the critical piece — rendering the SwiftUI view in a live UIWindow and capturing it.

- [ ] **Step 1: Implement ViewCapture**

```swift
import SwiftUI
import UIKit

class ViewCapture {
    private var captureWindow: UIWindow?
    private var hostingController: UIHostingController<AnyView>?

    /// Set up a dedicated UIWindow for off-screen-ish capture.
    /// The window must be in the hierarchy for drawHierarchy to work.
    func setup(size: CGSize) {
        guard let scene = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first else { return }

        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(origin: .zero, size: size)
        window.windowLevel = .normal - 1 // Behind main window
        window.isHidden = false
        window.alpha = 0.01 // Nearly invisible but still in hierarchy

        let hc = UIHostingController(rootView: AnyView(EmptyView()))
        hc.view.frame = CGRect(origin: .zero, size: size)
        window.rootViewController = hc
        window.makeKeyAndVisible()

        self.captureWindow = window
        self.hostingController = hc
    }

    /// Capture a SwiftUI view as a UIImage using drawHierarchy.
    @MainActor
    func capture<V: View>(_ view: V, size: CGSize) async -> UIImage? {
        guard let hc = hostingController else { return nil }

        hc.rootView = AnyView(view)
        hc.view.frame = CGRect(origin: .zero, size: size)
        hc.view.setNeedsLayout()
        hc.view.layoutIfNeeded()

        // Give the glass effect time to render
        try? await Task.sleep(for: .milliseconds(50))

        let renderer = UIGraphicsImageRenderer(size: size)
        let image = renderer.image { _ in
            hc.view.drawHierarchy(in: CGRect(origin: .zero, size: size), afterScreenUpdates: true)
        }
        return image
    }

    func teardown() {
        captureWindow?.isHidden = true
        captureWindow = nil
        hostingController = nil
    }
}
```

- [ ] **Step 2: Test capture with a simple colored view**

Add a temporary test in ContentView that captures a red rectangle and displays it:

```swift
struct ContentView: View {
    @State private var capturedImage: UIImage?
    let viewCapture = ViewCapture()

    var body: some View {
        VStack {
            if let img = capturedImage {
                Image(uiImage: img)
                    .resizable()
                    .frame(width: 256, height: 256)
                Text("Capture OK")
            }
            Button("Test Capture") {
                Task { await testCapture() }
            }
        }
    }

    func testCapture() async {
        let size = CGSize(width: 256, height: 256)
        viewCapture.setup(size: size)
        let testView = Color.red.frame(width: 256, height: 256)
        capturedImage = await viewCapture.capture(testView, size: size)
        viewCapture.teardown()
    }
}
```

Run on device. Tap "Test Capture". Expected: A red square appears, confirming drawHierarchy works.

- [ ] **Step 3: Test capture WITH glass effect**

Replace the test view:

```swift
let testView = ZStack {
    Color.blue
    Color.clear
        .frame(width: 256, height: 256)
        .glassEffect(.regular, in: .circle)
}
.frame(width: 256, height: 256)
```

Run on device. Expected: A blue square with visible glass distortion in a circle. If the glass appears, the capture pipeline works. If it's blank/just blue, we need to increase the sleep duration or try a different capture approach.

- [ ] **Step 4: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/ViewCapture.swift
git commit -m "feat: add drawHierarchy-based view capture for glass effects"
```

---

### Task 5: Displacement & Specular Extraction

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture/DisplacementExtractor.swift`

- [ ] **Step 1: Implement UV gradient image generator**

```swift
import UIKit
import CoreGraphics
import Accelerate

class DisplacementExtractor {
    /// Generate a UV gradient image where R = normalized x, G = normalized y, B = 0.
    /// Each pixel encodes its own coordinate.
    static func generateUVGradient(size: CGSize) -> UIImage {
        let width = Int(size.width)
        let height = Int(size.height)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                pixels[i + 0] = UInt8(Float(x) / Float(width - 1) * 255) // R = x
                pixels[i + 1] = UInt8(Float(y) / Float(height - 1) * 255) // G = y
                pixels[i + 2] = 0  // B = 0
                pixels[i + 3] = 255 // A = 1
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let cgImage = ctx.makeImage()!
        return UIImage(cgImage: cgImage)
    }

    /// Extract displacement field by comparing UV capture with original UV gradient.
    /// Returns a float array of [dx, dy] per pixel, normalized to [-1, 1].
    static func extractDisplacement(
        uvCapture: UIImage,
        size: CGSize
    ) -> [Float] {
        let width = Int(size.width)
        let height = Int(size.height)
        let pixelCount = width * height

        // Get pixel data from captured UV image
        guard let cgImage = uvCapture.cgImage else { return [] }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var capturedPixels = [UInt8](repeating: 0, count: pixelCount * 4)
        let ctx = CGContext(
            data: &capturedPixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.draw(cgImage, in: CGRect(origin: .zero, size: size))

        // Compute displacement: captured_uv - original_uv
        var displacement = [Float](repeating: 0, count: pixelCount * 2)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let di = (y * width + x) * 2

                let capturedX = Float(capturedPixels[i + 0]) / 255.0
                let capturedY = Float(capturedPixels[i + 1]) / 255.0
                let originalX = Float(x) / Float(width - 1)
                let originalY = Float(y) / Float(height - 1)

                displacement[di + 0] = capturedX - originalX // dx
                displacement[di + 1] = capturedY - originalY // dy
            }
        }

        return displacement
    }

    /// Extract specular highlights by capturing glass over a black background.
    /// Returns a float array of intensity per pixel.
    static func extractSpecular(
        blackCapture: UIImage,
        size: CGSize
    ) -> [Float] {
        let width = Int(size.width)
        let height = Int(size.height)
        let pixelCount = width * height

        guard let cgImage = blackCapture.cgImage else { return [] }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: pixelCount * 4)
        let ctx = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.draw(cgImage, in: CGRect(origin: .zero, size: size))

        // Luminance of each pixel = specular intensity
        var specular = [Float](repeating: 0, count: pixelCount)
        for i in 0..<pixelCount {
            let pi = i * 4
            let r = Float(pixels[pi + 0]) / 255.0
            let g = Float(pixels[pi + 1]) / 255.0
            let b = Float(pixels[pi + 2]) / 255.0
            specular[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        return specular
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/DisplacementExtractor.swift
git commit -m "feat: add displacement and specular extraction via UV trick"
```

---

### Task 6: Dataset Exporter

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture/DatasetExporter.swift`

- [ ] **Step 1: Implement DatasetExporter**

Saves frames as PNG, displacement as binary float32 files (simpler than EXR on iOS, Python reads these trivially with numpy).

```swift
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

    /// Save an input frame as PNG
    func saveInput(_ image: CGImage, index: Int) throws {
        let url = outputDir.appendingPathComponent(String(format: "input_%05d.png", index))
        let uiImage = UIImage(cgImage: image)
        guard let data = uiImage.pngData() else { return }
        try data.write(to: url)
    }

    /// Save composited (glass-applied) frame as PNG
    func saveComposited(_ image: UIImage, index: Int) throws {
        let url = outputDir.appendingPathComponent(String(format: "composited_%05d.png", index))
        guard let data = image.pngData() else { return }
        try data.write(to: url)
    }

    /// Save displacement field as raw float32 binary (width * height * 2 floats: dx, dy)
    func saveDisplacement(_ displacement: [Float], width: Int, height: Int) throws {
        let url = outputDir.appendingPathComponent("displacement.bin")
        let data = displacement.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url)

        // Also save metadata
        let meta: [String: Any] = ["width": width, "height": height, "channels": 2, "dtype": "float32"]
        let metaData = try JSONSerialization.data(withJSONObject: meta)
        let metaUrl = outputDir.appendingPathComponent("displacement_meta.json")
        try metaData.write(to: metaUrl)
    }

    /// Save specular field as raw float32 binary (width * height floats)
    func saveSpecular(_ specular: [Float], width: Int, height: Int) throws {
        let url = outputDir.appendingPathComponent("specular.bin")
        let data = specular.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url)

        let meta: [String: Any] = ["width": width, "height": height, "channels": 1, "dtype": "float32"]
        let metaData = try JSONSerialization.data(withJSONObject: meta)
        let metaUrl = outputDir.appendingPathComponent("specular_meta.json")
        try metaData.write(to: metaUrl)
    }

    /// Save dataset manifest
    func saveManifest(frameCount: Int, size: CGSize) throws {
        let manifest: [String: Any] = [
            "frame_count": frameCount,
            "width": Int(size.width),
            "height": Int(size.height),
            "format": "input_XXXXX.png + composited_XXXXX.png + displacement.bin + specular.bin",
            "displacement_note": "Single displacement field for static glass shape. float32, shape [H, W, 2], values in [-1, 1]",
            "specular_note": "Single specular field. float32, shape [H, W], values in [0, 1]"
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: .prettyPrinted)
        let url = outputDir.appendingPathComponent("manifest.json")
        try data.write(to: url)
    }

    /// Get the output directory URL for sharing
    var shareURL: URL { outputDir }
}
```

- [ ] **Step 2: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/DatasetExporter.swift
git commit -m "feat: add dataset exporter with PNG and binary float32 output"
```

---

### Task 7: Capture Engine (Orchestration)

**Files:**
- Create: `LiquidGlassCapture/LiquidGlassCapture/CaptureEngine.swift`

This ties everything together: extract frames → capture displacement → capture specular → capture composited per frame → export.

- [ ] **Step 1: Implement CaptureEngine**

```swift
import SwiftUI
import UIKit

@Observable
class CaptureEngine {
    var status: Status = .idle
    var progress: Float = 0
    var statusText: String = "Ready"
    var datasetURL: URL?

    enum Status {
        case idle, extractingFrames, capturingDisplacement, capturingSpecular, capturingFrames, saving, done, error
    }

    private let captureSize = CGSize(width: 256, height: 256)
    private let viewCapture = ViewCapture()
    private let exporter = DatasetExporter()

    @MainActor
    func process(videoURL: URL) async {
        do {
            // Step 1: Extract frames
            status = .extractingFrames
            statusText = "Extracting frames..."
            progress = 0

            let frames = try await FrameExtractor.extractFrames(
                from: videoURL,
                targetSize: captureSize
            ) { p in
                Task { @MainActor in
                    self.progress = Float(p.current) / Float(max(p.total, 1))
                    self.statusText = "Extracting frames: \(p.current)/\(p.total)"
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

            let glassShape = CaptureView.GlassShape.roundedRect(cornerRadius: 32)

            // Step 2: Capture displacement (once — static glass)
            status = .capturingDisplacement
            statusText = "Capturing displacement field..."
            progress = 0

            let uvImage = DisplacementExtractor.generateUVGradient(size: captureSize)
            let uvView = CaptureView(backgroundImage: uvImage, glassShape: glassShape, size: captureSize)
            guard let uvCapture = await viewCapture.capture(uvView, size: captureSize) else {
                status = .error
                statusText = "Failed to capture UV"
                return
            }
            let displacement = DisplacementExtractor.extractDisplacement(uvCapture: uvCapture, size: captureSize)
            try exporter.saveDisplacement(displacement, width: Int(captureSize.width), height: Int(captureSize.height))

            // Step 3: Capture specular (once — black background)
            status = .capturingSpecular
            statusText = "Capturing specular field..."

            let blackImage = UIImage.solidColor(.black, size: captureSize)
            let blackView = CaptureView(backgroundImage: blackImage, glassShape: glassShape, size: captureSize)
            guard let blackCapture = await viewCapture.capture(blackView, size: captureSize) else {
                status = .error
                statusText = "Failed to capture specular"
                return
            }
            let specular = DisplacementExtractor.extractSpecular(blackCapture: blackCapture, size: captureSize)
            try exporter.saveSpecular(specular, width: Int(captureSize.width), height: Int(captureSize.height))

            // Step 4: Save all input frames + capture composited
            status = .capturingFrames
            let total = frames.count

            for (i, frame) in frames.enumerated() {
                let inputImage = UIImage(cgImage: frame)

                // Save input frame
                try exporter.saveInput(frame, index: i)

                // Capture composited (glass over real frame)
                let frameView = CaptureView(backgroundImage: inputImage, glassShape: glassShape, size: captureSize)
                if let composited = await viewCapture.capture(frameView, size: captureSize) {
                    try exporter.saveComposited(composited, index: i)
                }

                progress = Float(i + 1) / Float(total)
                statusText = "Capturing frames: \(i + 1)/\(total)"
            }

            // Step 5: Save manifest
            status = .saving
            statusText = "Saving manifest..."
            try exporter.saveManifest(frameCount: frames.count, size: captureSize)

            status = .done
            statusText = "Done! \(frames.count) frames captured."
            datasetURL = exporter.shareURL

        } catch {
            status = .error
            statusText = "Error: \(error.localizedDescription)"
        }
    }
}

// Helper
extension UIImage {
    static func solidColor(_ color: UIColor, size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { ctx in
            color.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/CaptureEngine.swift
git commit -m "feat: add capture engine orchestrating full dataset pipeline"
```

---

### Task 8: Main UI (Video Picker + Progress + Export)

**Files:**
- Modify: `LiquidGlassCapture/LiquidGlassCapture/ContentView.swift`

- [ ] **Step 1: Implement full ContentView with video picker and progress**

```swift
import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var engine = CaptureEngine()
    @State private var selectedVideo: PhotosPickerItem?
    @State private var showShareSheet = false

    var body: some View {
        VStack(spacing: 24) {
            Text("Liquid Glass Capture")
                .font(.largeTitle.bold())

            statusSection

            if engine.status == .idle {
                PhotosPicker(selection: $selectedVideo, matching: .videos) {
                    Label("Select Video", systemImage: "video.badge.plus")
                        .font(.title3)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.blue)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding(.horizontal)
            }

            if engine.status == .done, engine.datasetURL != nil {
                Button {
                    showShareSheet = true
                } label: {
                    Label("Share Dataset (AirDrop)", systemImage: "square.and.arrow.up")
                        .font(.title3)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.green)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding(.horizontal)

                Button("Start Over") {
                    engine = CaptureEngine()
                    selectedVideo = nil
                }
                .padding()
            }
        }
        .padding()
        .onChange(of: selectedVideo) { _, newValue in
            guard let item = newValue else { return }
            Task { await loadAndProcess(item: item) }
        }
        .sheet(isPresented: $showShareSheet) {
            if let url = engine.datasetURL {
                ShareSheet(url: url)
            }
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        VStack(spacing: 8) {
            Text(engine.statusText)
                .font(.headline)
                .foregroundStyle(.secondary)

            if engine.status != .idle && engine.status != .done && engine.status != .error {
                ProgressView(value: engine.progress)
                    .padding(.horizontal)
                Text("\(Int(engine.progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private func loadAndProcess(item: PhotosPickerItem) async {
        guard let movie = try? await item.loadTransferable(type: VideoTransferable.self) else {
            engine.statusText = "Failed to load video"
            return
        }
        await engine.process(videoURL: movie.url)
    }
}

struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("mov")
            try FileManager.default.copyItem(at: received.file, to: tempURL)
            return Self(url: tempURL)
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
```

- [ ] **Step 2: Build and run on device**

Run: Xcode Build & Run
Expected: App shows video picker button. Select a video. Progress bar shows extraction → displacement capture → specular capture → frame-by-frame composited capture → done. Share button appears.

- [ ] **Step 3: Commit**

```bash
git add LiquidGlassCapture/LiquidGlassCapture/ContentView.swift
git commit -m "feat: add main UI with video picker, progress, and AirDrop export"
```

---

### Task 9: End-to-End Test + AirDrop Transfer

- [ ] **Step 1: Record or select a short test video (~10 seconds)**

Use a short video first to validate the full pipeline before processing the real 3-minute video.

- [ ] **Step 2: Run the app, select the test video, wait for processing**

Expected output in Documents/liquid_glass_dataset/:
- `input_00000.png` through `input_00299.png` (for a 10s/30fps video)
- `composited_00000.png` through `composited_00299.png`
- `displacement.bin` + `displacement_meta.json`
- `specular.bin` + `specular_meta.json`
- `manifest.json`

- [ ] **Step 3: Visually validate**

Open a few input/composited pairs side by side. The composited version should show visible glass distortion over the input. If the composited looks identical to the input, the glass effect is not being captured — go back to Task 4 and increase the sleep duration or try alternative capture approaches.

- [ ] **Step 4: Validate displacement field**

AirDrop the dataset to Mac. Run this Python script to visualize:

```python
import numpy as np
import matplotlib.pyplot as plt
import json

with open("displacement_meta.json") as f:
    meta = json.load(f)
w, h = meta["width"], meta["height"]

disp = np.fromfile("displacement.bin", dtype=np.float32).reshape(h, w, 2)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(disp[:,:,0], cmap="RdBu"); plt.title("dx"); plt.colorbar()
plt.subplot(1, 2, 2); plt.imshow(disp[:,:,1], cmap="RdBu"); plt.title("dy"); plt.colorbar()
plt.savefig("displacement_vis.png"); plt.show()
```

Expected: The displacement field should show a smooth, lens-like distortion pattern centered in the frame. Values should be near 0 at edges (no glass) and nonzero in the center (glass region).

- [ ] **Step 5: AirDrop the full dataset**

Process the full 3-minute video. AirDrop the entire `liquid_glass_dataset` folder to the training machine.

- [ ] **Step 6: Commit any final fixes**

```bash
git add -A
git commit -m "feat: complete dataset capture app, validated end-to-end"
```

---

## Performance Estimate

- Frame extraction: ~30 seconds for 5000 frames
- Displacement capture: ~1 second (single capture)
- Specular capture: ~1 second (single capture)
- Composited frame capture: ~50ms per frame × 5000 = ~4 minutes
- Saving PNGs: ~2 minutes
- **Total: ~7 minutes** on iPhone 16 Pro

## Dataset Size Estimate

- 5000 input PNGs (256×256): ~500 MB
- 5000 composited PNGs: ~500 MB
- displacement.bin: 512 KB
- specular.bin: 256 KB
- **Total: ~1 GB** — fits in a single AirDrop transfer
