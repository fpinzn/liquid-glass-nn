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
