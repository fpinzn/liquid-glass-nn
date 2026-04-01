import SwiftUI
import UIKit

class ViewCapture {
    private var captureWindow: UIWindow?
    private var hostingController: UIHostingController<AnyView>?

    func setup(size: CGSize) {
        guard let scene = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first else { return }

        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(origin: .zero, size: size)
        window.windowLevel = .normal - 1
        window.isHidden = false
        window.alpha = 0.01

        let hc = UIHostingController(rootView: AnyView(EmptyView()))
        hc.view.frame = CGRect(origin: .zero, size: size)
        window.rootViewController = hc
        window.makeKeyAndVisible()

        self.captureWindow = window
        self.hostingController = hc
    }

    @MainActor
    func capture<V: View>(_ view: V, size: CGSize) async -> UIImage? {
        guard let hc = hostingController else { return nil }

        hc.rootView = AnyView(view)
        hc.view.frame = CGRect(origin: .zero, size: size)
        hc.view.setNeedsLayout()
        hc.view.layoutIfNeeded()

        // Glass effect needs compositor time to render
        try? await Task.sleep(for: .milliseconds(100))

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
