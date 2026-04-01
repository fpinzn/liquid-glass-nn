import SwiftUI

struct CaptureView: View {
    let backgroundImage: UIImage
    let size: CGSize
    let glassDiameter: CGFloat

    init(backgroundImage: UIImage, size: CGSize, glassFraction: CGFloat = 0.8) {
        self.backgroundImage = backgroundImage
        self.size = size
        self.glassDiameter = min(size.width, size.height) * glassFraction
    }

    var body: some View {
        ZStack {
            Image(uiImage: backgroundImage)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: size.width, height: size.height)
                .clipped()

            Color.clear
                .frame(width: glassDiameter, height: glassDiameter)
                .glassEffect(.regular, in: .circle)
        }
        .frame(width: size.width, height: size.height)
    }
}
