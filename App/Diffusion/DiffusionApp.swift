import SwiftUI
import DiffusionGeneration
import DiffusionModel

@main
struct DiffusionApp: App {
    init() {
        FontManager.registerCustomFonts()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1180, minHeight: 720)
        }
        .defaultSize(width: 1360, height: 820)
    }
}
