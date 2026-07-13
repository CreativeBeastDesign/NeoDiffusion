import SwiftUI

/// Shared layout constants that must stay identical across `ChatView` and
/// `FlagsInspectorView` so the sidebar, detail, and inspector top bars — and
/// the dividers directly beneath them — align across all three
/// `NavigationSplitView` columns.
public enum LayoutMetrics {
    public static let columnHeaderHeight: CGFloat = 60
}
