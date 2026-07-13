import SwiftUI

public struct TokenChipView: View {
    let token: String
    let isMasked: Bool
    let confidence: Float
    let isEdited: Bool
    let isTransferred: Bool
    let isDebugMode: Bool
    
    @State private var borderFlash = false
    
    public init(
        token: String,
        isMasked: Bool,
        confidence: Float,
        isEdited: Bool,
        isTransferred: Bool,
        isDebugMode: Bool
    ) {
        self.token = token
        self.isMasked = isMasked
        self.confidence = confidence
        self.isEdited = isEdited
        self.isTransferred = isTransferred
        self.isDebugMode = isDebugMode
    }
    
    public var body: some View {
        VStack(spacing: 2) {
            Text(isMasked ? "░░" : token)
                .font(.custom("Illinois Mono", size: 11))
                .foregroundColor(isMasked ? .secondary.opacity(0.6) : .primary)
                .padding(.horizontal, 5)
                .padding(.vertical, 3)
                .background(isMasked ? Color.blue.opacity(0.12) : (isDebugMode ? confidenceColor(confidence).opacity(0.22) : Color.primary.opacity(0.05)))
                .background(isMasked ? .thinMaterial : .regularMaterial)
                .cornerRadius(4)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(borderStrokeColor, lineWidth: borderStrokeWidth)
                )
                .help(isDebugMode ? String(format: "Confidence: %.2f", confidence) : "")
                .onChange(of: isEdited) { oldValue, newValue in
                    if newValue {
                        triggerFlash()
                    }
                }
                .onAppear {
                    if isEdited {
                        triggerFlash()
                    }
                }
            
            if isDebugMode && !isMasked {
                Text(String(format: "%.2f", confidence))
                    .font(.custom("Illinois Mono", size: 8))
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private func triggerFlash() {
        withAnimation(.easeInOut(duration: 0.15).repeatCount(4, autoreverses: true)) {
            borderFlash = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            borderFlash = false
        }
    }
    
    private var borderStrokeColor: Color {
        if isDebugMode {
            if borderFlash {
                return .red
            } else if isEdited {
                return .orange
            }
        }
        return .clear
    }
    
    private var borderStrokeWidth: CGFloat {
        if isDebugMode && (isEdited || borderFlash) {
            return 1.5
        }
        return 0
    }
    
    private func confidenceColor(_ conf: Float) -> Color {
        if conf >= 0.8 {
            return .green
        } else if conf >= 0.5 {
            return .yellow
        } else {
            return .orange
        }
    }
}
