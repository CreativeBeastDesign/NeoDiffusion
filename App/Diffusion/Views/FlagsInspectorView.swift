import SwiftUI
import DiffusionGeneration

public struct FlagsInspectorView: View {
    @ObservedObject var settings = Settings.shared
    let isGenerating: Bool

    @State private var selectedStability: Stability = .stable

    public init(isGenerating: Bool) {
        self.isGenerating = isGenerating
    }

    public var body: some View {
        VStack(spacing: 0) {
            Picker("Settings Tier", selection: $selectedStability) {
                Text("Stable").tag(Stability.stable)
                Text("Experimental").tag(Stability.experimental)
            }
            .pickerStyle(.segmented)
            .font(.custom("Lexend Deca Regular", size: 12))
            .padding(.horizontal, 16)
            .frame(height: LayoutMetrics.columnHeaderHeight)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if selectedStability == .stable {
                        stableSection
                    } else {
                        experimentalSection
                    }
                }
                .padding(16)
            }
            .disabled(isGenerating)
            .opacity(isGenerating ? 0.5 : 1.0)

            Divider()

            // Footer with Reset
            HStack(spacing: 12) {
                Button("Revert to Defaults", role: .destructive) {
                    settings.resetToDefaults()
                }
                .disabled(isGenerating)

                Spacer()

                Button("Save Config") {
                    settings.saveConfigJSON()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating)
            }
            .font(.custom("Lexend Deca Regular", size: 12))
            .padding(16)
        }
        .frame(minWidth: 320)
    }

    private var stableSection: some View {
        SectionCard(title: "Core Generation Settings") {
            SliderRow(label: "τ_mask Threshold", value: $settings.config.threshold, range: 0.0...1.0)
            SliderRow(label: "τ_edit Threshold", value: $settings.config.editingThreshold, range: 0.0...1.0)
            StepperRow(label: "Max Post Steps", value: $settings.config.maxPostSteps, range: 0...64)
            StepperRow(label: "Num to Transfer", value: $settings.config.numToTransfer, range: 1...32)
            ToggleRow(label: "EOS Early Stop", isOn: $settings.config.eosEarlyStop)
            SliderRow(label: "Temperature", value: $settings.config.temperature, range: 0.0...2.0)
            StepperRow(label: "Block Length", value: $settings.config.blockLength, range: 8...256, step: 8)
            StepperRow(label: "Gen Length", value: $settings.config.genLength, range: 32...1024, step: 32)
        }
    }

    private var experimentalSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            SectionCard(title: "MultiBD Parallelism") {
                StepperRow(label: "nBuf Slots", value: $settings.config.nBuf, range: 1...2)

                if settings.config.nBuf > 1 {
                    SliderRow(label: "τ_add Activation", value: $settings.config.tauAdd, range: 0.0...2.0)
                    SliderRow(label: "τ_semi Completion", value: $settings.config.tauSemi, range: 0.0...1.0)
                }
            }

            SectionCard(title: "Speculative Verification") {
                MenuPickerRow(label: "Speculation Mode", selection: $settings.config.speculation, options: [
                    ("None", "none"),
                    ("S2D2", "s2d2")
                ])

                if settings.config.speculation != "none" {
                    StepperRow(label: "Speculation K", value: $settings.config.speculationK, range: 1...8)
                    StepperRow(label: "τ_span Speculation", value: $settings.config.tauSpan, range: 1...16)
                }
            }

            SectionCard(title: "Dynamic Calibration") {
                SliderRow(label: "Dynamic τ_mask Alpha", value: $settings.config.dynamicTauAlpha, range: 0.0...1.0)
                ToggleRow(label: "EOS Early Exit", isOn: $settings.config.eosEarlyExit)
            }

            SectionCard(title: "JOT (Just-on-Time) Optimizations") {
                ToggleRow(label: "JOT Token Freezing", isOn: $settings.config.jotEnabled)

                if settings.config.jotEnabled {
                    StepperRow(label: "JOT K-steps", value: $settings.config.jotK, range: 1...10)
                    SliderRow(label: "JOT Threshold", value: $settings.config.jotThreshold, range: 0.0...1.0)
                    ToggleRow(label: "JOT Faithful KV Hold", isOn: $settings.config.jotFaithful)
                    ToggleRow(label: "Sub-block Commit", isOn: $settings.config.subBlockCommit)

                    if settings.config.subBlockCommit {
                        StepperRow(label: "Min Prefix to Commit", value: $settings.config.subBlockMinPrefix, range: 1...32)
                    }

                    SliderRow(label: "MoE Capacity Ratio", value: $settings.config.moeCapacityRatio, range: 0.0...1.0)
                }
            }

            SectionCard(title: "FlashBlock Metal Acceleration") {
                ToggleRow(label: "FlashBlock Cache", isOn: $settings.config.flashBlockEnabled)

                if settings.config.flashBlockEnabled {
                    StepperRow(label: "Dirty Token Threshold", value: $settings.config.flashBlockTau, range: 1...16)
                }
            }

            SectionCard(title: "In-Place CoT (ICE)") {
                ToggleRow(label: "ICE CoT Enabled", isOn: $settings.config.iceEnabled)

                if settings.config.iceEnabled {
                    SliderRow(label: "ICE Early Exit Tau", value: $settings.config.iceTau, range: 0.0...1.0)
                    StepperRow(label: "Reasoning Steps (Nt)", value: $settings.config.iceNt, range: 1...10)
                    StepperRow(label: "Thinking Length", value: $settings.config.iceThinkingLength, range: 8...256, step: 8)
                }
            }

            SectionCard(title: "Credit Decoding") {
                ToggleRow(label: "Credit Decoding", isOn: $settings.config.creditDecodingEnabled)

                if settings.config.creditDecodingEnabled {
                    SliderRow(label: "Credit Alpha", value: $settings.config.creditAlpha, range: 0.0...2.0)
                    SliderRow(label: "Credit Beta", value: $settings.config.creditBeta, range: 0.0...1.0)
                    SliderRow(label: "Credit Gamma", value: $settings.config.creditGamma, range: 0.0...1.0)
                }
            }

            SectionCard(title: "Temporal Self-Consistency (TSCV)") {
                ToggleRow(label: "Temporal Voting", isOn: $settings.config.temporalVotingEnabled)

                if settings.config.temporalVotingEnabled {
                    SliderRow(label: "Voting Alpha (Decay)", value: $settings.config.temporalVotingAlpha, range: 0.0...2.0)
                    SliderRow(label: "Voting Cutoff", value: $settings.config.temporalVotingCutoff, range: 0.0...1.0)
                }
            }
        }
    }
}

// MARK: - Reusable row components
//
// Form's automatic label-column layout mis-measures composite label views that
// contain an internal Spacer (label + value pairs), clipping text from the
// leading edge. These rows lay out label/value explicitly instead.

private struct SectionCard<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title.uppercased())
                .font(.custom("Lexend Deca Regular", size: 11))
                .fontWeight(.semibold)
                .tracking(0.6)
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 14) {
                content
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }
}

private struct SliderRow: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(label)
                    .font(.custom("Lexend Deca Regular", size: 12))
                Spacer(minLength: 8)
                Text(String(format: "%.2f", value))
                    .font(.custom("Illinois Mono", size: 11))
                    .foregroundColor(.secondary)
                    .frame(minWidth: 36, alignment: .trailing)
            }
            Slider(value: $value, in: range)
                .controlSize(.small)
        }
    }
}

private struct StepperRow: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    var step: Int = 1

    var body: some View {
        HStack {
            Text(label)
                .font(.custom("Lexend Deca Regular", size: 12))
            Spacer(minLength: 8)
            Text("\(value)")
                .font(.custom("Illinois Mono", size: 11))
                .foregroundColor(.secondary)
                .frame(minWidth: 30, alignment: .trailing)
            Stepper("", value: $value, in: range, step: step)
                .labelsHidden()
        }
    }
}

private struct ToggleRow: View {
    let label: String
    @Binding var isOn: Bool

    var body: some View {
        Toggle(isOn: $isOn) {
            Text(label)
                .font(.custom("Lexend Deca Regular", size: 12))
        }
        .toggleStyle(.switch)
        .controlSize(.small)
    }
}

private struct MenuPickerRow: View {
    let label: String
    @Binding var selection: String
    let options: [(String, String)]

    var body: some View {
        HStack {
            Text(label)
                .font(.custom("Lexend Deca Regular", size: 12))
            Spacer(minLength: 8)
            Picker("", selection: $selection) {
                ForEach(options, id: \.1) { option in
                    Text(option.0)
                        .font(.custom("Lexend Deca Regular", size: 12))
                        .tag(option.1)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: 140)
        }
    }
}
