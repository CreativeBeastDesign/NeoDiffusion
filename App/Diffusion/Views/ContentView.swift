import SwiftUI
import DiffusionModel
import DiffusionGeneration

struct ContentView: View {
    @State private var isRunning = false
    @State private var progress = 0.0
    @State private var logText = "Ready to generate.\n"
    @State private var modelLoaded = false
    
    var body: some View {
        VStack(spacing: 24) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("NeoDiffusion Engine")
                        .font(.system(.title, design: .rounded))
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    Text("Experimental Diffusion Language Model Inference")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                // Hardware Status Badge
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                    Text("Apple Silicon GPU Active")
                        .font(.caption2)
                        .fontWeight(.semibold)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.green.opacity(0.15))
                .cornerRadius(20)
            }
            .padding(.horizontal)
            .padding(.top)
            
            // Console Output Area
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    Text(logText)
                        .font(.system(.body, design: .monospaced))
                        .foregroundColor(.green)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .multilineTextAlignment(.leading)
                }
                .padding()
            }
            .frame(height: 240)
            .background(Color(.windowBackgroundColor).opacity(0.5))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.primary.opacity(0.1), lineWidth: 1)
            )
            .padding(.horizontal)
            
            // Progress Section
            if isRunning {
                VStack(spacing: 8) {
                    ProgressView(value: progress)
                        .progressViewStyle(LinearProgressViewStyle(tint: .accentColor))
                    Text("Denoising latent space: \(Int(progress * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
            }
            
            // Action Controls
            HStack(spacing: 16) {
                Button(action: loadModel) {
                    Label("Load Config", systemImage: "cpu")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.bordered)
                .disabled(isRunning || modelLoaded)
                
                Button(action: runDenoising) {
                    Label("Run Denoising", systemImage: "play.fill")
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || !modelLoaded)
            }
            .padding(.horizontal)
            .padding(.bottom)
        }
        .frame(minWidth: 500, minHeight: 400)
    }
    
    private func loadModel() {
        logText += "Initializing MLX configuration...\n"
        let config = DiffusionModelConfig(vocabSize: 32000, hiddenDim: 512, numLayers: 4)
        let _ = DiffusionModel(config: config)
        logText += "Model architecture configured (Hidden Dim: 512, Layers: 4).\n"
        modelLoaded = true
    }
    
    private func runDenoising() {
        guard modelLoaded else { return }
        isRunning = true
        progress = 0.0
        logText += "Running denoising loop scheduler...\n"
        
        let config = DiffusionModelConfig(vocabSize: 32000, hiddenDim: 512, numLayers: 4)
        let model = DiffusionModel(config: config)
        let generator = DiffusionGeneration(model: model)
        
        let steps = 10
        let _ = generator.generate(steps: steps) { step, latent in
            let stepPercent = Double(step + 1) / Double(steps)
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(step) * 0.1) {
                self.progress = stepPercent
                self.logText += " -> Step \(step + 1)/\(steps) complete (Latent shape: \(latent.shape))\n"
                
                if step + 1 == steps {
                    self.isRunning = false
                    self.logText += "Generation finished successfully!\n"
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
