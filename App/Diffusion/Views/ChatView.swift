import SwiftUI
import DiffusionGeneration
import DiffusionModel

public struct TokenState: Identifiable, Equatable, Sendable {
    public var id: Int
    public let token: String
    public let tokenId: Int
    public let isMasked: Bool
    public let confidence: Float
    public let isEdited: Bool
    public let isTransferred: Bool
}

public struct BlockState: Identifiable, Equatable, Sendable {
    public var id: Int
    public let text: String
    public let stepCount: Int
    public let meanConfidence: Float
}

public struct UIMessage: Identifiable, Equatable, Sendable {
    public let id: String
    public let role: String // "user" or "assistant"
    public var text: String
    public var blocks: [BlockState]
    public var configJson: String
    public var traceJsonRef: String?
    public let createdAt: Date
}

public class ChatViewModel: ObservableObject {
    @Published public var isGenerating = false
    @Published public var selectedBackend: ConnectionTarget = .local
    @Published public var selectedModel: String = "llada2-1-mini"
    @Published public var isDebugMode = false
    
    // Live unmasking canvas states for the active block
    @Published public var activeBlockIndex = 0
    @Published public var activeStepInBlock = 0
    @Published public var activeTokens: [TokenState] = []
    @Published public var activeBlockResultText: String = ""
    @Published public var activeBlockStepCount = 0
    @Published public var activeBlockMeanConfidence: Float = 0.0
    
    // Sparkline/debug telemetry
    @Published public var liveMeanConfidences: [Float] = []
    @Published public var liveStepsPerBlock: [Int] = []
    
    public var tokenizer: DiffusionTokenizer?
    
    public init() {}
    
    public func clearLiveTracing() {
        self.activeTokens = []
        self.activeBlockIndex = 0
        self.activeStepInBlock = 0
        self.activeBlockResultText = ""
        self.activeBlockStepCount = 0
        self.activeBlockMeanConfidence = 0.0
        self.liveMeanConfidences = []
        self.liveStepsPerBlock = []
    }
}

public struct ChatView: View {
    @StateObject private var vm = ChatViewModel()
    @ObservedObject private var store = ConversationStore.shared
    @ObservedObject private var tailscale = TailscaleDiscovery.shared
    
    @State private var promptText = ""
    @State private var showAutocomplete = false
    @State private var showBackendSettings = false
    @State private var manualHost = "100.100.100.100"
    @State private var manualPort = "8080"
    
    private let availableModels = ["llada2-1-mini", "sumi", "llada2-1-flash", "fast-dllm"]
    
    public init() {}
    
    public var body: some View {
        NavigationSplitView {
            // Sidebar: Conversations List
            VStack(spacing: 0) {
                HStack {
                    Button(action: createNewConversation) {
                        Label("New Chat", systemImage: "plus")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .font(.custom("Lexend Deca Regular", size: 12))

                    Button(action: { store.clearAll() }) {
                        Image(systemName: "trash")
                            .help("Clear history")
                    }
                    .buttonStyle(.bordered)
                    .foregroundColor(.red)
                }
                .padding(.horizontal)
                .frame(height: LayoutMetrics.columnHeaderHeight)

                Divider()

                List(store.conversations, selection: Binding(
                    get: { store.activeConversationId },
                    set: { if let id = $0 { store.selectConversation(id: id) } }
                )) { convo in
                    Text(convo.title)
                        .font(.custom("Lexend Deca Regular", size: 12))
                        .lineLimit(1)
                        .contextMenu {
                            Button("Delete", role: .destructive) {
                                store.deleteConversation(id: convo.id)
                            }
                        }
                }
            }
            .navigationTitle("Conversations")
            .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)

        } detail: {
            // Main Column
            VStack(spacing: 0) {
                // Header Panel
                headerPanel

                Divider()

                // Chat Canvas
                chatContentArea

                Divider()

                // Input and Autocomplete
                VStack(spacing: 0) {
                    if showAutocomplete {
                        autocompletePopover
                    }
                    inputArea
                }
            }
            .background(.ultraThinMaterial)
            .navigationTitle("")
            .navigationSplitViewColumnWidth(min: 600, ideal: 700)
        }
        .inspector(isPresented: .constant(true)) {
            // Right Sidebar: Flags Config
            FlagsInspectorView(isGenerating: vm.isGenerating)
                .inspectorColumnWidth(min: 320, ideal: 360, max: 440)
        }
        .onAppear {
            FontManager.registerCustomFonts()
            Task {
                await tailscale.refreshPeers()
            }
        }
    }
    
    // MARK: - Header
    
    private var headerPanel: some View {
        HStack(alignment: .center, spacing: 16) {
            // Backend Status Badge
            VStack(alignment: .leading, spacing: 2) {
                headerCaption("Backend")
                backendMenu
            }

            // Model Picker
            VStack(alignment: .leading, spacing: 2) {
                headerCaption("Model")
                Picker("", selection: $vm.selectedModel) {
                    ForEach(availableModels, id: \.self) { m in
                        Text(m)
                            .font(.custom("Illinois Mono", size: 11))
                            .tag(m)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 170)
            }

            Spacer(minLength: 16)

            // Pleasant vs Debug Mode Segmented Control
            VStack(alignment: .center, spacing: 2) {
                headerCaption("Mode")
                Picker("", selection: $vm.isDebugMode) {
                    Text("Pleasant").tag(false)
                    Text("Debug").tag(true)
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 160)
            }

            Button(action: {
                Task {
                    await tailscale.refreshPeers()
                }
            }) {
                Image(systemName: "arrow.clockwise")
                    .help("Refresh Tailscale Discovery")
            }
            .buttonStyle(.plain)
            .disabled(tailscale.isSearching)
        }
        .padding(.horizontal, 20)
        .frame(height: LayoutMetrics.columnHeaderHeight)
    }

    private func headerCaption(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.custom("Lexend Deca Regular", size: 9))
            .fontWeight(.semibold)
            .tracking(0.5)
            .foregroundColor(.secondary)
    }

    private var backendMenu: some View {
        Menu {
                Button("Local Inference Backend") {
                    vm.selectedBackend = .local
                }
                
                Divider()
                
                Text("Discovered Tailscale Peers:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                ForEach(tailscale.peers) { peer in
                    Button(action: {
                        vm.selectedBackend = .remote(host: peer.dnsName, port: 8080)
                    }) {
                        HStack {
                            Circle()
                                .fill(peer.isReachable ? Color.green : Color.orange)
                                .frame(width: 6, height: 6)
                            Text(peer.hostName)
                        }
                    }
                }
                
                Divider()
                
                Button("Manual Host / Port...") {
                    showBackendSettings = true
                }
            } label: {
                HStack(spacing: 6) {
                    Circle()
                        .fill(backendColor)
                        .frame(width: 8, height: 8)
                    Text(vm.selectedBackend.description)
                        .font(.custom("Lexend Deca Regular", size: 12))
                        .fontWeight(.semibold)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.secondary.opacity(0.12))
                .cornerRadius(20)
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .sheet(isPresented: $showBackendSettings) {
                VStack(spacing: 16) {
                    Text("Configure Custom Target")
                        .font(.custom("Lexend Deca Regular", size: 14))
                        .fontWeight(.bold)
                    
                    TextField("Host Name / IP", text: $manualHost)
                        .textFieldStyle(.roundedBorder)
                        .font(.custom("Illinois Mono", size: 12))
                    
                    TextField("Port", text: $manualPort)
                        .textFieldStyle(.roundedBorder)
                        .font(.custom("Illinois Mono", size: 12))
                    
                    HStack {
                        Button("Cancel") {
                            showBackendSettings = false
                        }
                        Spacer()
                        Button("Apply") {
                            if let p = Int(manualPort) {
                                vm.selectedBackend = .remote(host: manualHost, port: p)
                            }
                            showBackendSettings = false
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                .padding()
                .frame(width: 300)
            }
    }

    private var backendColor: Color {
        switch vm.selectedBackend {
        case .local:
            return .green
        case .remote:
            return .blue
        }
    }
    
    // MARK: - Chat Canvas
    
    private var chatContentArea: some View {
        ScrollView {
            ScrollViewReader { proxy in
                VStack(spacing: 24) {
                    ForEach(store.currentConversationTurns) { turn in
                        ChatTurnRow(turn: turn, isDebugMode: vm.isDebugMode)
                    }
                    
                    // Live generation target block rendering
                    if vm.isGenerating {
                        liveGenerationRow
                            .id("liveRow")
                    }
                }
                .padding()
                .onChange(of: store.currentConversationTurns) { _, _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: vm.isGenerating) { _, newValue in
                    if newValue {
                        scrollToBottom(proxy)
                    }
                }
            }
        }
    }
    
    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation {
            proxy.scrollTo("liveRow", anchor: .bottom)
        }
    }
    
    // MARK: - Live Generation Row
    
    private var liveGenerationRow: some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(Color.accentColor)
                .frame(width: 24, height: 24)
                .overlay(
                    Image(systemName: "cpu")
                        .font(.caption2)
                        .foregroundColor(.white)
                )
            
            VStack(alignment: .leading, spacing: 12) {
                Text("NeoDiffusion Assistant")
                    .font(.custom("Lexend Deca Regular", size: 12))
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)
                
                // Completed Blocks
                if !vm.activeBlockResultText.isEmpty {
                    Text(vm.activeBlockResultText)
                        .font(.custom("Lexend Deca Regular", size: 13))
                        .padding(.vertical, 4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                
                // Active Denoising Block Container (Pleasant/Debug)
                if !vm.activeTokens.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Denoising Active Block...")
                                .font(.custom("Lexend Deca Regular", size: 11))
                                .foregroundColor(.secondary)
                            Spacer()
                            if vm.isDebugMode {
                                Text("Step \(vm.activeStepInBlock) | Mean Conf: \(String(format: "%.2f", vm.activeBlockMeanConfidence))")
                                    .font(.custom("Illinois Mono", size: 9))
                                    .foregroundColor(.secondary)
                            }
                        }
                        
                        // Token Chips Flow Grid
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 45, maximum: 80))], alignment: .leading, spacing: 6) {
                            ForEach(vm.activeTokens) { tok in
                                TokenChipView(
                                    token: tok.token,
                                    isMasked: tok.isMasked,
                                    confidence: tok.confidence,
                                    isEdited: tok.isEdited,
                                    isTransferred: tok.isTransferred,
                                    isDebugMode: vm.isDebugMode
                                )
                            }
                        }
                        
                        // Sparkline telemetry in debug mode
                        if vm.isDebugMode && vm.liveMeanConfidences.count > 1 {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Live Confidence Sparkline")
                                    .font(.custom("Lexend Deca Regular", size: 8))
                                    .foregroundColor(.secondary)
                                SparklineView(data: vm.liveMeanConfidences)
                                    .frame(height: 18)
                                    .background(Color.primary.opacity(0.03))
                                    .cornerRadius(3)
                            }
                            .padding(.top, 4)
                        }
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.regularMaterial)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.accentColor.opacity(0.3), lineWidth: 1)
                    )
                }
            }
            Spacer()
        }
        .padding(.vertical, 8)
    }
    
    // MARK: - Input Area
    
    private var inputArea: some View {
        HStack(spacing: 12) {
            Button(action: { store.undoLastTurn() }) {
                Image(systemName: "arrow.uturn.backward")
                    .help("Undo last edit/turn")
            }
            .disabled(vm.isGenerating || store.currentConversationTurns.isEmpty)
            .buttonStyle(.plain)
            
            TextField("Type / for commands, or prompt here...", text: $promptText, onCommit: {
                submitPrompt()
            })
            .textFieldStyle(.roundedBorder)
            .font(.custom("Lexend Deca Regular", size: 12))
            .onChange(of: promptText) { _, newValue in
                showAutocomplete = newValue.hasPrefix("/")
            }
            
            Button(action: submitPrompt) {
                Image(systemName: "paperplane.fill")
                    .help("Send prompt")
            }
            .disabled(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || vm.isGenerating)
            .buttonStyle(.plain)
        }
        .padding()
    }
    
    // MARK: - Autocomplete Popover
    
    private var autocompletePopover: some View {
        VStack(alignment: .leading, spacing: 0) {
            let suggestions = autocompleteSuggestions(for: promptText)
            if !suggestions.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(suggestions, id: \.cmd) { s in
                            Button(action: {
                                promptText = s.cmd
                                showAutocomplete = false
                            }) {
                                HStack {
                                    Text(s.cmd)
                                        .font(.custom("Illinois Mono", size: 11))
                                        .foregroundColor(.accentColor)
                                        .fontWeight(.bold)
                                    Spacer()
                                    Text(s.desc)
                                        .font(.custom("Lexend Deca Regular", size: 11))
                                        .foregroundColor(.secondary)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            Divider()
                        }
                    }
                }
                .frame(maxHeight: 180)
                .background(Color(.windowBackgroundColor))
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.primary.opacity(0.12), lineWidth: 1)
                )
                .padding(.horizontal)
                .padding(.bottom, 4)
            }
        }
    }
    
    private struct Suggestion {
        let cmd: String
        let desc: String
    }
    
    private func autocompleteSuggestions(for text: String) -> [Suggestion] {
        guard text.hasPrefix("/") else { return [] }
        let clean = text.lowercased()
        
        let all = [
            Suggestion(cmd: "/q", desc: "Apply Quality Preset (τ_mask=0.7, τ_edit=0.5, nBuf=1)"),
            Suggestion(cmd: "/f", desc: "Apply Fast Preset (τ_mask=0.5, τ_edit=0.0)"),
            Suggestion(cmd: "/f nbuf=2 tauAdd=0.85 speculationK=4", desc: "Apply Fast Preset with custom overrides"),
            Suggestion(cmd: "/reset", desc: "Revert all settings to Settings.swift defaults"),
            Suggestion(cmd: "/model llada2-1-mini", desc: "Switch model to llada2-1-mini"),
            Suggestion(cmd: "/model sumi", desc: "Switch model to sumi")
        ]
        
        if clean == "/f " {
            return [
                Suggestion(cmd: "/f nbuf=", desc: "Buffer slot size (1 or 2)"),
                Suggestion(cmd: "/f tauAdd=", desc: "MultiBD activation threshold"),
                Suggestion(cmd: "/f speculationK=", desc: "Speculation batch size"),
                Suggestion(cmd: "/f dynamicTauAlpha=", desc: "Dynamic threshold adjustment coefficient"),
                Suggestion(cmd: "/f jotEnabled=", desc: "Toggle JOT token freezing")
            ]
        }
        
        return all.filter { $0.cmd.lowercased().hasPrefix(clean) }
    }
    
    // MARK: - Actions
    
    private func createNewConversation() {
        _ = store.createConversation(title: "New Conversation")
    }
    
    private func submitPrompt() {
        let text = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        // Clear input
        promptText = ""
        showAutocomplete = false
        
        // Parse slash commands
        if text.hasPrefix("/") {
            let handled = parseAndApplySlashCommand(text)
            if handled {
                print("Applied command: \(text)")
                return
            }
        }
        
        // Save user turn
        store.addTurn(role: "user", text: text, config: Settings.shared.config)
        
        // Trigger inference
        runAssistantInference(prompt: text)
    }
    
    private func runAssistantInference(prompt: String) {
        vm.isGenerating = true
        vm.clearLiveTracing()
        
        let backend: InferenceBackend
        switch vm.selectedBackend {
        case .local:
            backend = LocalInferenceBackend()
        case .remote(let host, let port):
            backend = RemoteInferenceBackend(host: host, port: port)
        }
        
        let config = Settings.shared.config
        
        Task {
            do {
                let output = try await backend.generate(
                    prompt: prompt,
                    config: config,
                    onTrace: { trace in
                        // Live unmasking step trace callback
                        if let tok = (backend as? LocalInferenceBackend) {
                            // We need to resolve tokens
                            // Access is safe within dispatch queue
                            let B = config.blockLength
                            var states: [TokenState] = []
                            for i in 0 ..< B {
                                let tokenId = trace.argmaxToken[i]
                                // Map tokenId to decoded string
                                let tokenStr: String
                                if tokenId == (tok.tokenizer?.maskId ?? 156895) {
                                    tokenStr = "░░"
                                } else {
                                    tokenStr = tok.tokenizer?.decode(tokens: [tokenId]) ?? "?"
                                }
                                states.append(TokenState(
                                    id: i,
                                    token: tokenStr,
                                    tokenId: tokenId,
                                    isMasked: trace.masked[i],
                                    confidence: trace.confidence[i],
                                    isEdited: trace.edited[i],
                                    isTransferred: trace.transferred[i]
                                ))
                            }
                            
                            // Calculate mean confidence over unmasked tokens
                            let activeConfidences = trace.confidence
                            let avg = activeConfidences.reduce(0.0, +) / Float(B)
                            
                            DispatchQueue.main.async {
                                vm.activeTokens = states
                                vm.activeStepInBlock = trace.stepInBlock
                                vm.activeBlockIndex = trace.blockIndex
                                vm.activeBlockMeanConfidence = avg
                                vm.liveMeanConfidences.append(avg)
                            }
                        }
                    },
                    onTokenStream: { blockIds in
                        // Block committed callback
                        DispatchQueue.main.async {
                            // We can render incremental progress
                        }
                    },
                    onTextStream: { deltaText in
                        // Incremental text chunks streamed back
                        DispatchQueue.main.async {
                            vm.activeBlockResultText += deltaText
                        }
                    }
                )
                
                // Generation completed
                DispatchQueue.main.async {
                    vm.isGenerating = false
                    // Save assistant response
                    store.addTurn(role: "assistant", text: vm.activeBlockResultText, config: config)
                    vm.clearLiveTracing()
                }
            } catch {
                print("Inference error: \(error)")
                DispatchQueue.main.async {
                    vm.isGenerating = false
                    store.addTurn(role: "assistant", text: "Error: \(error.localizedDescription)", config: config)
                    vm.clearLiveTracing()
                }
            }
        }
    }
    
    // MARK: - Slash Command Handler
    
    private func parseAndApplySlashCommand(_ input: String) -> Bool {
        let parts = input.dropFirst().split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        guard !parts.isEmpty else { return false }
        let cmd = parts[0].lowercased()
        
        switch cmd {
        case "q":
            Settings.shared.config.threshold = 0.7
            Settings.shared.config.editingThreshold = 0.5
            Settings.shared.config.nBuf = 1
            Settings.shared.config.speculation = "none"
            Settings.shared.config.dynamicTauAlpha = 0.0
            Settings.shared.saveConfigJSON()
            return true
            
        case "reset":
            Settings.shared.resetToDefaults()
            return true
            
        case "model":
            if parts.count > 1 {
                let modelName = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
                vm.selectedModel = modelName
            }
            return true
            
        case "f":
            // Fast mode defaults
            Settings.shared.config.threshold = 0.5
            Settings.shared.config.editingThreshold = 0.0
            Settings.shared.config.nBuf = 1
            Settings.shared.config.speculation = "none"
            
            if parts.count > 1 {
                let paramsStr = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
                let pairs = paramsStr.split(separator: " ")
                for pair in pairs {
                    let kv = pair.split(separator: "=", maxSplits: 1)
                    if kv.count == 2 {
                        let key = String(kv[0]).trimmingCharacters(in: .whitespacesAndNewlines)
                        let val = String(kv[1]).trimmingCharacters(in: .whitespacesAndNewlines)
                        applyParameter(key: key, value: val)
                    }
                }
            }
            Settings.shared.saveConfigJSON()
            return true
            
        default:
            return false
        }
    }
    
    private func applyParameter(key: String, value: String) {
        switch key {
        case "threshold", "tauMask":
            if let f = Float(value) { Settings.shared.config.threshold = f }
        case "editingThreshold", "tauEdit":
            if let f = Float(value) { Settings.shared.config.editingThreshold = f }
        case "maxPostSteps":
            if let i = Int(value) { Settings.shared.config.maxPostSteps = i }
        case "numToTransfer":
            if let i = Int(value) { Settings.shared.config.numToTransfer = i }
        case "eosEarlyStop":
            if let b = Bool(value) { Settings.shared.config.eosEarlyStop = b }
        case "temperature":
            if let f = Float(value) { Settings.shared.config.temperature = f }
        case "blockLength":
            if let i = Int(value) { Settings.shared.config.blockLength = i }
        case "genLength":
            if let i = Int(value) { Settings.shared.config.genLength = i }
        case "nBuf":
            if let i = Int(value) { Settings.shared.config.nBuf = i }
        case "tauAdd":
            if let f = Float(value) { Settings.shared.config.tauAdd = f }
        case "tauSemi":
            if let f = Float(value) { Settings.shared.config.tauSemi = f }
        case "speculation":
            Settings.shared.config.speculation = value
        case "speculationK":
            if let i = Int(value) { Settings.shared.config.speculationK = i }
        case "tauSpan":
            if let i = Int(value) { Settings.shared.config.tauSpan = i }
        case "dynamicTauAlpha":
            if let f = Float(value) { Settings.shared.config.dynamicTauAlpha = f }
        case "eosEarlyExit":
            if let b = Bool(value) { Settings.shared.config.eosEarlyExit = b }
        case "jotEnabled":
            if let b = Bool(value) { Settings.shared.config.jotEnabled = b }
        case "jotK":
            if let i = Int(value) { Settings.shared.config.jotK = i }
        case "jotThreshold":
            if let f = Float(value) { Settings.shared.config.jotThreshold = f }
        case "jotFaithful":
            if let b = Bool(value) { Settings.shared.config.jotFaithful = b }
        case "moeCapacityRatio":
            if let f = Float(value) { Settings.shared.config.moeCapacityRatio = f }
        default:
            print("Unknown parameter: \(key)")
        }
    }
}

// MARK: - Row Rendering

struct ChatTurnRow: View {
    let turn: ChatTurn
    let isDebugMode: Bool
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if turn.role == "user" {
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("User")
                        .font(.custom("Lexend Deca Regular", size: 11))
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)
                    
                    Text(turn.text)
                        .font(.custom("Lexend Deca Regular", size: 13))
                        .padding(10)
                        .background(Color.accentColor.opacity(0.12))
                        .cornerRadius(8)
                }
                
                Circle()
                    .fill(Color.accentColor.opacity(0.2))
                    .frame(width: 24, height: 24)
                    .overlay(
                        Image(systemName: "person.fill")
                            .font(.caption2)
                            .foregroundColor(.accentColor)
                    )
            } else {
                Circle()
                    .fill(Color.primary.opacity(0.08))
                    .frame(width: 24, height: 24)
                    .overlay(
                        Image(systemName: "cpu")
                            .font(.caption2)
                    )
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("NeoDiffusion Assistant")
                        .font(.custom("Lexend Deca Regular", size: 11))
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)
                    
                    Text(turn.text)
                        .font(.custom("Lexend Deca Regular", size: 13))
                        .padding(10)
                        .background(Color.primary.opacity(0.04))
                        .cornerRadius(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    
                    if isDebugMode {
                        HStack(spacing: 8) {
                            Text("Config:")
                                .font(.custom("Lexend Deca Regular", size: 9))
                                .foregroundColor(.secondary)
                            Text(turn.configJson)
                                .font(.custom("Illinois Mono", size: 8))
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        .padding(.top, 2)
                    }
                }
                
                Spacer()
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Sparkline View

struct SparklineView: View {
    let data: [Float]
    
    var body: some View {
        GeometryReader { geo in
            Path { path in
                guard data.count > 1 else { return }
                
                let width = geo.size.width
                let height = geo.size.height
                let stepX = width / CGFloat(data.count - 1)
                
                let minVal = data.min() ?? 0.0
                let maxVal = data.max() ?? 1.0
                let valRange = max(maxVal - minVal, 0.01)
                
                let firstY = height - CGFloat((data[0] - minVal) / valRange) * height
                path.move(to: CGPoint(x: 0, y: firstY))
                
                for i in 1 ..< data.count {
                    let x = CGFloat(i) * stepX
                    let y = height - CGFloat((data[i] - minVal) / valRange) * height
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            .stroke(Color.accentColor, lineWidth: 1.5)
        }
    }
}
