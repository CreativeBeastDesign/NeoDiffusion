import Foundation

public struct TailscalePeer: Identifiable, Codable, Equatable, Sendable {
    public var id: String { dnsName }
    public let hostName: String
    public let dnsName: String
    public let ip: String
    public var isOnline: Bool
    public var isReachable: Bool
    
    public init(hostName: String, dnsName: String, ip: String, isOnline: Bool, isReachable: Bool = false) {
        self.hostName = hostName
        self.dnsName = dnsName
        self.ip = ip
        self.isOnline = isOnline
        self.isReachable = isReachable
    }
}

public class TailscaleDiscovery: ObservableObject {
    @Published public var peers: [TailscalePeer] = []
    @Published public var isSearching = false
    @Published public var errorWarning: String?
    
    public static let shared = TailscaleDiscovery()
    
    private init() {}
    
    public func refreshPeers() async {
        DispatchQueue.main.async {
            self.isSearching = true
            self.errorWarning = nil
        }
        
        let paths = [
            "/opt/homebrew/bin/tailscale",
            "/usr/local/bin/tailscale",
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            "/Applications/Tailscale.app/Contents/Resources/bin/tailscale"
        ]
        
        var jsonOutput: String? = nil
        var lastError: Error? = nil
        
        for path in paths {
            if FileManager.default.fileExists(atPath: path) {
                do {
                    jsonOutput = try runProcess(executable: path, arguments: ["status", "--json"])
                    if jsonOutput != nil { break }
                } catch {
                    lastError = error
                }
            }
        }
        
        if jsonOutput == nil {
            // Log sandbox warnings or CLI missing details
            DispatchQueue.main.async {
                self.isSearching = false
                if let err = lastError {
                    self.errorWarning = "Tailscale query failed: \(err.localizedDescription) (Sandbox restriction likely)."
                } else {
                    self.errorWarning = "Tailscale CLI not found in typical locations. Sandbox active or Tailscale not installed."
                }
                self.peers = []
            }
            return
        }
        
        guard let data = jsonOutput?.data(using: .utf8) else {
            DispatchQueue.main.async {
                self.isSearching = false
                self.peers = []
            }
            return
        }
        
        do {
            let status = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            var discovered: [TailscalePeer] = []
            
            if let peerMap = status?["Peer"] as? [String: [String: Any]] {
                for (ip, info) in peerMap {
                    guard let hostName = info["HostName"] as? String,
                          let dnsName = info["DNSName"] as? String,
                          let isOnline = info["Online"] as? Bool else { continue }
                    
                    // Simple cleaning of DNS name (Tailscale adds trailing dots sometimes)
                    var cleanDNS = dnsName
                    if cleanDNS.hasSuffix(".") {
                        cleanDNS = String(cleanDNS.dropLast())
                    }
                    
                    discovered.append(TailscalePeer(
                        hostName: hostName,
                        dnsName: cleanDNS,
                        ip: ip,
                        isOnline: isOnline,
                        isReachable: false
                    ))
                }
            }
            
            // Sort peers alphabetically by hostName
            discovered.sort { $0.hostName.localizedCompare($1.hostName) == .orderedAscending }
            
            DispatchQueue.main.async {
                self.peers = discovered
            }
            
            // Check reachability for online peers concurrently
            await withTaskGroup(of: (String, Bool).self) { group in
                for peer in discovered where peer.isOnline {
                    group.addTask {
                        let reachable = await self.pingPeer(host: peer.dnsName)
                        return (peer.dnsName, reachable)
                    }
                }
                
                for await (dnsName, reachable) in group {
                    DispatchQueue.main.async {
                        if let index = self.peers.firstIndex(where: { $0.dnsName == dnsName }) {
                            var updated = self.peers[index]
                            updated.isReachable = reachable
                            self.peers[index] = updated
                        }
                    }
                }
            }
            
        } catch {
            print("[TailscaleDiscovery] Failed to parse JSON: \(error)")
        }
        
        DispatchQueue.main.async {
            self.isSearching = false
        }
    }
    
    private func runProcess(executable: String, arguments: [String]) throws -> String? {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: executable)
        task.arguments = arguments
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        task.standardOutput = outputPipe
        task.standardError = errorPipe
        
        try task.run()
        task.waitUntilExit()
        
        let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }
    
    private func pingPeer(host: String) async -> Bool {
        // Simple reachability check by trying to connect to port 8080 (the default server port)
        // or check simple URL response
        guard let url = URL(string: "http://\(host):8080/health") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0 // Quick timeout
        request.httpMethod = "GET"
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpRes = response as? HTTPURLResponse, httpRes.statusCode == 200 {
                return true
            }
        } catch {}
        
        return false
    }
}
