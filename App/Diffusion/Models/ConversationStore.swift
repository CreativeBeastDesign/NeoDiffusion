import Foundation
import Libsql
import Combine

public struct Conversation: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public var title: String
    public let createdAt: Date
    
    public init(id: String = UUID().uuidString, title: String, createdAt: Date = Date()) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
    }
}

public struct ChatTurn: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let conversationId: String
    public let role: String // "user" or "assistant"
    public var text: String
    public var configJson: String
    public var traceJsonRef: String?
    public let createdAt: Date
    
    public init(
        id: String = UUID().uuidString,
        conversationId: String,
        role: String,
        text: String,
        configJson: String,
        traceJsonRef: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.conversationId = conversationId
        self.role = role
        self.text = text
        self.configJson = configJson
        self.traceJsonRef = traceJsonRef
        self.createdAt = createdAt
    }
}

public class ConversationStore: ObservableObject {
    @Published public var conversations: [Conversation] = []
    @Published public var currentConversationTurns: [ChatTurn] = []
    @Published public var activeConversationId: String?
    
    private var db: Database?
    private var conn: Connection?
    
    public static let shared = ConversationStore()
    
    private init() {
        setupDatabase()
        loadConversations()
    }
    
    private var databaseURL: URL {
        let paths = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        let appSupportDir = paths[0].appendingPathComponent("NeoDiffusion", isDirectory: true)
        try? FileManager.default.createDirectory(at: appSupportDir, withIntermediateDirectories: true)
        return appSupportDir.appendingPathComponent("conversations.db")
    }
    
    private func setupDatabase() {
        do {
            let path = databaseURL.path
            print("[ConversationStore] Opening database at \(path)")
            let database = try Database(path)
            let connection = try database.connect()
            
            // Create tables
            try connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                createdAt DOUBLE NOT NULL
            );
            """)
            
            try connection.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                conversationId TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                config_json TEXT NOT NULL,
                trace_json_ref TEXT,
                createdAt DOUBLE NOT NULL,
                FOREIGN KEY(conversationId) REFERENCES conversations(id) ON DELETE CASCADE
            );
            """)
            
            self.db = database
            self.conn = connection
        } catch {
            print("[ConversationStore] Failed to setup database: \(error)")
        }
    }
    
    public func loadConversations() {
        guard let conn = self.conn else { return }
        
        do {
            let rows = try conn.query("SELECT id, title, createdAt FROM conversations ORDER BY createdAt DESC")
            var loaded: [Conversation] = []
            for row in rows {
                do {
                    let id = try row.getString(0)
                    let title = try row.getString(1)
                    let createdAtVal = try row.getDouble(2)
                    loaded.append(Conversation(id: id, title: title, createdAt: Date(timeIntervalSince1970: createdAtVal)))
                } catch {
                    print("[ConversationStore] Row mapping error: \(error)")
                    continue
                }
            }
            
            DispatchQueue.main.async {
                self.conversations = loaded
                if self.activeConversationId == nil, let first = loaded.first {
                    self.selectConversation(id: first.id)
                }
            }
        } catch {
            print("[ConversationStore] Failed to load conversations: \(error)")
        }
    }
    
    public func selectConversation(id: String) {
        self.activeConversationId = id
        loadTurns(for: id)
    }
    
    public func createConversation(title: String) -> String {
        guard let conn = self.conn else { return UUID().uuidString }
        
        let id = UUID().uuidString
        let now = Date().timeIntervalSince1970
        do {
            try conn.execute("INSERT INTO conversations (id, title, createdAt) VALUES (?, ?, ?)", id, title, now)
            loadConversations()
            selectConversation(id: id)
            return id
        } catch {
            print("[ConversationStore] Failed to create conversation: \(error)")
            return id
        }
    }
    
    public func deleteConversation(id: String) {
        guard let conn = self.conn else { return }
        
        do {
            try conn.execute("DELETE FROM conversations WHERE id = ?", id)
            // Trigger cascading deletion of turns manually if foreign key cascade is disabled in this sqlite build
            try conn.execute("DELETE FROM turns WHERE conversationId = ?", id)
            
            loadConversations()
            
            DispatchQueue.main.async {
                if self.activeConversationId == id {
                    self.activeConversationId = self.conversations.first?.id
                    if let newActive = self.activeConversationId {
                        self.loadTurns(for: newActive)
                    } else {
                        self.currentConversationTurns = []
                    }
                }
            }
        } catch {
            print("[ConversationStore] Failed to delete conversation: \(error)")
        }
    }
    
    public func loadTurns(for conversationId: String) {
        guard let conn = self.conn else { return }
        
        do {
            let rows = try conn.query("SELECT id, role, text, config_json, trace_json_ref, createdAt FROM turns WHERE conversationId = ? ORDER BY createdAt ASC", conversationId)
            var loaded: [ChatTurn] = []
            for row in rows {
                do {
                    let id = try row.getString(0)
                    let role = try row.getString(1)
                    let text = try row.getString(2)
                    let configJson = try row.getString(3)
                    let traceJsonRef = try? row.getString(4)
                    let createdAtVal = try row.getDouble(5)
                    
                    loaded.append(ChatTurn(
                        id: id,
                        conversationId: conversationId,
                        role: role,
                        text: text,
                        configJson: configJson,
                        traceJsonRef: traceJsonRef,
                        createdAt: Date(timeIntervalSince1970: createdAtVal)
                    ))
                } catch {
                    print("[ConversationStore] Row mapping error: \(error)")
                    continue
                }
            }
            
            DispatchQueue.main.async {
                self.currentConversationTurns = loaded
            }
        } catch {
            print("[ConversationStore] Failed to load turns: \(error)")
        }
    }
    
    public func addTurn(role: String, text: String, config: SettingsSchema, traceJsonRef: String? = nil) {
        guard let conversationId = self.activeConversationId ?? (conversations.first?.id ?? nil) else {
            // Auto create conversation
            let newId = createConversation(title: text.prefix(30) + "...")
            addTurnToDb(conversationId: newId, role: role, text: text, config: config, traceJsonRef: traceJsonRef)
            return
        }
        
        addTurnToDb(conversationId: conversationId, role: role, text: text, config: config, traceJsonRef: traceJsonRef)
    }
    
    private func addTurnToDb(conversationId: String, role: String, text: String, config: SettingsSchema, traceJsonRef: String?) {
        guard let conn = self.conn else { return }
        
        let id = UUID().uuidString
        let now = Date().timeIntervalSince1970
        let configJson: String
        if let configData = try? JSONEncoder().encode(config) {
            configJson = String(data: configData, encoding: .utf8) ?? "{}"
        } else {
            configJson = "{}"
        }
        
        do {
            try conn.execute(
                "INSERT INTO turns (id, conversationId, role, text, config_json, trace_json_ref, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?)",
                id, conversationId, role, text, configJson, traceJsonRef ?? "", now
            )
            loadTurns(for: conversationId)
        } catch {
            print("[ConversationStore] Failed to insert turn: \(error)")
        }
    }
    
    public func undoLastTurn() {
        guard let conn = self.conn, let conversationId = self.activeConversationId, !currentConversationTurns.isEmpty else { return }
        
        // Remove the last turn. If the last turn was assistant, remove user's query too.
        let lastRole = currentConversationTurns.last?.role
        let deleteCount = (lastRole == "assistant") ? 2 : 1
        
        do {
            // Get ids of the last N turns
            let rows = try conn.query("SELECT id FROM turns WHERE conversationId = ? ORDER BY createdAt DESC LIMIT ?", [conversationId, deleteCount])
            for row in rows {
                if let turnId = try? row.getString(0) {
                    try conn.execute("DELETE FROM turns WHERE id = ?", turnId)
                }
            }
            loadTurns(for: conversationId)
        } catch {
            print("[ConversationStore] Failed to undo last turn: \(error)")
        }
    }
    
    public func clearAll() {
        guard let conn = self.conn else { return }
        do {
            try conn.execute("DELETE FROM conversations")
            try conn.execute("DELETE FROM turns")
            loadConversations()
            DispatchQueue.main.async {
                self.currentConversationTurns = []
                self.activeConversationId = nil
            }
        } catch {
            print("[ConversationStore] Failed to clear all: \(error)")
        }
    }
}
