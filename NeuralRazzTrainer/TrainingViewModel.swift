import Foundation
import Combine

/// ViewModel for the training interface. Communicates with Python backend via HTTP.
class TrainingViewModel: ObservableObject {
    // Server
    var serverManager: ServerManager?

    // Config
    @Published var selectedMode: TrainingMode = .strategy
    @Published var iterations: Int = 100_000
    @Published var learningRate: Double = 0.001
    @Published var batchSize: Int = 256
    @Published var handScope: String = "premium"
    @Published var minVisits: Int = 50
    @Published var trainInterval: Int = 5_000
    @Published var reservoirSize: Int = 2_000_000
    @Published var enableHindsight: Bool = false

    // Value mode config
    @Published var mcSamples: Int = 200
    @Published var trainStepsPerRetrain: Int = 200

    // Opponent config (set by ContentView before training)
    var opponentConfig: [String: Any]? = nil

    // Training State
    @Published var isTraining = false
    @Published var currentIteration: Int = 0
    @Published var totalIterations: Int = 0
    @Published var loss: Double = 0
    @Published var lossHistory: [Double] = []
    @Published var heroInfoSets: Int = 0
    @Published var villainInfoSets: Int = 0
    @Published var reservoirCount: Int = 0
    @Published var trainSteps: Int = 0
    @Published var elapsedSeconds: Double = 0
    @Published var handsInScope: Int = 0
    @Published var hasModel: Bool = false

    // Arena
    @Published var arenaRunning = false
    @Published var arenaResults: [ArenaResult] = []

    // Battery
    @Published var batteryResults: [BatteryResultItem] = []
    @Published var batteryScore: Double = 0
    @Published var batteryTotal: Int = 0
    @Published var batteryPassed: Int = 0

    // Export
    @Published var exportPath: String = ""
    @Published var exportSizeMB: Double = 0

    // Status
    @Published var statusMessage: String = ""

    private var pollTimer: Timer?
    private var baseURL: String { serverManager?.baseURL ?? "http://127.0.0.1:5050" }

    /// Fetch initial state from server — picks up auto-loaded checkpoints
    func checkInitialState() {
        fetchStatus()
        // Retry a few times since server might still be starting
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            self?.fetchStatus()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            self?.fetchStatus()
        }
    }

    // MARK: - Training

    func startTraining() {
        guard !isTraining else { return }
        guard serverManager?.isHealthy ?? false else {
            statusMessage = "Server not ready"
            return
        }

        let modeString: String
        switch selectedMode {
        case .strategy: modeString = "strategy"
        case .regret: modeString = "deep_cfr"
        case .value: modeString = "value"
        }

        var body: [String: Any] = [
            "mode": modeString,
            "iterations": iterations,
            "learning_rate": learningRate,
            "batch_size": batchSize,
            "hand_scope": handScope,
            "min_visits": minVisits,
            "train_interval": trainInterval,
            "reservoir_size": reservoirSize,
            "enable_hindsight": enableHindsight,
        ]
        if selectedMode == .value {
            body["mc_samples"] = mcSamples
            body["train_steps"] = trainStepsPerRetrain
        }
        if let oppConfig = opponentConfig {
            body["opponent_config"] = oppConfig
        }

        post("/api/train/start", body: body) { [weak self] data in
            DispatchQueue.main.async {
                self?.isTraining = true
                self?.lossHistory = []
                self?.statusMessage = "Training started..."
                self?.startPolling()
            }
        }
    }

    func stopTraining() {
        post("/api/train/stop", body: [:]) { [weak self] _ in
            DispatchQueue.main.async {
                self?.statusMessage = "Stopping..."
            }
        }
    }

    func resetTraining() {
        post("/api/train/reset", body: [:]) { [weak self] _ in
            DispatchQueue.main.async {
                self?.statusMessage = "Reset — networks and reservoirs cleared"
                self?.currentIteration = 0
                self?.totalIterations = 0
                self?.loss = 0
                self?.lossHistory = []
                self?.heroInfoSets = 0
                self?.villainInfoSets = 0
                self?.reservoirCount = 0
                self?.trainSteps = 0
                self?.hasModel = false
                self?.arenaResults = []
            }
        }
    }

    func loadCheckpoint() {
        post("/api/checkpoint/load", body: [:]) { [weak self] data in
            DispatchQueue.main.async {
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    let status = json["status"] as? String ?? "unknown"
                    if status == "loaded" {
                        let iteration = json["base_iteration"] as? Int ?? 0
                        self?.hasModel = true
                        self?.currentIteration = iteration
                        self?.totalIterations = iteration
                        self?.statusMessage = "Checkpoint loaded — iteration \(iteration)"
                    } else {
                        self?.statusMessage = json["message"] as? String ?? "No checkpoint found"
                    }
                }
            }
        }
    }

    private func startPolling() {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.fetchStatus()
        }
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func fetchStatus() {
        get("/api/train/status") { [weak self] result in
            guard let json = result as? [String: Any] else { return }

            DispatchQueue.main.async {
                self?.currentIteration = json["iteration"] as? Int ?? 0
                self?.totalIterations = json["total_iterations"] as? Int ?? 0
                self?.loss = json["loss"] as? Double ?? 0
                self?.heroInfoSets = json["hero_info_sets"] as? Int ?? 0
                self?.villainInfoSets = json["villain_info_sets"] as? Int ?? 0
                self?.reservoirCount = json["reservoir_size"] as? Int ?? 0
                self?.trainSteps = json["train_steps"] as? Int ?? 0
                self?.elapsedSeconds = json["elapsed_seconds"] as? Double ?? 0
                self?.handsInScope = json["hands_in_scope"] as? Int ?? 0
                self?.hasModel = json["has_model"] as? Bool ?? false

                if let history = json["loss_history"] as? [Double] {
                    self?.lossHistory = history
                }

                let running = json["running"] as? Bool ?? false
                if !running && (self?.isTraining ?? false) {
                    self?.isTraining = false
                    self?.stopPolling()
                    self?.statusMessage = "Training complete"
                }
            }
        }
    }

    // MARK: - Arena

    func runArena(opponent: String, numHands: Int, handScope: String? = nil, singleHand: String? = nil) {
        guard !arenaRunning else { return }
        arenaRunning = true
        let scopeLabel = singleHand ?? handScope ?? "random"
        statusMessage = "Running arena vs \(opponent) (\(scopeLabel))..."

        var body: [String: Any] = [
            "opponent": opponent,
            "num_hands": numHands,
        ]
        if let scope = handScope {
            body["hand_scope"] = scope
        }
        if let hand = singleHand, !hand.isEmpty {
            body["single_hand"] = hand
        }

        post("/api/test/arena", body: body) { [weak self] data in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                DispatchQueue.main.async {
                    self?.arenaRunning = false
                    self?.statusMessage = "Arena failed"
                }
                return
            }

            DispatchQueue.main.async {
                // Parse hand histories
                var histories: [HandHistoryEntry] = []
                if let rawHistories = json["hand_histories"] as? [[String: Any]] {
                    for hh in rawHistories {
                        histories.append(HandHistoryEntry(
                            handNum: hh["hand_num"] as? Int ?? 0,
                            heroStart: hh["hero_start"] as? String ?? "?",
                            villainStart: hh["villain_start"] as? String ?? "?",
                            heroDoor: hh["hero_door"] as? String ?? "?",
                            villainDoor: hh["villain_door"] as? String ?? "?",
                            heroFinal: hh["hero_final"] as? String ?? "?",
                            villainFinal: hh["villain_final"] as? String ?? "?",
                            result: hh["result"] as? String ?? "?",
                            payoff: hh["payoff"] as? Double ?? 0,
                            heroFolded: hh["hero_folded"] as? Bool ?? false,
                            actions: hh["actions"] as? [String] ?? [],
                            pot: hh["pot"] as? Double ?? 0
                        ))
                    }
                }

                var result = ArenaResult(
                    opponent: json["opponent"] as? String ?? opponent,
                    numHands: json["num_hands"] as? Int ?? numHands,
                    winRate: json["win_rate"] as? Double ?? 0,
                    bbPer100: json["bb_per_100"] as? Double ?? 0,
                    foldRate: json["fold_rate"] as? Double ?? 0,
                    sdWinRate: json["sd_win_rate"] as? Double ?? 0,
                    handHistories: histories
                )
                self?.arenaResults.append(result)
                self?.arenaRunning = false
                self?.statusMessage = "Arena: \(result.winRate)% win, \(String(format: "%+.1f", result.bbPer100)) BB/100 vs \(opponent)"
            }
        }
    }

    // MARK: - Battery Test

    func runBattery() {
        guard !arenaRunning else { return }
        arenaRunning = true
        statusMessage = "Running battery test..."

        post("/api/test/battery", body: [:]) { [weak self] data in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                DispatchQueue.main.async {
                    self?.arenaRunning = false
                    self?.statusMessage = "Battery test failed"
                }
                return
            }

            DispatchQueue.main.async {
                self?.arenaRunning = false
                self?.batteryPassed = json["total_passed"] as? Int ?? 0
                self?.batteryTotal = json["total_tests"] as? Int ?? 0
                self?.batteryScore = json["score"] as? Double ?? 0

                var items: [BatteryResultItem] = []
                if let results = json["results"] as? [[String: Any]] {
                    for r in results {
                        items.append(BatteryResultItem(
                            name: r["name"] as? String ?? "?",
                            tier: r["tier"] as? Int ?? 0,
                            passed: r["passed"] as? Bool ?? false,
                            predicted: r["predicted_action"] as? String ?? "?",
                            expected: r["expected_actions"] as? [String] ?? [],
                            confidence: r["confidence"] as? Double ?? 0,
                            explanation: r["explanation"] as? String ?? ""
                        ))
                    }
                }
                self?.batteryResults = items
                self?.statusMessage = "Battery: \(self?.batteryPassed ?? 0)/\(self?.batteryTotal ?? 0) (\(Int((self?.batteryScore ?? 0) * 100))%)"
            }
        }
    }

    // MARK: - Export

    func exportModel() {
        post("/api/model/export", body: [:]) { [weak self] data in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

            DispatchQueue.main.async {
                self?.exportPath = json["path"] as? String ?? ""
                self?.exportSizeMB = json["size_mb"] as? Double ?? 0
                self?.statusMessage = "Exported: \(self?.exportPath ?? "") (\(String(format: "%.1f", self?.exportSizeMB ?? 0)) MB)"
            }
        }
    }

    // MARK: - HTTP Helpers

    func get(_ path: String, completion: @escaping (Any?) -> Void) {
        guard let url = URL(string: "\(baseURL)\(path)") else { return }
        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) else {
                completion(nil)
                return
            }
            completion(json)
        }.resume()
    }

    func post(_ path: String, body: [String: Any], completion: @escaping (Data?) -> Void) {
        guard let url = URL(string: "\(baseURL)\(path)") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: req) { data, _, _ in
            completion(data)
        }.resume()
    }
}

// MARK: - Models

enum TrainingMode: String, CaseIterable {
    case strategy = "Strategy"
    case regret = "Deep CFR"
    case value = "Value"
}

struct HandHistoryEntry: Identifiable {
    let id = UUID()
    let handNum: Int
    let heroStart: String
    let villainStart: String
    let heroDoor: String
    let villainDoor: String
    let heroFinal: String
    let villainFinal: String
    let result: String
    let payoff: Double
    let heroFolded: Bool
    let actions: [String]
    let pot: Double
}

struct ArenaResult: Identifiable {
    let id = UUID()
    let opponent: String
    let numHands: Int
    let winRate: Double
    let bbPer100: Double
    let foldRate: Double
    let sdWinRate: Double
    var handHistories: [HandHistoryEntry] = []
}

struct BatteryResultItem: Identifiable {
    let id = UUID()
    let name: String
    let tier: Int
    let passed: Bool
    let predicted: String
    let expected: [String]
    let confidence: Double
    let explanation: String
}
