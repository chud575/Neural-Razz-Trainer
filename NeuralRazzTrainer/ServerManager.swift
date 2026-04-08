import Foundation
import Combine

/// Manages the Python backend server lifecycle.
/// Launches `python3 backend/server.py` as a child process and monitors health.
class ServerManager: ObservableObject {
    @Published var isRunning = false
    @Published var isHealthy = false
    @Published var serverVersion: String = ""
    @Published var serverLog: [String] = []

    private var process: Process?
    private var healthTimer: Timer?

    let baseURL = "http://127.0.0.1:5050"

    func start() {
        guard !isRunning else { return }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")

        // Find the backend directory relative to the app
        let backendDir = findBackendDir()
        process.arguments = ["-u", backendDir + "/server.py"]
        process.currentDirectoryURL = URL(fileURLWithPath: backendDir)

        // Capture stdout/stderr
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let str = String(data: data, encoding: .utf8), !str.isEmpty {
                DispatchQueue.main.async {
                    let lines = str.split(separator: "\n").map(String.init)
                    self?.serverLog.append(contentsOf: lines)
                    // Keep last 200 lines
                    if (self?.serverLog.count ?? 0) > 200 {
                        self?.serverLog = Array(self?.serverLog.suffix(200) ?? [])
                    }
                }
            }
        }

        do {
            try process.run()
            self.process = process
            isRunning = true
            log("Server process started (PID \(process.processIdentifier))")

            // Start health checks after a brief delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                self?.startHealthChecks()
            }
        } catch {
            log("Failed to start server: \(error)")
        }
    }

    func stop() {
        healthTimer?.invalidate()
        healthTimer = nil

        if let process = process, process.isRunning {
            process.terminate()
            log("Server process terminated")
        }
        process = nil
        isRunning = false
        isHealthy = false
    }

    private func startHealthChecks() {
        healthTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.checkHealth()
        }
        checkHealth()
    }

    private func checkHealth() {
        guard let url = URL(string: "\(baseURL)/api/health") else { return }

        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    if !(self?.isHealthy ?? false) {
                        self?.log("Server is healthy")
                    }
                    self?.isHealthy = true
                    // Parse version from health response
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let version = json["version"] as? String {
                        self?.serverVersion = version
                    }
                } else {
                    self?.isHealthy = false
                }
            }
        }.resume()
    }

    private func findBackendDir() -> String {
        // Strategy: walk up from the source file location (Package.swift directory)
        // to find the backend/ sibling directory. This works for both the main repo
        // and worktree checkouts.
        let candidates = [
            // Development: source tree (various nesting levels from DerivedData)
            Bundle.main.bundlePath + "/../../../../../backend",
            Bundle.main.bundlePath + "/../../../../backend",
            Bundle.main.bundlePath + "/../../../../../../backend",
            // Explicit paths — worktree first, then main repo
            NSHomeDirectory() + "/Documents/Poker Apps/Neural-Razz-Trainer/.claude/worktrees/eloquent-perlman/backend",
            NSHomeDirectory() + "/Documents/Poker Apps/Neural-Razz-Trainer/backend",
        ]

        for path in candidates {
            let resolved = (path as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: resolved + "/server.py") {
                log("Found backend at: \(resolved)")
                return resolved
            }
        }

        // Fallback
        let fallback = NSHomeDirectory() + "/Documents/Poker Apps/Neural-Razz-Trainer/backend"
        log("WARNING: Using fallback backend path: \(fallback)")
        return fallback
    }

    private func log(_ msg: String) {
        print("[ServerManager] \(msg)")
        DispatchQueue.main.async {
            self.serverLog.append(msg)
        }
    }
}
