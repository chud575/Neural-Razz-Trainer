import SwiftUI

struct ContentView: View {
    @ObservedObject var serverManager: ServerManager
    @ObservedObject var vm: TrainingViewModel

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(spacing: 16) {
                        serverStatusCard
                        trainingConfigCard
                        if vm.isTraining || vm.currentIteration > 0 {
                            trainingProgressCard
                        }
                        if !vm.lossHistory.isEmpty {
                            lossChartCard
                        }
                        if vm.hasModel {
                            arenaCard
                            exportCard
                        }
                        if !vm.arenaResults.isEmpty {
                            arenaResultsCard
                        }
                        if !vm.batteryResults.isEmpty {
                            batteryResultsCard
                        }
                        if vm.hasModel {
                            autoTrainCard
                        }
                    }
                    .padding()
                }

                // Server log pinned to bottom, fills remaining space
                serverLogCard
                    .padding([.horizontal, .bottom])
            }
        }
        .frame(minWidth: 900, minHeight: 600)
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List {
            Section("Mode") {
                ForEach(TrainingMode.allCases, id: \.self) { mode in
                    Label(mode.rawValue, systemImage: iconForMode(mode))
                        .tag(mode)
                        .foregroundColor(vm.selectedMode == mode ? .accentColor : .primary)
                        .onTapGesture { vm.selectedMode = mode }
                }
            }

            Section("Status") {
                HStack {
                    Circle().fill(serverManager.isHealthy ? .green : .red).frame(width: 8, height: 8)
                    Text(serverManager.isHealthy ? "Server Online" : "Server Offline")
                        .font(.caption)
                }
                if vm.hasModel {
                    HStack {
                        Image(systemName: "brain").foregroundColor(.purple)
                        Text("Model Ready").font(.caption)
                    }
                }
            }

            Section("Quick Stats") {
                if vm.heroInfoSets > 0 {
                    Label("\(formatK(vm.heroInfoSets)) hero info sets", systemImage: "tablecells")
                        .font(.caption)
                }
                if vm.trainSteps > 0 {
                    Label("\(vm.trainSteps) train steps", systemImage: "arrow.triangle.2.circlepath")
                        .font(.caption)
                }
                if vm.loss > 0 {
                    Label(String(format: "Loss: %.4f", vm.loss), systemImage: "chart.line.downtrend.xyaxis")
                        .font(.caption)
                }
            }
        }
        .listStyle(.sidebar)
        .frame(minWidth: 180)
    }

    // MARK: - Server Status

    private var serverStatusCard: some View {
        HStack {
            Circle().fill(serverManager.isHealthy ? .green : .orange).frame(width: 12, height: 12)
            Text(serverManager.isHealthy ? "Backend Server Running" : "Starting Backend Server...")
                .font(.headline)
            if serverManager.isHealthy && !serverManager.serverVersion.isEmpty {
                Text("v\(serverManager.serverVersion)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
            if !vm.statusMessage.isEmpty {
                Text(vm.statusMessage)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Training Config

    private var trainingConfigCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Training Configuration").font(.headline)
                Spacer()
                Text(vm.selectedMode.rawValue)
                    .font(.caption.bold())
                    .padding(.horizontal, 8).padding(.vertical, 2)
                    .background(Capsule().fill(.purple.opacity(0.2)))
            }

            HStack {
                Text("Iterations").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.iterations) {
                    Text("10K").tag(10_000)
                    Text("50K").tag(50_000)
                    Text("100K").tag(100_000)
                    Text("500K").tag(500_000)
                    Text("1M").tag(1_000_000)
                }.pickerStyle(.segmented)
            }

            HStack {
                Text("Hand Scope").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.handScope) {
                    Text("Premium (56)").tag("premium")
                    Text("Top 50% (220)").tag("top50")
                    Text("All Unpaired (286)").tag("allUnpaired")
                    Text("All Hands (455)").tag("allHands")
                }.pickerStyle(.segmented)
            }

            HStack {
                Text("EV Group").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.handScope) {
                    Text("—").tag("premium")
                    Text("G1: Elite 70%+ (35)").tag("ev_group_1")
                    Text("G2: Strong 60-70% (60)").tag("ev_group_2")
                    Text("G3: Good 50-60% (55)").tag("ev_group_3")
                    Text("G4: Playable 45-50% (114)").tag("ev_group_4")
                    Text("G5: Marginal 40-45% (55)").tag("ev_group_5")
                    Text("G6: Weak 35-40% (26)").tag("ev_group_6")
                    Text("G7: Bad 25-35% (62)").tag("ev_group_7")
                    Text("G8: Trash <25% (48)").tag("ev_group_8")
                }.pickerStyle(.menu)
            }

            HStack {
                Text("Learning Rate").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.learningRate) {
                    Text("0.01").tag(0.01)
                    Text("0.001").tag(0.001)
                    Text("0.0003").tag(0.0003)
                    Text("0.0001").tag(0.0001)
                }.pickerStyle(.segmented)
            }

            HStack {
                Text("Batch Size").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.batchSize) {
                    Text("128").tag(128)
                    Text("256").tag(256)
                    Text("512").tag(512)
                    Text("1024").tag(1024)
                }.pickerStyle(.segmented)
            }

            HStack {
                Text("Min Visits").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.minVisits) {
                    Text("10").tag(10)
                    Text("50").tag(50)
                    Text("100").tag(100)
                    Text("500").tag(500)
                }.pickerStyle(.segmented)
            }

            HStack {
                Text("Reservoir").frame(width: 100, alignment: .leading)
                Picker("", selection: $vm.reservoirSize) {
                    Text("200K").tag(200_000)
                    Text("500K").tag(500_000)
                    Text("1M").tag(1_000_000)
                    Text("2M").tag(2_000_000)
                    Text("5M").tag(5_000_000)
                    Text("10M").tag(10_000_000)
                }.pickerStyle(.segmented)
            }

            // Value mode options
            if vm.selectedMode == .value {
                HStack {
                    Text("MC Samples").frame(width: 100, alignment: .leading)
                    Picker("", selection: $vm.mcSamples) {
                        Text("50").tag(50)
                        Text("100").tag(100)
                        Text("200").tag(200)
                        Text("500").tag(500)
                    }.pickerStyle(.segmented)
                }

                HStack {
                    Text("Train Steps").frame(width: 100, alignment: .leading)
                    Picker("", selection: $vm.trainStepsPerRetrain) {
                        Text("50").tag(50)
                        Text("100").tag(100)
                        Text("200").tag(200)
                        Text("500").tag(500)
                    }.pickerStyle(.segmented)
                }

                HStack {
                    Text("Train Interval").frame(width: 100, alignment: .leading)
                    Picker("", selection: $vm.trainInterval) {
                        Text("500").tag(500)
                        Text("1K").tag(1_000)
                        Text("2K").tag(2_000)
                        Text("5K").tag(5_000)
                    }.pickerStyle(.segmented)
                }
            }

            // Hindsight toggle (Deep CFR only)
            if vm.selectedMode == .regret {
                HStack {
                    Toggle(isOn: $vm.enableHindsight) {
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Hindsight Correction").font(.subheadline)
                            Text(vm.enableHindsight
                                 ? "Replays hands with perfect info to find missed bets — slower but learns aggression"
                                 : "Standard Deep CFR training — no hindsight analysis")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                    }
                    .tint(.pink)
                }

                // Opponent controls
                opponentControlsSection
            }

            Divider()

            HStack(spacing: 12) {
                if vm.isTraining {
                    Button(action: { vm.stopTraining() }) {
                        Label("Stop", systemImage: "stop.fill")
                    }.buttonStyle(.borderedProminent).tint(.red)
                } else {
                    Button(action: {
                        // Pass opponent config to VM before training
                        let enabled = Array(enabledOpponents)
                        if balancedTraining {
                            vm.opponentConfig = [
                                "enabled": enabled,
                                "balanced": true,
                            ]
                        } else {
                            let weights = enabled.reduce(into: [String: Double]()) { dict, opp in
                                dict[opp] = oppWeights[opp] ?? 10
                            }
                            vm.opponentConfig = [
                                "enabled": enabled,
                                "balanced": false,
                                "weights": weights,
                            ]
                        }
                        vm.startTraining()
                    }) {
                        Label(vm.hasModel ? "Continue" : "Train", systemImage: vm.hasModel ? "arrow.clockwise" : "bolt.fill")
                    }
                    .buttonStyle(.borderedProminent).tint(vm.hasModel ? .orange : .purple)
                    .disabled(!serverManager.isHealthy)

                    if vm.hasModel {
                        Button(action: { vm.resetTraining() }) {
                            Label("Reset", systemImage: "trash")
                        }
                        .buttonStyle(.bordered).tint(.red)
                    }

                    if !vm.hasModel {
                        Button(action: { vm.loadCheckpoint() }) {
                            Label("Load Checkpoint", systemImage: "square.and.arrow.down")
                        }
                        .buttonStyle(.bordered).tint(.blue)
                        .disabled(!serverManager.isHealthy)
                    }
                }

                Spacer()

                if vm.handsInScope > 0 {
                    Text("\(vm.handsInScope) hands in scope")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Training Progress

    private var trainingProgressCard: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Training Progress").font(.headline)
                Spacer()
                if vm.elapsedSeconds > 0 {
                    let itersPerSec = Double(vm.currentIteration) / max(vm.elapsedSeconds, 0.01)
                    Text("\(Int(itersPerSec)) iter/s")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }

            ProgressView(value: Double(vm.currentIteration), total: Double(max(vm.totalIterations, 1)))

            HStack {
                Text("\(formatK(vm.currentIteration)) / \(formatK(vm.totalIterations))")
                    .font(.system(.caption, design: .monospaced))
                Spacer()
                if vm.elapsedSeconds > 0 {
                    Text(formatTime(vm.elapsedSeconds))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }

            HStack(spacing: 20) {
                statPill("Hero IS", formatK(vm.heroInfoSets), .blue)
                statPill("Villain IS", formatK(vm.villainInfoSets), .orange)
                statPill("Reservoir", formatK(vm.reservoirCount), .green)
                statPill("Train Steps", "\(vm.trainSteps)", .purple)
                if vm.loss > 0 {
                    statPill("Loss", String(format: "%.4f", vm.loss), .red)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Loss Chart

    private var lossChartCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Loss History").font(.headline)

            // Simple sparkline using SwiftUI shapes
            GeometryReader { geo in
                let data = vm.lossHistory
                if data.count > 1 {
                    let maxVal = data.max() ?? 1
                    let minVal = data.min() ?? 0
                    let range = max(maxVal - minVal, 0.001)

                    Path { path in
                        for (i, val) in data.enumerated() {
                            let x = CGFloat(i) / CGFloat(data.count - 1) * geo.size.width
                            let y = (1.0 - CGFloat((val - minVal) / range)) * geo.size.height
                            if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                            else { path.addLine(to: CGPoint(x: x, y: y)) }
                        }
                    }
                    .stroke(Color.purple, lineWidth: 2)
                }
            }
            .frame(height: 100)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.purple.opacity(0.05)))

            HStack {
                if let first = vm.lossHistory.first {
                    Text(String(format: "Start: %.4f", first)).font(.caption2).foregroundColor(.secondary)
                }
                Spacer()
                if let last = vm.lossHistory.last {
                    Text(String(format: "Current: %.4f", last)).font(.caption2).foregroundColor(.purple)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Opponent Controls

    private let allOpponentTypes: [(id: String, label: String)] = [
        ("self_play", "Self-Play"),
        ("tag", "TAG"),
        ("lag", "LAG"),
        ("pure_ev", "Pure EV"),
        ("rebel", "ReBeL"),
        ("bucketed_cfr", "Pure CFR"),
        ("calling_station", "Calling Station"),
        ("random", "Random"),
    ]

    private var opponentControlsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            HStack {
                Text("Training Opponents").font(.subheadline).fontWeight(.medium)
                Spacer()
                Toggle(isOn: $balancedTraining) {
                    Text(balancedTraining ? "Balanced" : "Weighted")
                        .font(.caption2)
                }
                .toggleStyle(.switch)
                .tint(.cyan)
            }

            // Opponent checklist
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 4) {
                ForEach(allOpponentTypes, id: \.id) { opp in
                    HStack(spacing: 6) {
                        Button(action: {
                            if enabledOpponents.contains(opp.id) {
                                enabledOpponents.remove(opp.id)
                            } else {
                                enabledOpponents.insert(opp.id)
                            }
                        }) {
                            Image(systemName: enabledOpponents.contains(opp.id) ? "checkmark.square.fill" : "square")
                                .foregroundColor(enabledOpponents.contains(opp.id) ? .green : .secondary)
                                .font(.caption)
                        }
                        .buttonStyle(.plain)

                        Text(opp.label)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(enabledOpponents.contains(opp.id) ? .primary : .secondary)

                        if !balancedTraining && enabledOpponents.contains(opp.id) {
                            Spacer()
                            Slider(value: Binding(
                                get: { oppWeights[opp.id] ?? 10 },
                                set: { oppWeights[opp.id] = $0 }
                            ), in: 1...50, step: 1)
                            .frame(width: 60)
                            Text("\(Int(oppWeights[opp.id] ?? 10))%")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.secondary)
                                .frame(width: 25)
                        }
                        Spacer()
                    }
                }
            }

            Text("\(enabledOpponents.count) opponents active\(balancedTraining ? " (equal weight)" : "")")
                .font(.caption2).foregroundColor(.secondary)
        }
    }

    // MARK: - Arena

    @State private var arenaScope: String = "premium"
    @State private var arenaSingleHand: String = ""
    @State private var arenaNumHands: Int = 5000
    @State private var showHandHistory: Bool = false

    // Opponent controls
    @State private var enabledOpponents: Set<String> = ["self_play", "tag", "lag", "pure_ev", "rebel", "bucketed_cfr", "calling_station", "random"]
    @State private var balancedTraining: Bool = true
    @State private var oppWeights: [String: Double] = [
        "self_play": 50, "tag": 25, "lag": 20, "pure_ev": 20, "rebel": 15,
        "bucketed_cfr": 10, "calling_station": 5, "random": 5,
    ]

    private var arenaCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Arena Test").font(.headline)
            Text("Test the trained model against various opponents")
                .font(.caption).foregroundColor(.secondary)

            // Hand scope picker
            HStack {
                Text("Hero Hands").font(.caption)
                Spacer()
                Picker("", selection: $arenaScope) {
                    Text("Premium (56)").tag("premium")
                    Text("Top 50% (220)").tag("top50")
                    Text("Unpaired (286)").tag("allUnpaired")
                    Text("All (455)").tag("allHands")
                    Divider()
                    Text("G1: Elite 70%+ (35)").tag("ev_group_1")
                    Text("G2: Strong 60-70% (60)").tag("ev_group_2")
                    Text("G3: Good 50-60% (55)").tag("ev_group_3")
                    Text("G4: Playable 45-50% (114)").tag("ev_group_4")
                    Text("G5: Marginal 40-45% (55)").tag("ev_group_5")
                    Text("G6: Weak 35-40% (26)").tag("ev_group_6")
                    Text("G7: Bad 25-35% (62)").tag("ev_group_7")
                    Text("G8: Trash <25% (48)").tag("ev_group_8")
                    Divider()
                    Text("Random").tag("random")
                    Text("Single Hand").tag("single")
                }.pickerStyle(.menu).frame(width: 180)
            }

            // Single hand input (only when "Single Hand" selected)
            if arenaScope == "single" {
                HStack {
                    Text("Hand").font(.caption)
                    TextField("e.g. A23, 369", text: $arenaSingleHand)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 120)
                }
            }

            HStack {
                Text("Hands").font(.caption)
                Spacer()
                ForEach([5000, 10000, 25000], id: \.self) { n in
                    Button(n >= 10000 ? "\(n/1000)K" : "5K") { arenaNumHands = n }
                        .buttonStyle(.bordered).controlSize(.small)
                        .tint(arenaNumHands == n ? .accentColor : .secondary)
                }
            }

            HStack(spacing: 8) {
                ForEach(["calling_station", "tag", "lag", "random"], id: \.self) { opp in
                    Button(action: {
                        let scope = arenaScope == "random" ? nil : (arenaScope == "single" ? nil : arenaScope)
                        let single = arenaScope == "single" ? arenaSingleHand : nil
                        vm.runArena(opponent: opp, numHands: arenaNumHands, handScope: scope, singleHand: single)
                    }) {
                        Text(opp.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.caption)
                    }
                    .buttonStyle(.bordered).controlSize(.small)
                    .disabled(vm.arenaRunning)
                }

                Divider().frame(height: 20)

                Button(action: { vm.runBattery() }) {
                    Label("Battery", systemImage: "checklist")
                        .font(.caption)
                }
                .buttonStyle(.bordered).controlSize(.small)
                .tint(.orange)
                .disabled(vm.arenaRunning)

                if vm.arenaRunning {
                    ProgressView().controlSize(.small)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    private var arenaResultsCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Arena Results").font(.headline)
                Spacer()
                if let last = vm.arenaResults.last, !last.handHistories.isEmpty {
                    Button(action: { showHandHistory.toggle() }) {
                        Label(showHandHistory ? "Hide Hands" : "Hand History", systemImage: "list.bullet.rectangle")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered).controlSize(.small)
                }
            }

            ForEach(vm.arenaResults.suffix(8)) { result in
                HStack {
                    Text(result.opponent.replacingOccurrences(of: "_", with: " ").capitalized)
                        .font(.system(.caption, design: .monospaced)).frame(width: 120, alignment: .leading)
                    Text(String(format: "Win: %.1f%%", result.winRate))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(result.winRate > 50 ? .green : .red)
                        .frame(width: 80)
                    Text(String(format: "BB/100: %+.1f", result.bbPer100))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(result.bbPer100 > 0 ? .green : .red)
                        .frame(width: 100)
                    Text(String(format: "Fold: %.1f%%", result.foldRate))
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 70)
                    Text(String(format: "SD Win: %.1f%%", result.sdWinRate))
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 90)
                    Text("\(result.numHands) hands")
                        .font(.caption2).foregroundColor(.secondary)
                }
            }

            // Hand History (collapsible)
            if showHandHistory, let last = vm.arenaResults.last {
                Divider()
                Text("Hand History (sorted by biggest loss)").font(.caption.bold())

                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(last.handHistories.prefix(50)) { hh in
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text("#\(hh.handNum)")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundColor(.secondary)
                                        .frame(width: 40, alignment: .leading)
                                    Text("Hero: \(hh.heroStart)")
                                        .font(.system(size: 10, design: .monospaced).bold())
                                        .foregroundColor(.blue)
                                    Text("vs \(hh.villainDoor)")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundColor(.red)
                                    Spacer()
                                    Text(hh.result)
                                        .font(.system(size: 10, design: .monospaced).bold())
                                        .foregroundColor(hh.payoff > 0 ? .green : (hh.payoff < 0 ? .red : .secondary))
                                    Text(String(format: "%+.1f", hh.payoff))
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundColor(hh.payoff > 0 ? .green : .red)
                                        .frame(width: 40, alignment: .trailing)
                                }

                                // Action details (expandable)
                                ForEach(Array(hh.actions.enumerated()), id: \.offset) { _, action in
                                    Text(action)
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundColor(.secondary)
                                        .padding(.leading, 44)
                                }

                                if !hh.heroFolded {
                                    HStack {
                                        Spacer()
                                        Text("Final: Hero=\(hh.heroFinal) Opp=\(hh.villainFinal)")
                                            .font(.system(size: 9, design: .monospaced))
                                            .foregroundColor(.secondary)
                                    }
                                }

                                Divider().opacity(0.3)
                            }
                        }
                    }
                }
                .frame(maxHeight: 400)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Battery Results

    private var batteryResultsCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Battery Results").font(.headline)
                Spacer()
                Text("\(vm.batteryPassed)/\(vm.batteryTotal) (\(Int(vm.batteryScore * 100))%)")
                    .font(.system(.caption, design: .monospaced).bold())
                    .foregroundColor(vm.batteryScore >= 0.8 ? .green : vm.batteryScore >= 0.5 ? .orange : .red)
            }

            ForEach(vm.batteryResults) { r in
                HStack(spacing: 6) {
                    Text(r.passed ? "✅" : "❌").font(.caption)
                    Text("T\(r.tier)").font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary).frame(width: 20)
                    Text(r.name).font(.caption).frame(maxWidth: .infinity, alignment: .leading)
                    Text(r.predicted.uppercased())
                        .font(.system(.caption, design: .monospaced).bold())
                        .foregroundColor(r.passed ? .green : .red)
                        .frame(width: 50)
                    Text(String(format: "%.0f%%", r.confidence * 100))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                        .frame(width: 35)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Export

    private var exportCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Export Model").font(.headline)
                Spacer()
                Button(action: { vm.exportModel() }) {
                    Label("Export JSON", systemImage: "square.and.arrow.up")
                }.buttonStyle(.bordered).controlSize(.small)
            }

            if !vm.exportPath.isEmpty {
                Text(vm.exportPath)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
                Text(String(format: "%.1f MB", vm.exportSizeMB))
                    .font(.caption).foregroundColor(.green)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    // MARK: - Auto-Train

    @State private var autoTrainIterations: Int = 50_000
    @State private var autoTrainStartGroup: Int = 0
    @State private var autoTrainRunning = false
    @State private var autoTrainPhase: String = "idle"
    @State private var autoTrainGroup: Int = 0
    @State private var autoTrainGroupName: String = ""
    @State private var autoTrainResults: [[String: Any]] = []
    @State private var autoTrainPollTimer: Timer?

    private var autoTrainCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Auto-Training Pipeline").font(.headline)
                Spacer()
                if autoTrainRunning {
                    Text(autoTrainPhase.capitalized)
                        .font(.caption.bold())
                        .foregroundColor(.orange)
                }
            }

            Text("Trains all 8 EV groups sequentially with curriculum injection, hindsight correction, battery tests, and arena validation.")
                .font(.caption).foregroundColor(.secondary)

            if !autoTrainRunning {
                HStack {
                    Text("Iterations per group").font(.caption)
                    Spacer()
                    ForEach([50_000, 100_000, 500_000, 1_000_000], id: \.self) { n in
                        Button(n >= 1_000_000 ? "1M" : "\(n/1000)K") {
                            autoTrainIterations = n
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                        .tint(autoTrainIterations == n ? .orange : .secondary)
                    }
                }

                HStack {
                    Text("Start from group").font(.caption)
                    Spacer()
                    Picker("", selection: $autoTrainStartGroup) {
                        Text("G1: Elite 70%+").tag(0)
                        Text("G2: Strong 60-70%").tag(1)
                        Text("G3: Good 50-60%").tag(2)
                        Text("G4: Playable 45-50%").tag(3)
                        Text("G5: Marginal 40-45%").tag(4)
                        Text("G6: Weak 35-40%").tag(5)
                        Text("G7: Bad 25-35%").tag(6)
                        Text("G8: Trash <25%").tag(7)
                    }
                    .pickerStyle(.menu)
                    .frame(width: 200)
                }

                HStack {
                    Button(action: { startAutoTrain() }) {
                        Label(autoTrainStartGroup > 0 ? "Resume Auto-Train (G\(autoTrainStartGroup + 1))" : "Start Auto-Train",
                              systemImage: autoTrainStartGroup > 0 ? "arrow.clockwise" : "bolt.trianglebadge.exclamationmark.fill")
                    }
                    .buttonStyle(.borderedProminent).tint(.orange)
                    .disabled(!vm.hasModel)

                    Spacer()

                    if !vm.hasModel {
                        Text("Requires trained model as foundation")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                }
            } else {
                // Progress
                VStack(alignment: .leading, spacing: 4) {
                    Text("Group \(autoTrainGroup)/8: \(autoTrainGroupName)")
                        .font(.system(.caption, design: .monospaced).bold())
                        .foregroundColor(.orange)

                    ProgressView(value: Double(autoTrainGroup), total: 8)
                        .tint(.orange)

                    Button(action: { stopAutoTrain() }) {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent).tint(.red).controlSize(.small)
                }

                // Group results
                if !autoTrainResults.isEmpty {
                    Divider()
                    ForEach(Array(autoTrainResults.enumerated()), id: \.offset) { _, result in
                        let passed = result["passed"] as? Bool ?? false
                        let group = result["group"] as? String ?? "?"
                        let battery = result["battery_score"] as? Double ?? 0
                        let bb = result["bb_vs_tag"] as? Double ?? 0
                        HStack {
                            Image(systemName: passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundColor(passed ? .green : .red)
                            Text(group).font(.caption)
                            Spacer()
                            Text(String(format: "Battery: %.0f%%", battery * 100))
                                .font(.system(.caption2, design: .monospaced))
                            Text(String(format: "TAG: %+.0f", bb))
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundColor(bb >= 0 ? .green : .red)
                        }
                    }
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
    }

    private func startAutoTrain() {
        autoTrainRunning = true
        if autoTrainStartGroup == 0 {
            autoTrainResults = []  // Fresh start — clear results
        }

        let body: [String: Any] = [
            "iterations_per_group": autoTrainIterations,
            "start_group": autoTrainStartGroup,
            "learning_rate": vm.learningRate,
            "batch_size": vm.batchSize,
            "enable_hindsight": true,
            "reservoir_size": vm.reservoirSize,
            "inject_curriculum": true,
        ]

        vm.post("/api/auto-train/start", body: body) { _ in }

        // Start polling
        autoTrainPollTimer?.invalidate()
        autoTrainPollTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { _ in
            pollAutoTrain()
        }
    }

    private func stopAutoTrain() {
        vm.post("/api/auto-train/stop", body: [:]) { _ in }
    }

    private func pollAutoTrain() {
        vm.get("/api/auto-train/status") { data in
            guard let json = data as? [String: Any] else { return }
            DispatchQueue.main.async {
                autoTrainRunning = json["running"] as? Bool ?? false
                autoTrainPhase = json["phase"] as? String ?? "idle"
                autoTrainGroup = json["current_group"] as? Int ?? 0
                autoTrainGroupName = json["current_group_name"] as? String ?? ""
                if let results = json["group_results"] as? [[String: Any]] {
                    autoTrainResults = results
                }
                if !autoTrainRunning {
                    autoTrainPollTimer?.invalidate()
                    autoTrainPollTimer = nil
                }
            }
        }
    }

    // MARK: - Server Log

    private var serverLogCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Server Log").font(.headline)
                Spacer()
                Button("Clear") { serverManager.serverLog.removeAll() }
                    .buttonStyle(.bordered).controlSize(.mini)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(serverManager.serverLog.suffix(50).joined(separator: "\n"))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("logBottom")
                }
                .onChange(of: serverManager.serverLog.count) { _ in
                    withAnimation {
                        proxy.scrollTo("logBottom", anchor: .bottom)
                    }
                }
            }
            .frame(minHeight: 80, maxHeight: .infinity)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(.ultraThinMaterial))
        .frame(minHeight: 120, maxHeight: .infinity)
    }

    // MARK: - Helpers

    private func statPill(_ label: String, _ value: String, _ color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(.caption, design: .monospaced).bold()).foregroundColor(color)
            Text(label).font(.system(size: 9)).foregroundColor(.secondary)
        }
    }

    private func iconForMode(_ mode: TrainingMode) -> String {
        switch mode {
        case .strategy: return "brain"
        case .regret: return "chart.bar"
        case .value: return "function"
        }
    }

    private func formatK(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    private func formatTime(_ seconds: Double) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%d:%02d", m, s)
    }
}

#Preview {
    ContentView(serverManager: ServerManager(), vm: TrainingViewModel())
}
