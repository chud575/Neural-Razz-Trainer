import SwiftUI
import AppKit

/// Make the app a proper foreground application with Dock icon, menu bar, Cmd+Tab
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Set as regular (foreground) app — gives us Dock icon, Cmd+Tab, menu bar
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)

        // Ensure the window is visible and key
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            if let window = NSApplication.shared.windows.first {
                window.makeKeyAndOrderFront(nil)
                window.title = "Neural Razz Trainer"
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true  // Cmd+Q / closing window quits the app
    }
}

@main
struct NeuralRazzTrainerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var serverManager = ServerManager()
    @StateObject private var trainingVM = TrainingViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(serverManager: serverManager, vm: trainingVM)
                .onAppear {
                    serverManager.start()
                    trainingVM.serverManager = serverManager
                    // Check for auto-loaded checkpoint after server starts
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                        trainingVM.checkInitialState()
                    }
                }
                .onDisappear {
                    serverManager.stop()
                }
        }
        .commands {
            // Add standard Cmd+Q support
            CommandGroup(replacing: .appTermination) {
                Button("Quit Neural Razz Trainer") {
                    serverManager.stop()
                    NSApplication.shared.terminate(nil)
                }
                .keyboardShortcut("q")
            }
        }
    }
}
