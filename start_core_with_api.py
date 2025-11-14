#!/usr/bin/env python3
"""
Startup script for MaiM Bot Core with API Server
This script runs both the original bot and the API server concurrently
"""

import asyncio
import os
import sys
import subprocess
import signal
import time
import threading
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class CoreServiceManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_bot(self):
        """Start the original MaiM Bot"""
        print("Starting MaiM Bot...")
        process = subprocess.Popen(
            [sys.executable, "bot.py"],
            cwd=current_dir,
            stdout=sys.stdout,  # Forward to container stdout
            stderr=sys.stderr,  # Forward to container stderr
            text=True
        )
        self.processes.append(("Bot", process))
        return process
        
    def start_api_server(self):
        """Start the API server"""
        print("Starting Core API Server...")
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8002"],
            cwd=current_dir,
            stdout=sys.stdout,  # Forward to container stdout
            stderr=sys.stderr,  # Forward to container stderr
            text=True
        )
        self.processes.append(("API Server", process))
        return process
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down services...")
        self.running = False
        self.shutdown()
        
    def shutdown(self):
        """Shutdown all processes"""
        for name, process in self.processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
                
    def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        while self.running:
            for name, process in self.processes:
                if process.poll() is not None:
                    # Process has actually stopped
                    print(f"{name} has stopped (exit code: {process.returncode}), restarting...")
                    if name == "Bot":
                        self.start_bot()
                    elif name == "API Server":
                        print("API Server stopped, restarting...")
                        self.start_api_server()
                        time.sleep(2)  # Give API server time to start
                elif process.poll() is None:
                    # Process is still running, this is good
                    pass
            time.sleep(10)  # Increased sleep time to reduce false restarts
            
    def run(self):
        """Run all services"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start services
            self.start_bot()
            time.sleep(5)  # Give bot time to initialize
            self.start_api_server()
            
            print("All core services started successfully!")
            print("Bot running on port 8000")
            print("Core API Server running on port 8002")
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown()

def main():
    """Main entry point"""
    print("Starting MaiM Bot Core with API Server...",flush=True)
    
    # Set the working directory to the script location
    os.chdir(current_dir)
    
    # Create and run service manager
    manager = CoreServiceManager()
    manager.run()

if __name__ == "__main__":
    main() 