#!/usr/bin/env python3
"""
Startup script for MaiM Bot API Server
This script runs the FastAPI server that provides HTTP endpoints for the MaiM Bot
"""

import asyncio
import os
import sys
import uvicorn
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Start the API server"""
    print("Starting MaiM Bot API Server...")
    
    # Set the working directory to the script location
    os.chdir(current_dir)
    
    # Import and run the API server
    from src.api_server import app
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )

if __name__ == "__main__":
    main() 