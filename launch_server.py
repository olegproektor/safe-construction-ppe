#!/usr/bin/env python3
"""
Simple Server Launcher - Bypasses all command line issues
"""
import os
import sys
from pathlib import Path

def main():
    print("🏗️ PPE Detection System - Server Launcher")
    print("=" * 60)
    
    # Change to the correct directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    print(f"📁 Working directory: {project_dir}")
    
    # Check if minimal_app.py exists
    minimal_app = project_dir / "minimal_app.py"
    if not minimal_app.exists():
        print("❌ minimal_app.py not found!")
        print("Creating minimal app...")
        create_minimal_app()
    
    print("🚀 Starting server...")
    print("📍 URL: http://localhost:8080")
    print("📍 Demo: http://localhost:8080/demo")
    print("📍 Health: http://localhost:8080/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the minimal app
        import minimal_app
        # The minimal_app.py has if __name__ == "__main__" block that will run uvicorn
        
    except ImportError:
        print("❌ Failed to import minimal_app, creating it now...")
        create_minimal_app()
        import minimal_app
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running manually: python minimal_app.py")

def create_minimal_app():
    """Create minimal_app.py if it doesn't exist"""
    content = '''"""
Minimal FastAPI App - Web Interface Only
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="PPE Detection - Minimal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>PPE Detection System</title></head>
    <body style="font-family: Arial; background: #1a1a1a; color: white; padding: 40px; text-align: center;">
        <h1>🏗️ PPE Detection System</h1>
        <h2>✅ Server is Running!</h2>
        <p>The web interface is now accessible.</p>
        <div style="margin: 30px;">
            <a href="/health" style="color: #00ff00; margin: 10px;">📊 Health Check</a>
            <a href="/demo" style="color: #00ff00; margin: 10px;">🎥 Demo</a>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Server running successfully"}

@app.get("/demo")
async def demo():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Demo Page</title></head>
    <body style="font-family: Arial; background: #2a5298; color: white; padding: 40px; text-align: center;">
        <h1>🎥 Demo Page</h1>
        <p>Server is working correctly!</p>
        <p><a href="/" style="color: #ffd700;">← Back to Home</a></p>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
    
    with open("minimal_app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Created minimal_app.py")

if __name__ == "__main__":
    main()