"""
Launch the Interactive UFC Fight Predictor

Simple script to start the interactive fight prediction web app.
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("🥊 Starting Interactive UFC Fight Predictor...")
    
    # Check if required files exist
    if not Path("data/ufc-master.csv").exists():
        print("❌ Historical data not found (data/ufc-master.csv)")
        print("💡 Please ensure you have the UFC historical data file")
        return
    
    if not Path("models").exists() or not any(Path("models").glob("*.pkl")):
        print("❌ No trained models found in models/ directory")
        print("💡 Please train models first using: python -m scripts.train --model gbdt")
        return
    
    print("✅ Data and models found")
    print("\n🚀 Launching interactive fight predictor at http://localhost:8502")
    print("🥊 Select any two fighters to see who would win!")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    # Launch Streamlit
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8502")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "fight_predictor_app.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Fight predictor stopped. Thanks for using UFC Fight Predictor!")
    except Exception as e:
        print(f"❌ Error starting fight predictor: {e}")
        print("💡 Try running manually: streamlit run fight_predictor_app.py")

if __name__ == "__main__":
    main()