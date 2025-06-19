"""
Launch the UFC Fight Prediction Dashboard

Simple script to start the Streamlit web application.
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("🥊 Starting UFC Fight Prediction Dashboard...")
    
    # Check if prediction files exist
    if not Path("predictions.csv").exists():
        print("⚠️  predictions.csv not found. Running prediction scripts...")
        try:
            subprocess.run([sys.executable, "-m", "scripts.predict_upcoming"], check=True)
            print("✅ Logistic regression predictions generated")
        except subprocess.CalledProcessError:
            print("❌ Failed to generate logistic regression predictions")
    
    if not Path("predictions_gbdt.csv").exists():
        print("⚠️  predictions_gbdt.csv not found. Running gradient boosting predictions...")
        try:
            subprocess.run([sys.executable, "-m", "scripts.predict_with_best_model"], check=True)
            print("✅ Gradient boosting predictions generated")
        except subprocess.CalledProcessError:
            print("❌ Failed to generate gradient boosting predictions")
    
    print("\n🚀 Launching dashboard at http://localhost:8501")
    print("📱 The dashboard will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    # Launch Streamlit
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thanks for using UFC Fight Predictor!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Try running manually: streamlit run app.py")

if __name__ == "__main__":
    main()