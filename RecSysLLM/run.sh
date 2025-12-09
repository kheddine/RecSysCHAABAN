#!/bin/bash

# Spotify Mood Mixer Pro - Startup Script

echo "🎵 Starting Spotify Mood Mixer Pro..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python 3 found"

# Check dependencies
echo "📦 Installing dependencies..."
pip install flask flask-cors pandas numpy scikit-learn requests python-dotenv --break-system-packages -q 2>/dev/null || pip install flask flask-cors pandas numpy scikit-learn requests python-dotenv -q

echo "✓ Dependencies installed"
echo ""

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "🚀 Starting servers..."
echo ""

# Start backend in background
echo "Starting Backend (Port 5000)..."
python app_enhanced.py &
BACKEND_PID=$!

sleep 2

# Start frontend server in background
echo "Starting Frontend (Port 8000)..."
python -m http.server 8000 --directory . &
FRONTEND_PID=$!

sleep 2

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║        ✓ Spotify Mood Mixer Pro is RUNNING!  🎵           ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Open your browser:"
echo "   http://localhost:8000"
echo ""
echo "📝 Backend API:"
echo "   http://localhost:5000/api/health"
echo ""
echo "To stop: Press Ctrl+C"
echo ""

# Keep scripts running
wait
