#ngrok http --url=indirectly-credible-egret.ngrok-free.app 5000
import eventlet
eventlet.monkey_patch()

import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template_string, request, Response, jsonify, stream_with_context, session
import os
import sys
import json
import re
import subprocess
import time
from typing import List, Dict, Optional, Any
from io import BytesIO
from pathlib import Path

# --- GLOBAL LOGGING SETUP ---
LOG_FILE = 'zylo_activity.log'
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=50*1024*1024, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
))
file_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

app = Flask(__name__)
app.logger.addHandler(file_handler)
app.config['SECRET_KEY'] = os.environ.get("ZYLO_SECRET_KEY", "zylo_master_key_2026")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 4096 * 1024 * 1024  # 4GB max

def redact_sensitive(data):
    """Redact passwords and keys from logs."""
    if not isinstance(data, dict):
        return data
    redacted = data.copy()
    sensitive_keys = ['password', 'pass', 'api_key', 'key', 'secret', 'token', 'authorization']
    for k, v in redacted.items():
        if any(sk in k.lower() for sk in sensitive_keys):
            redacted[k] = "[REDACTED]"
        elif isinstance(v, dict):
            redacted[k] = redact_sensitive(v)
    return redacted

@app.before_request
def log_request_info():
    try:
        user_id = session.get('user_id', 'ANONYMOUS')
        payload = {}
        if request.is_json:
            payload = request.get_json(silent=True) or {}
        elif request.form:
            payload = request.form.to_dict()
        
        redacted_payload = redact_sensitive(payload)
        files = [f.filename for f in request.files.values()] if request.files else []
        
        app.logger.info(
            f"REQUEST | User: {user_id} | IP: {request.remote_addr} | "
            f"Method: {request.method} | URL: {request.url} | "
            f"Payload: {json.dumps(redacted_payload)} | Files: {files}"
        )
    except Exception as e:
        app.logger.error(f"Logging Error (before_request): {e}")

@app.after_request
def log_response_info(response):
    try:
        user_id = session.get('user_id', 'ANONYMOUS')
        # Only log first 500 chars of response if it's text/json
        content_preview = ""
        if response.direct_passthrough:
            content_preview = "[STREAMING/FILE]"
        elif response.mimetype in ['application/json', 'text/html', 'text/plain', 'text/event-stream']:
            try:
                content_preview = response.get_data(as_text=True)[:500]
            except Exception:
                content_preview = "[BINARY/NON-UTF8 DATA]"
        else:
            content_preview = f"[BINARY: {response.mimetype}]"
            
        app.logger.info(
            f"RESPONSE | User: {user_id} | Status: {response.status} | "
            f"URL: {request.url} | Content: {content_preview}..."
        )
    except Exception as e:
        app.logger.error(f"Logging Error (after_request): {e}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    # Log the full stack trace for any unhandled exception
    app.logger.error(f"UNHANDLED EXCEPTION | URL: {request.url} | Error: {str(e)}", exc_info=True)
    # Re-raise or return 500
    return "Internal Server Error", 500

# --- END GLOBAL LOGGING SETUP ---

from Cloud_Storage import cloud_bp
from ZYlO_RiG0R import rigor_bp
from chat import chat_bp, socketio
from ZYLOVEIL import veil_bp
import weather as zylo_feels

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from zhipuai import ZhipuAI
    HAS_ZHIPU = True
except ImportError:
    HAS_ZHIPU = False

# Register Blueprints
app.register_blueprint(cloud_bp)
app.register_blueprint(rigor_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(veil_bp, url_prefix='/veil')

# Import and register ZYLO ZENITH blueprint
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("AI_Zenith", os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI~Zenith.py"))
AI_Zenith = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AI_Zenith)
app.register_blueprint(AI_Zenith.zenith_bp)

# Initialize SocketIO
socketio.init_app(app)

LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>ZYLO | Next Gen</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #050505;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-shine: rgba(255, 255, 255, 0.1);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text-main: #ffffff;
            --text-muted: #9ca3af;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; -webkit-tap-highlight-color: transparent; }

        body {
            background-color: var(--bg-deep);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
            min-height: 100vh;
            min-height: 100dvh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: hidden;
            overflow-y: auto;
            color: var(--text-main);
        }

        .container {
            width: 100%;
            max-width: 1000px;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
            z-index: 1;
            margin: auto;
            position: relative;
        }

        .header {
            text-align: center;
            animation: fadeInDown 1s ease;
            flex-shrink: 0;
        }

        .logo {
            font-size: 4rem;
            font-weight: 800;
            letter-spacing: -2px;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
        }

        .subtitle {
            font-size: 1.1rem;
            color: var(--text-muted);
            font-weight: 300;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .cards {
            display: flex;
            gap: 30px;
            width: 100%;
            justify-content: center;
            flex-wrap: wrap;
            perspective: 1000px;
            flex-shrink: 0;
            margin-top: 6px;
        }

        .card-wrapper {
            position: relative;
            width: 280px;
            height: 320px;
            border-radius: 24px;
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.8s ease backwards;
        }

        .card-wrapper:nth-child(1) { animation-delay: 0.1s; }
        .card-wrapper:nth-child(2) { animation-delay: 0.2s; }
        .card-wrapper:nth-child(3) { animation-delay: 0.3s; }
        .card-wrapper:nth-child(4) { animation-delay: 0.4s; }

        .card-wrapper:hover {
            transform: translateY(-10px);
            z-index: 10;
        }

        .glass-card {
            width: 100%;
            height: 100%;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            overflow: hidden;
            position: relative;
            transition: all 0.3s ease;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            text-decoration: none;
            text-align: center; /* Ensure text is centered on desktop */
        }
        
        a.glass-card { text-decoration: none; }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
            z-index: 0;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .glass-card:hover {
            border-color: var(--accent);
            box-shadow: 0 0 30px var(--accent-glow);
        }

        .glass-card:hover::before { opacity: 1; }

        .icon-box {
            font-size: 3.5rem;
            margin-bottom: 20px;
            z-index: 1;
            background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: transform 0.3s ease;
        }

        .glass-card:hover .icon-box {
            transform: scale(1.1);
            background: linear-gradient(135deg, var(--accent) 0%, #fff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            z-index: 1;
            margin-bottom: 8px;
            color: var(--text-main);
        }

        .card-desc {
            font-size: 0.9rem;
            color: var(--text-muted);
            z-index: 1;
            text-align: center;
            padding: 0 20px;
        }

        .wide-card {
            width: auto;
            max-width: none;
            margin: 0;
            position: absolute;
            top: 6px;
            right: 8px;
            height: auto;
            z-index: 5;
        }
        .wide-card .glass-card {
            flex-direction: row;
            justify-content: center;
            align-items: center;
            padding: 0;
            gap: 0;
            text-align: left;
            height: auto;
            min-height: 0;
            background: transparent;
            border: none;
            box-shadow: none;
        }
        .feels-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .feels-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            font-weight: 600;
            color: var(--text-main);
            font-size: 0.85rem;
            line-height: 1;
        }
        .feels-pill img {
            width: 20px;
            height: 20px;
        }

        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        /* Background Orb */
        .orb {
            position: fixed;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: -1;
            pointer-events: none;
            animation: pulse 10s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.5; }
            50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.8; }
        }
        
        /* --- Mobile Adaptations --- */
        @media (max-width: 768px) {
            body { align-items: flex-start; }
            .logo { font-size: 2.5rem; }
            .subtitle { font-size: 0.9rem; }
            .container { padding: 28px 12px; gap: 12px; margin-top: 8px; }
            .header { margin-bottom: 10px; }
            .cards { gap: 12px; }
            .card-wrapper { width: 100%; max-width: 320px; height: 160px; margin: 0 auto; }
            .glass-card { flex-direction: row; padding: 0 20px; justify-content: flex-start; gap: 20px; text-align: left; }
            .icon-box { margin-bottom: 0; font-size: 2.5rem; }
            .card-title { font-size: 1.2rem; margin-bottom: 4px; }
            .card-desc { text-align: left; padding: 0; font-size: 0.8rem; }
            .wide-card { position: static; width: auto; align-self: flex-end; }
        }

        /* --- Modal Styles --- */
        .modal-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            z-index: 2000;
            display: flex; justify-content: center; align-items: center;
            opacity: 0; pointer-events: none;
            transition: opacity 0.3s ease;
        }
        
        .modal-overlay.active { opacity: 1; pointer-events: all; }
        
        .modal-content {
            background: rgba(20, 20, 25, 0.6);
            border: 1px solid var(--glass-border);
            padding: 40px;
            border-radius: 24px;
            max-width: 500px;
            width: 90%;
            transform: scale(0.9);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            position: relative;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        
        .modal-overlay.active .modal-content { transform: scale(1); }
        
        .modal-header { margin-bottom: 30px; text-align: center; }
        .modal-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 10px; }
        .modal-sub { color: var(--text-muted); font-size: 0.9rem; }
        
        .modal-close {
            position: absolute; top: 20px; right: 20px;
            background: transparent; border: none; color: var(--text-muted);
            font-size: 1.2rem; cursor: pointer; transition: 0.2s;
        }
        .modal-close:hover { color: white; transform: rotate(90deg); }
        
        .modal-options { display: flex; flex-direction: column; gap: 15px; }
        
        .option-card {
            display: flex; align-items: center; gap: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            text-decoration: none;
            color: white;
            transition: all 0.2s;
            position: relative;
            overflow: hidden;
        }
        
        .option-card:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent);
            transform: translateY(-2px);
        }
        
        .option-card.disabled { opacity: 0.6; cursor: not-allowed; filter: grayscale(1); }
        .option-card.disabled:hover { transform: none; border-color: var(--glass-border); background: rgba(255, 255, 255, 0.05); }
        
        .option-icon {
            width: 50px; height: 50px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            display: flex; justify-content: center; align-items: center;
            font-size: 1.5rem;
            color: var(--accent);
        }
        
        .option-info { flex: 1; }
        .option-name { font-weight: 700; font-size: 1.1rem; margin-bottom: 4px; }
        .option-detail { font-size: 0.8rem; color: var(--text-muted); }
        
        .tag {
            font-size: 0.6rem; font-weight: 800;
            padding: 4px 8px; border-radius: 6px;
            background: var(--accent); color: white;
            letter-spacing: 0.5px;
        }
        .tag.dev { background: #333; border: 1px solid #555; }
        
    </style>
</head>
<body>

    <div class="orb"></div>

    <div class="container">
        <div class="header">
            <div class="logo">ZYLO</div>
            <div class="subtitle" id="secret-trigger" style="cursor: default; user-select: none;">Unified Digital Intelligence</div>
        </div>

        <!-- ZYLO FEELS Wide Card -->
        <a href="/feels" class="card-wrapper wide-card">
            <div class="feels-pill">
                <img id="feels-icon" src="" alt="">
                <span id="feels-temp">--°</span>
            </div>
        </a>

        <div class="cards">
            
            <!-- AI Card -->
            <div class="card-wrapper" id="card-ai">
                <div class="glass-card" onclick="openAIModal()">
                    <div class="icon-box"><i class="fas fa-brain"></i></div>
                    <div>
                        <div class="card-title">ZYLO AI</div>
                        <div class="card-desc">Advanced reasoning & computation engine</div>
                    </div>
                </div>
            </div>

            <!-- Link Card -->
            <div class="card-wrapper">
                <a href="/link" class="glass-card">
                    <div class="icon-box"><i class="fas fa-link"></i></div>
                    <div>
                        <div class="card-title">ZYLO LINK</div>
                        <div class="card-desc">Secure encrypted real-time communication</div>
                    </div>
                </a>
            </div>

            <!-- Cloud Card -->
            <div class="card-wrapper">
                <a href="/cloud" class="glass-card">
                    <div class="icon-box"><i class="fas fa-cloud"></i></div>
                    <div>
                        <div class="card-title">ZYLO CLOUD</div>
                        <div class="card-desc">Premium storage vault</div>
                    </div>
                </a>
            </div>

        </div>
    </div>
    
    <!-- AI Selection Modal -->
    <div class="modal-overlay" id="ai-modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeAIModal()"><i class="fas fa-times"></i></button>
            <div class="modal-header">
                <div class="modal-title">Select Intelligence</div>
                <div class="modal-sub">Choose your specialized neural engine</div>
            </div>
            
            <div class="modal-options">
                <a href="/rigor" class="option-card">
                    <div class="option-icon"><i class="fas fa-microchip"></i></div>
                    <div class="option-info">
                        <div class="option-name">ZYLO RIGOR</div>
                        <div class="option-detail">Research, Math & Logic Engine</div>
                    </div>
                    <i class="fas fa-arrow-right" style="opacity:0.5;"></i>
                </a>

                <a href="/zenith" class="option-card">
                    <div class="option-icon"><i class="fas fa-brain"></i></div>
                    <div class="option-info">
                        <div class="option-name">ZYLO ZENITH</div>
                        <div class="option-detail">Premium Multi-Model AI with Search</div>
                    </div>
                    <i class="fas fa-arrow-right" style="opacity:0.5;"></i>
                </a>
            </div>
        </div>
    </div>

    <script>
        function openAIModal() {
            document.getElementById('ai-modal').classList.add('active');
        }
        
        function closeAIModal() {
            document.getElementById('ai-modal').classList.remove('active');
        }
        
        // Close on click outside
        document.getElementById('ai-modal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('ai-modal')) {
                closeAIModal();
            }
        });

        // Secret Gesture logic
        (function() {
            const trigger = document.getElementById('secret-trigger');
            let isMoving = false;
            let startX, startY;
            const threshold = 100; // pixels to move

            function handleStart(e) {
                isMoving = true;
                const touch = e.touches ? e.touches[0] : e;
                startX = touch.clientX;
                startY = touch.clientY;
            }

            function handleMove(e) {
                if (!isMoving) return;
                const touch = e.touches ? e.touches[0] : e;
                const dx = touch.clientX - startX;
                const dy = touch.clientY - startY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance > threshold) {
                    isMoving = false;
                    window.location.href = '/veil';
                }
            }

            function handleEnd() {
                isMoving = false;
            }

            trigger.addEventListener('mousedown', handleStart);
            window.addEventListener('mousemove', handleMove);
            window.addEventListener('mouseup', handleEnd);

            trigger.addEventListener('touchstart', handleStart);
            trigger.addEventListener('touchmove', handleMove);
            trigger.addEventListener('touchend', handleEnd);
        })();

        // ZYLO FEELS preview (temp + icon)
        (function() {
            const tempEl = document.getElementById('feels-temp');
            const iconEl = document.getElementById('feels-icon');
            if (!tempEl || !iconEl) return;
            fetch('/api/weather')
                .then(r => r.json())
                .then(d => {
                    if (!d || d.error) return;
                    const temp = d?.TopMetrics?.Temperature;
                    const icon = d?.Location?.Condition?.icon;
                    if (temp !== null && temp !== undefined) tempEl.textContent = `${temp}°`;
                    if (icon) iconEl.src = icon;
                })
                .catch(() => {});
        })();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(LANDING_PAGE)

@app.route('/feels')
def feels():
    return render_template_string(zylo_feels.HTML, refresh=zylo_feels.REFRESH_SECONDS)

@app.route('/api/weather')
def feels_api_weather():
    try:
        query = request.args.get("q") or None
        return jsonify(zylo_feels.fetch_weather(query=query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/speak", methods=["POST"])
def feels_api_speak():
    return zylo_feels.api_speak()

@app.route('/api/music', methods=['POST'])
def feels_api_music():
    try:
        payload = request.get_json(silent=True) or {}
        data = payload.get("weather") or zylo_feels.fetch_weather()
        return jsonify(zylo_feels.start_music_job(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/music/status')
def feels_api_music_status():
    try:
        job_key = (request.args.get("job_key") or "").strip()
        if not job_key:
            return jsonify({"error": "Missing job_key."}), 400
        return jsonify(zylo_feels.get_music_job_status(job_key))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("💎 ZYLO UNIFIED SERVER STARTED | PORT 5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

