#!/usr/bin/env python3
"""
ZYLO ZENITH FLASK CHAT SERVER - Premium UI Version
"""

import os
import sys
import re
import json
import mimetypes
import base64
import time
import subprocess
from typing import List, Dict, Optional, Any
import requests
from flask import Flask, render_template_string, request, Response, jsonify, stream_with_context, Blueprint
from openai import OpenAI
from io import BytesIO

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

# ====================== CONFIGURATION ======================
API_KEY = os.getenv("NVIDIA_API_KEY", "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")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA-P_5BvsOqJ8OKsT1wo9qep86Baaq0Vl0")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "642a5c77f75141ceb178ed3106bf8a83.ETf0547Pst0azRUL")

# NEW KEYS
OPENROUTER_API_KEY = "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"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
ZENROWS_API_KEY = os.getenv("ZENROWS_API_KEY", "52e3f8ac45c2e7601a919a42111df7dd7ca1b9b5")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-vwm5r6wnyrk2nkwyfnxdhc8yc5n3xy3tvjc6pehtmdwvpv56")
SCRAPERAPI_KEY = "00a3ca6f13d8a07a87512e583cc27b57"

# NVIDIA TTS (from voice.py)
TTS_API_KEY = os.getenv("NVIDIA_TTS_API_KEY", "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")
TTS_FUNCTION_ID = os.getenv("NVIDIA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
TTS_SERVER = os.getenv("NVIDIA_TTS_SERVER", "grpc.nvcf.nvidia.com:443")
# Use absolute path found via search to ensure it works regardless of CWD
TTS_TALK_SCRIPT = os.getenv("TTS_TALK_SCRIPT", "/home/ujan/python-clients/scripts/tts/talk.py")

# Voices mapped to Magpie-Multilingual styles (distinct)
VOICES = [
    {"id": "Magpie-Multilingual.EN-US.Leo.Neutral", "name": "Neutral Male", "lang": "en", "tld": "us"},
    {"id": "Magpie-Multilingual.EN-US.Mia.Neutral", "name": "Neutral Female", "lang": "en", "tld": "co.uk"},
    {"id": "Magpie-Multilingual.EN-US.Jason.Calm", "name": "Calm Narrator", "lang": "en", "tld": "ie"},
    {"id": "Magpie-Multilingual.EN-US.Aria.Happy", "name": "Energetic Female", "lang": "en", "tld": "com.au"},
    {"id": "Magpie-Multilingual.EN-US.Leo.Angry", "name": "Aggressive / Alert", "lang": "en", "tld": "ca"},
    {"id": "Magpie-Multilingual.EN-US.Mia.Sad", "name": "Soft / Sad", "lang": "en", "tld": "co.in"}
]

MODELS = [
    {"id": "openai/gpt-oss-120b", "name": "ChatGPT", "desc": "Powerful open model", "reasoning": True, "provider": "nvidia", "extra_body": None},
    {"id": "google/gemini-3-flash-preview", "name": "Gemini 3", "desc": "Multimodal - images, video, audio", "reasoning": True, "provider": "google", "extra_body": None},
    {"id": "z-ai/glm4.7", "name": "GLM-4.7", "desc": "Strong coding & math specialist", "reasoning": True, "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}, "temperature": 1, "top_p": 1, "max_tokens": 16384},
    {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5", "desc": "Advanced multimodal reasoning", "reasoning": True, "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}, "temperature": 1, "top_p": 1, "max_tokens": 16384},
    {"id": "deepseek-ai/deepseek-v3.2", "name": "DeepSeek V3.2", "desc": "General purpose reasoning", "reasoning": True, "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}, "temperature": 1, "top_p": 0.95, "max_tokens": 8192},
    {"id": "minimaxai/minimax-m2.1", "name": "MiniMax M2.1", "desc": "Efficient reasoning agent", "reasoning": True, "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
    {"id": "qwen/qwen3-coder-480b-a35b-instruct", "name": "Qwen3 Coder", "desc": "Specialized coding assistant", "reasoning": False, "provider": "nvidia", "extra_body": None},
    {"id": "meta/llama-3.1-405b-instruct", "name": "Llama 405b", "desc": "Largest open Llama", "reasoning": False, "provider": "nvidia", "extra_body": None},
    {"id": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "name": "Nemotron Ultra 253B", "desc": "Advanced Nemotron variant", "reasoning": True, "provider": "nvidia", "extra_body": {"chat_template_kwargs": {"thinking": True}}, "temperature": 0.6, "max_tokens": 4096},
    {"id": "mistralai/mistral-large-3-675b-instruct-2512", "name": "Mistral 675b", "desc": "High-performance multilingual", "reasoning": False, "provider": "nvidia", "extra_body": None}
]

# ====================== INITIALIZE CLIENTS ======================
print("\n" + chr(0x1F680) + " Initializing ZYLO ZENITH Chat Server...")

nvidia_client = None
gemini_client = None
zhipu_client = None
openrouter_client = None
cerebras_client = None

try:
    nvidia_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print(chr(0x2705) + " NVIDIA client initialized")
except Exception as e:
    print(chr(0x274C) + " NVIDIA client failed: " + str(e))

try:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    print(chr(0x2705) + " OpenRouter client initialized")
except Exception as e:
    print(chr(0x274C) + " OpenRouter client failed: " + str(e))

try:
    cerebras_client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=CEREBRAS_API_KEY,
    )
    print(chr(0x2705) + " Cerebras client initialized")
except Exception as e:
    print(chr(0x274C) + " Cerebras client failed: " + str(e))

def web_search(query):
    """Search the internet using Tavily or ZenRows."""
    try:
        # Try Tavily first
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5
            },
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get('results', [])
            return "\n".join([f"- {r['title']}: {r['content']} ({r['url']})" for r in results])
    except Exception as e:
        print(f"Tavily search failed: {e}")

    try:
        # Fallback to ZenRows (using it as a simple scraper/proxy for a search engine if Tavily fails)
        # Note: ZenRows is typically for scraping, here we assume it's used for a search URL
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        response = requests.get(
            "https://api.zenrows.com/v1/",
            params={"apikey": ZENROWS_API_KEY, "url": search_url},
            timeout=15
        )
        if response.status_code == 200:
            # Simple text extraction from HTML (very basic fallback)
            text = re.sub(r'<[^>]+>', '', response.text)
            return text[:2000] # Return first 2000 chars
    except Exception as e:
        print(f"ZenRows search failed: {e}")
    
    return "Search failed to return results."

def deep_search(query):
    """Deep Research using ScraperAPI structured Google search."""
    try:
        payload = { 'api_key': SCRAPERAPI_KEY, 'query': query }
        response = requests.get('https://api.scraperapi.com/structured/google/search/v1', params=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get('organic_results', [])
            results = []
            for r in organic_results[:8]: # Get up to 8 results for deep research
                title = r.get('title', 'No Title')
                snippet = r.get('snippet', 'No Snippet')
                link = r.get('link', '#')
                results.append(f"- {title}: {snippet} ({link})")
            
            if results:
                return "\n".join(results)
            return "No results found in Deep Research."
    except Exception as e:
        print(f"ScraperAPI deep search failed: {e}")
    
    return "Deep Research failed."

def fetch_url_content(url):
    """Fetch full page content using ScraperAPI to bypass blocks."""
    try:
        payload = { 'api_key': SCRAPERAPI_KEY, 'url': url, 'render': 'true' }
        response = requests.get('https://api.scraperapi.com/', params=payload, timeout=30)
        if response.status_code == 200:
            # Simple text extraction from HTML
            text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', response.text)
            text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:10000] # Return first 10k chars
    except Exception as e:
        print(f"ScraperAPI fetch failed for {url}: {e}")
    return None

if HAS_GEMINI:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(chr(0x2705) + " Gemini client initialized")
    except Exception as e:
        print(chr(0x274C) + " Gemini client failed: " + str(e))
else:
    print(chr(0x26A0) + " Gemini SDK not installed (optional)")

if HAS_ZHIPU:
    try:
        zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)
        print(chr(0x2705) + " Zhipu client initialized")
    except Exception as e:
        print(chr(0x274C) + " Zhipu client failed: " + str(e))
else:
    print(chr(0x26A0) + " ZhipuAI SDK not installed (optional)")

print(chr(0x1F4CD) + " Open http://127.0.0.1:5000 in your browser\n")

# ====================== FLASK BLUEPRINT ======================
zenith_bp = Blueprint('zenith', __name__)

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZYLO ZENITH</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --glass-bg: rgba(20, 20, 30, 0.6); /* More opaque for better contrast */
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-tertiary: #a855f7;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1; /* Lighter for better readability */
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --user-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --ai-gradient: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            --border-radius: 16px;
            --border-radius-sm: 12px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        /* SPLASH SCREEN */
        #zenith-splash {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100vh; /* Fallback */
            height: 100dvh;
            background: #020205;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: opacity 1s ease, visibility 1s;
            will-change: opacity;
        }
        .splash-content {
            text-align: center;
            width: 100%;
            max-width: 90%;
            padding: 20px;
        }
        .splash-typing {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem; /* Base size for mobile */
            font-weight: 600;
            color: #fff;
            margin-bottom: 1rem;
            min-height: 8rem; /* Enough height to prevent jumping */
            text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
            transition: font-size 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            word-wrap: break-word;
        }
        
        /* Larger size for the brand reveal */
        .splash-typing.brand-reveal {
            font-size: clamp(2.2rem, 10vw, 4rem);
            font-weight: 800;
            letter-spacing: -1px;
        }

        @media (min-width: 768px) {
            .splash-typing { font-size: 2rem; }
        }

        .splash-typing span {
            border-right: 3px solid var(--accent-primary);
            animation: blink 0.7s infinite;
            padding-right: 5px;
            margin-left: 2px;
        }
        @keyframes blink { 50% { border-color: transparent; } }
        
        .splash-loader {
            width: 200px;
            height: 2px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 20px auto;
            position: relative;
        }
        .splash-loader::after {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 0; height: 100%;
            background: var(--user-gradient);
            box-shadow: 0 0 15px var(--accent-glow);
            animation: load 2.5s ease-in-out forwards;
        }
        @keyframes load { to { width: 100%; } }

        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg-primary); color: var(--text-primary); min-height: 100vh; overflow: hidden; }
        .bg-animation { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; overflow: hidden; }
        .bg-animation::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.12) 0%, transparent 50%), radial-gradient(circle at 40% 40%, rgba(168, 85, 247, 0.08) 0%, transparent 40%); animation: bgFloat 20s ease-in-out infinite; }
        @keyframes bgFloat { 0%, 100% { transform: translate(0, 0) rotate(0deg); } 25% { transform: translate(2%, 2%) rotate(1deg); } 50% { transform: translate(0, 4%) rotate(0deg); } 75% { transform: translate(-2%, 2%) rotate(-1deg); } }
        .orb { position: absolute; border-radius: 50%; filter: blur(60px); opacity: 0.5; animation: orbFloat 15s ease-in-out infinite; }
        .orb-1 { width: 400px; height: 400px; background: var(--accent-primary); top: 10%; right: 10%; }
        .orb-2 { width: 300px; height: 300px; background: var(--accent-secondary); bottom: 20%; left: 5%; animation-delay: -5s; }
        .orb-3 { width: 250px; height: 250px; background: var(--accent-tertiary); top: 50%; left: 50%; animation-delay: -10s; }
        @keyframes orbFloat { 0%, 100% { transform: translate(0, 0) scale(1); } 33% { transform: translate(30px, -30px) scale(1.1); } 66% { transform: translate(-20px, 20px) scale(0.9); } }
        .container { display: flex; flex-direction: column; height: 100vh; max-width: 1400px; margin: 0 auto; padding: 20px; position: relative; z-index: 1; }
        
        /* HEADER - Added relative positioning and z-index for dropdown visibility */
        .header { background: var(--glass-bg); backdrop-filter: blur(20px); border: 1px solid var(--glass-border); border-radius: var(--border-radius); padding: 16px 24px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: var(--glass-shadow); position: relative; z-index: 100; }
        
        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 44px; height: 44px; background: var(--user-gradient); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 15px var(--accent-glow); }
        .logo-text { font-size: 1.4rem; font-weight: 700; background: var(--user-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .header-controls { display: flex; align-items: center; gap: 12px; }
        .model-selector { position: relative; }
        .model-btn { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: var(--border-radius-sm); padding: 10px 16px; color: var(--text-primary); font-family: inherit; font-size: 0.9rem; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: var(--transition); }
        .model-btn:hover { background: rgba(255, 255, 255, 0.08); border-color: var(--accent-primary); }
        
        /* DROPDOWN - Improved z-index and styling */
        .model-dropdown, .voice-dropdown { 
            position: fixed; 
            top: 80px; /* Default for desktop header area */
            right: 20px; 
            width: 320px; 
            background: #1a1a25; 
            border: 1px solid var(--glass-border); 
            border-radius: var(--border-radius); 
            box-shadow: 0 10px 40px rgba(0,0,0,0.5); 
            z-index: 2001; 
            max-height: 400px; 
            overflow-y: auto; 
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.2s, visibility 0.2s;
            pointer-events: none;
        }
        .voice-dropdown { width: 200px; top: 80px; right: 350px; }
        .model-dropdown.show, .voice-dropdown.show { 
            visibility: visible !important; 
            opacity: 1 !important; 
            pointer-events: all !important; 
        }
        
        #dropdownOverlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(4px);
            z-index: 2000;
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s;
        }
        #dropdownOverlay.show { visibility: visible !important; opacity: 1 !important; }
        .model-option, .voice-option { padding: 16px 20px; cursor: pointer; transition: var(--transition); border-bottom: 1px solid var(--glass-border); }
        .model-option:last-child, .voice-option:last-child { border-bottom: none; }
        .model-option:hover, .voice-option:hover { background: rgba(99, 102, 241, 0.1); }
        .model-option.selected, .voice-option.selected { background: rgba(99, 102, 241, 0.15); border-left: 3px solid var(--accent-primary); }
        
        .voice-selector { position: relative; }
        .model-name { font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
        .model-desc { font-size: 0.8rem; color: var(--text-secondary); }
        .reasoning-badge { font-size: 0.65rem; padding: 2px 6px; background: var(--user-gradient); border-radius: 4px; color: white; font-weight: 600; }
        
        .toggle-btn { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: var(--border-radius-sm); padding: 10px 14px; color: var(--text-secondary); font-family: inherit; font-size: 0.85rem; cursor: pointer; transition: var(--transition); }
        .toggle-btn:hover:not(:disabled) { background: rgba(255, 255, 255, 0.08); color: var(--text-primary); }
        .toggle-btn.active { background: rgba(99, 102, 241, 0.2); border-color: var(--accent-primary); color: var(--accent-primary); }
        .toggle-btn:disabled { opacity: 0.5; cursor: not-allowed; border-color: transparent; }
        
        .chat-container { flex: 1; background: var(--glass-bg); backdrop-filter: blur(20px); border: 1px solid var(--glass-border); border-radius: var(--border-radius); display: flex; flex-direction: column; overflow: hidden; box-shadow: var(--glass-shadow); position: relative; z-index: 1; }
        .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; padding-bottom: 40px; }
        .messages::-webkit-scrollbar { width: 6px; }
        .messages::-webkit-scrollbar-track { background: transparent; }
        .messages::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 3px; }
        
        .message { display: flex; gap: 14px; max-width: 85%; position: relative; group; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message-avatar { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
        .message.user .message-avatar { background: var(--user-gradient); }
        .message.assistant .message-avatar { background: var(--ai-gradient); border: 1px solid var(--glass-border); }
        
        .message-content-wrapper { display: flex; flex-direction: column; gap: 6px; width: 100%; }
        .message-content { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: var(--border-radius); padding: 16px 20px; line-height: 1.6; word-wrap: break-word; position: relative; }
        .message.user .message-content { background: var(--user-gradient); border: none; color: white; }
        
        /* File Preview in Message */
        .message-files { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
        .file-attachment { background: rgba(0, 0, 0, 0.2); border: 1px solid var(--glass-border); border-radius: 8px; padding: 8px 12px; font-size: 0.85rem; display: flex; align-items: center; gap: 8px; color: var(--text-primary); max-width: 100%; }
        .file-icon { color: var(--accent-secondary); font-size: 1.1em; }
        
        /* Message Actions */
        .message-actions { display: flex; gap: 8px; margin-top: 4px; opacity: 0; transition: opacity 0.2s; align-self: flex-start; }
        .message.user .message-actions { align-self: flex-end; flex-direction: row-reverse; }
        .message:hover .message-actions { opacity: 1; }
        .action-btn { background: transparent; border: none; color: var(--text-muted); cursor: pointer; padding: 4px; border-radius: 4px; transition: var(--transition); display: flex; align-items: center; justify-content: center; }
        .action-btn:hover { color: var(--text-primary); background: rgba(255,255,255,0.1); }
        .action-btn svg { width: 14px; height: 14px; }
        
        /* Markdown Styles */
        .message-content h1, .message-content h2, .message-content h3 { margin-top: 16px; margin-bottom: 8px; font-weight: 700; color: var(--text-primary); }
        .message-content h1 { font-size: 1.5em; border-bottom: 1px solid var(--glass-border); padding-bottom: 4px; }
        .message-content h2 { font-size: 1.3em; }
        .message-content h3 { font-size: 1.1em; }
        .message-content p { margin-bottom: 10px; }
        .message-content p:last-child { margin-bottom: 0; }
        .message-content strong { color: #fff; font-weight: 700; }
        .message-content em { font-style: italic; color: #e2e8f0; }
        .message-content ul, .message-content ol { margin-left: 20px; margin-bottom: 10px; }
        .message-content table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
        .message-content th, .message-content td { padding: 8px 12px; border: 1px solid var(--glass-border); text-align: left; }
        .message-content th { background: rgba(255, 255, 255, 0.05); font-weight: 600; }
        .message-content hr { border: none; border-top: 1px solid var(--glass-border); margin: 16px 0; }
        
        /* Code block styles */
        .code-block-wrapper { position: relative; margin: 12px 0; }
        .code-block-header { display: flex; justify-content: space-between; align-items: center; background: #1e1e2e; border: 1px solid var(--glass-border); border-bottom: none; border-radius: var(--border-radius-sm) var(--border-radius-sm) 0 0; padding: 8px 12px; font-size: 0.75rem; color: var(--text-muted); }
        .code-lang { font-family: 'JetBrains Mono', monospace; }
        .copy-btn { background: transparent; border: 1px solid var(--glass-border); border-radius: 6px; padding: 4px 10px; color: var(--text-secondary); font-size: 0.75rem; cursor: pointer; display: flex; align-items: center; gap: 4px; transition: var(--transition); }
        .copy-btn:hover { background: var(--accent-primary); color: white; border-color: var(--accent-primary); }
        .copy-btn.copied { background: var(--success); border-color: var(--success); color: white; }
        .code-block { background: #1e1e2e; border: 1px solid var(--glass-border); border-top: none; border-radius: 0 0 var(--border-radius-sm) var(--border-radius-sm); padding: 16px; overflow-x: auto; margin: 0; }
        .code-block code { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #e2e8f0; white-space: pre; }
        .message-content code:not(.code-block code) { background: rgba(99, 102, 241, 0.2); padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }
        
        /* Thinking section styles */
        .thinking-section { background: rgba(20, 20, 30, 0.6); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: var(--border-radius-sm); margin-bottom: 14px; overflow: hidden; }
        .thinking-header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; cursor: pointer; user-select: none; transition: var(--transition); background: rgba(139, 92, 246, 0.08); }
        .thinking-header:hover { background: rgba(139, 92, 246, 0.15); }
        .thinking-title { display: flex; align-items: center; gap: 8px; font-weight: 600; color: #a78bfa; font-size: 0.9rem; }
        .thinking-title .spinner { width: 14px; height: 14px; border: 2px solid rgba(167, 139, 250, 0.3); border-top-color: #a78bfa; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .thinking-title .check { color: var(--success); font-size: 16px; }
        .thinking-toggle { background: none; border: none; color: #cbd5e1; cursor: pointer; padding: 4px; display: flex; align-items: center; transition: transform 0.3s; }
        .thinking-toggle.expanded { transform: rotate(180deg); }
        .thinking-content { max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; background: rgba(0, 0, 0, 0.2); }
        .thinking-content.expanded { max-height: 500px; overflow-y: auto; }
        .thinking-content-inner { padding: 16px; font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; font-family: 'JetBrains Mono', monospace; opacity: 0.9; }
        
        .input-area { padding: 20px 24px; border-top: 1px solid var(--glass-border); background: rgba(0, 0, 0, 0.3); display: flex; flex-direction: column; gap: 10px; }
        
        /* File Preview */
        .file-preview { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px; }
        .file-chip { background: rgba(255, 255, 255, 0.1); border: 1px solid var(--glass-border); border-radius: 8px; padding: 6px 12px; font-size: 0.8rem; display: flex; align-items: center; gap: 8px; color: var(--text-secondary); }
        .file-remove { cursor: pointer; color: var(--text-muted); font-weight: bold; }
        .file-remove:hover { color: var(--error); }

        .input-wrapper { display: flex; gap: 12px; align-items: flex-end; position: relative; }
        .input-container { flex: 1; }
        .chat-input { width: 100%; background: var(--bg-tertiary); border: 1px solid var(--glass-border); border-radius: var(--border-radius); padding: 16px 20px; color: var(--text-primary); font-family: inherit; font-size: 1rem; resize: none; min-height: 56px; max-height: 200px; transition: var(--transition); }
        .chat-input:focus { outline: none; border-color: var(--accent-primary); box-shadow: 0 0 0 3px var(--accent-glow); }
        .chat-input::placeholder { color: var(--text-muted); }
        
        .attach-btn { background: transparent; border: 1px solid var(--glass-border); border-radius: 12px; padding: 14px; color: var(--text-secondary); cursor: pointer; transition: var(--transition); display: none; }
        .attach-btn:hover { background: rgba(255, 255, 255, 0.1); color: var(--text-primary); }

        .media-btn { background: transparent; border: 1px solid var(--glass-border); border-radius: 12px; padding: 14px; color: var(--text-secondary); cursor: pointer; transition: var(--transition); display: flex; align-items: center; justify-content: center; }
        .media-btn:hover { background: rgba(255, 255, 255, 0.1); color: var(--text-primary); border-color: var(--accent-primary); }

        .send-btn { background: var(--user-gradient); border: none; border-radius: var(--border-radius); padding: 16px 24px; color: white; font-family: inherit; font-size: 1rem; font-weight: 600; cursor: pointer; transition: var(--transition); box-shadow: 0 4px 15px var(--accent-glow); }
        .send-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 25px var(--accent-glow); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        
        .loading-dots { display: inline-flex; gap: 4px; }
        .loading-dots span { width: 8px; height: 8px; background: var(--accent-primary); border-radius: 50%; animation: bounce 1.4s ease-in-out infinite; }
        .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; } 40% { transform: scale(1.2); opacity: 1; } }
        
        .welcome { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 40px; }
        .welcome-icon { width: 80px; height: 80px; background: var(--user-gradient); border-radius: 24px; display: flex; align-items: center; justify-content: center; font-size: 36px; margin-bottom: 24px; box-shadow: 0 8px 30px var(--accent-glow); animation: float 3s ease-in-out infinite; }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        .welcome h2 { font-size: 1.8rem; margin-bottom: 12px; background: var(--user-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .welcome p { color: var(--text-secondary); font-size: 1.1rem; max-width: 500px; }
        
        .suggestions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 32px; justify-content: center; }
        .suggestion { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: var(--border-radius-sm); padding: 12px 18px; color: var(--text-secondary); cursor: pointer; transition: var(--transition); font-size: 0.9rem; }
        .suggestion:hover { background: rgba(99, 102, 241, 0.1); border-color: var(--accent-primary); color: var(--text-primary); transform: translateY(-2px); }
        
        .clear-btn { background: transparent; border: 1px solid var(--glass-border); border-radius: var(--border-radius-sm); padding: 10px 14px; color: var(--text-muted); font-family: inherit; font-size: 0.85rem; cursor: pointer; transition: var(--transition); }
        .clear-btn:hover { border-color: var(--error); color: var(--error); }
        
        .status-indicator { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: var(--text-muted); margin-top: 8px; }
        .status-dot { width: 6px; height: 6px; background: var(--success); border-radius: 50%; }
        
        /* Tools Menu Styles */
        .tools-container { position: relative; display: flex; align-items: center; }
        .tools-bubbles { 
            position: absolute; 
            bottom: calc(100% + 15px); 
            left: 50%; 
            transform: translateX(-50%) translateY(20px) scale(0.5); 
            display: flex; 
            flex-direction: column; 
            gap: 12px; 
            opacity: 0; 
            pointer-events: none; 
            transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55); 
            z-index: 10;
        }
        .tools-bubbles.show { 
            opacity: 1; 
            pointer-events: all; 
            transform: translateX(-50%) translateY(0) scale(1); 
        }
        .bubble { 
            background: var(--bg-tertiary) !important; 
            border: 1px solid var(--glass-border) !important;
            border-radius: 50% !important; 
            width: 44px !important; 
            height: 44px !important; 
            padding: 0 !important; 
            display: flex !important; 
            align-items: center !important; 
            justify-content: center !important;
            box-shadow: var(--glass-shadow);
            transition: var(--transition) !important;
        }
        .bubble:hover { background: var(--glass-bg) !important; border-color: var(--accent-primary) !important; transform: scale(1.1); }
        .bubble.active { background: rgba(99, 102, 241, 0.2) !important; border-color: var(--accent-primary) !important; color: var(--accent-primary) !important; }
        #toolsToggle { transition: transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55) !important; }
        #toolsToggle.active { transform: rotate(90deg); }

        .desktop-text { display: inline; }
        .mobile-text { display: none; }
        
        @media (max-width: 768px) { 
            .desktop-text { display: none; }
            .mobile-text { display: inline; }
            .container { padding: 0; height: 100dvh; }
            .header { 
                padding: 10px; 
                margin-bottom: 0; 
                border-radius: 0; 
                border-bottom: 1px solid var(--glass-border);
                flex-direction: column;
                gap: 8px;
                flex-shrink: 0;
            }
            body.dropdown-open { overflow: hidden !important; position: fixed; width: 100%; }
            .logo { width: 100%; justify-content: center; }
            .logo-icon { width: 32px; height: 32px; font-size: 16px; }
            .logo-text { font-size: 1.1rem; }
            
            .header-controls { 
                width: 100%; 
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 6px;
                padding: 4px 0;
            }
            
            .voice-selector, .model-selector { flex: 1; min-width: 0; }
            .model-btn, .voice-btn { width: 100%; justify-content: center; overflow: hidden; text-overflow: ellipsis; }
            
            .toggle-btn, .clear-btn { flex: 0 0 auto; }
            
            /* Bottom Sheet style for dropdowns */
            .model-dropdown, .voice-dropdown { 
                position: fixed !important; 
                top: auto !important; 
                bottom: 0 !important; 
                left: 0 !important; 
                right: 0 !important; 
                width: 100% !important; 
                max-width: none !important;
                margin: 0 !important;
                border-radius: 24px 24px 0 0 !important;
                max-height: 70vh !important;
                z-index: 2001 !important;
                box-shadow: 0 -10px 40px rgba(0,0,0,0.8) !important;
                animation: slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                pointer-events: all !important;
                visibility: hidden;
                opacity: 0;
            }
            .model-dropdown.show, .voice-dropdown.show { 
                visibility: visible !important; 
                opacity: 1 !important; 
            }
            
            #dropdownOverlay {
                z-index: 2000 !important;
                visibility: hidden;
                opacity: 0;
            }
            #dropdownOverlay.show { 
                visibility: visible !important; 
                opacity: 1 !important; 
            }
            
            .chat-container { border-radius: 0; border: none; }
            .messages { padding: 12px; gap: 16px; }
            .message { max-width: 92%; }
            .message-content { padding: 12px 14px; font-size: 0.95rem; }
            
            .input-area { 
                padding: 8px 12px env(safe-area-inset-bottom);
                background: var(--bg-secondary);
            }
            .input-wrapper { gap: 8px; }
            .media-btn { 
                width: 40px; 
                height: 40px; 
                padding: 0; 
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .send-btn { 
                width: 40px; 
                height: 40px; 
                padding: 0; 
                border-radius: 50%; 
                font-size: 1.1rem; 
                display: flex; 
                align-items: center; 
                justify-content: center;
            }
            .chat-input { padding: 10px 14px; font-size: 0.95rem; }
            
            .message-actions { 
                opacity: 1; 
                background: rgba(0,0,0,0.4);
                padding: 4px;
                border-radius: 8px;
                margin-top: 8px;
            }
            
            .welcome { padding: 20px; }
            .welcome h2 { font-size: 1.4rem; }
            .suggestion { padding: 10px 14px; font-size: 0.85rem; }
        }
    </style>
</head>
<body>
    <div id="zenith-splash">
        <div class="splash-content">
            <div class="splash-typing" id="splash-typing"></div>
            <div class="splash-loader"></div>
        </div>
    </div>
    <div class="bg-animation">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>
    <div class="container">
        <header class="header">
            <div class="logo">
                <div class="logo-icon">&#9889;</div>
                <span class="logo-text">ZYLO ZENITH</span>
            </div>
            <div class="header-controls">
                <div class="voice-selector">
                    <button class="model-btn" id="voiceBtn" type="button" style="padding: 10px 12px;">
                        <span id="currentVoiceName">Male (US)</span>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                    </button>
                </div>
                <div class="model-selector">
                    <button class="model-btn" id="modelBtn" type="button">
                        <span id="currentModelName">ChatGPT</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6,9 12,15 18,9"></polyline></svg>
                    </button>
                </div>
                <button class="toggle-btn active" id="reasoningToggle" type="button">
                    <span class="desktop-text">&#128161; Reasoning</span>
                    <span class="mobile-text">&#128161;</span>
                </button>
                <button class="clear-btn" id="clearBtn" type="button">
                    <span class="desktop-text">&#128465; Clear</span>
                    <span class="mobile-text">&#128465;</span>
                </button>
            </div>
        </header>
        <div id="dropdownOverlay"></div>
        <main class="chat-container">
            <div class="messages" id="messagesContainer">
                <div class="welcome" id="welcomeScreen">
                    <div class="welcome-icon">&#128640;</div>
                    <h2>Welcome to ZYLO ZENITH</h2>
                    <p>Experience powerful AI models with reasoning capabilities, code generation, and more.</p>
                    <div class="suggestions" id="suggestionsContainer"></div>
                </div>
            </div>
            <div class="input-area">
                <div class="file-preview" id="filePreview"></div>
                <div class="input-wrapper">
                    <input type="file" id="fileInput" multiple style="display: none;">
                    <button class="attach-btn" id="attachBtn" type="button" title="Attach Files">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
                    </button>
                    <div class="input-container">
                        <textarea class="chat-input" id="chatInput" placeholder="Type your message..." rows="1"></textarea>
                    </div>
                    <div class="tools-container" id="toolsContainer">
                        <div class="tools-bubbles" id="toolsBubbles">
                            <button class="media-btn bubble" id="deepResearchBtn" type="button" title="Deep Research">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
                            </button>
                            <button class="media-btn bubble" id="speakInputBtn" type="button" title="Speak Text">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
                            </button>
                            <button class="media-btn bubble" id="imageBtn" type="button" title="Generate Image">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                            </button>
                            <button class="media-btn bubble" id="videoBtn" type="button" title="Generate Video">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg>
                            </button>
                        </div>
                        <button class="media-btn" id="toolsToggle" type="button" title="Tools">
                            <svg id="toolsIcon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                            <svg id="closeToolsIcon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        </button>
                    </div>
                    <button class="send-btn" id="sendBtn" type="button">
                        <span class="desktop-text">Send &#10148;</span>
                        <span class="mobile-text">&#10148;</span>
                    </button>
                </div>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Connected</span>
                </div>
            </div>
        </main>
    </div>

    <div id="modelDropdown" class="model-dropdown"></div>
    <div id="voiceDropdown" class="voice-dropdown"></div>

    <script>
    (function() {
        'use strict';
        
        // Splash Screen Logic
        function initSplash() {
            const splash = document.getElementById('zenith-splash');
            const typingEl = document.getElementById('splash-typing');
            const phrases = [
                "Experience the Ultimate Intelligence",
                "ZYLO ZENITH"
            ];
            let phraseIndex = 0;
            let charIndex = 0;
            let isDeleting = false;

            function typeEffect() {
                const currentPhrase = phrases[phraseIndex];
                
                // Toggle responsive class for brand reveal
                if (phraseIndex === 1) {
                    typingEl.classList.add('brand-reveal');
                } else {
                    typingEl.classList.remove('brand-reveal');
                }

                if (!isDeleting) {
                    // Typing phase
                    typingEl.innerHTML = currentPhrase.substring(0, charIndex) + '<span></span>';
                    if (charIndex < currentPhrase.length) {
                        charIndex++;
                        setTimeout(typeEffect, 40 + Math.random() * 60);
                    } else {
                        // Phrase complete
                        if (phraseIndex < phrases.length - 1) {
                            // Wait before deleting
                            setTimeout(() => {
                                isDeleting = true;
                                typeEffect();
                            }, 300);
                        } else {
                            // Final reveal, fade out splash
                            setTimeout(() => {
                                splash.style.opacity = '0';
                                setTimeout(() => {
                                    splash.style.visibility = 'hidden';
                                }, 1000);
                            }, 1500);
                        }
                    }
                } else {
                    // Deleting phase (Backspace)
                    typingEl.innerHTML = currentPhrase.substring(0, charIndex) + '<span></span>';
                    if (charIndex > 0) {
                        charIndex--;
                        // Very fast backspace
                        setTimeout(typeEffect, 20);
                    } else {
                        // Deletion finished, move to next phrase
                        isDeleting = false;
                        phraseIndex++;
                        charIndex = 0;
                        setTimeout(typeEffect, 500);
                    }
                }
            }

            // Start typing after a short delay
            setTimeout(typeEffect, 500);
        }
        
        // Run splash
        initSplash();
        
        // State
        var state = {
            currentModelIndex: 0,
            currentVoiceId: 'Magpie-Multilingual.EN-US.Leo.Neutral',
            showReasoning: true,
            deepResearch: false,
            conversationHistory: [],
            isStreaming: false,
            files: []
        };
        
        // Models & Voices
        var models = {{ models_json | safe }};
        var voices = {{ voices_json | safe }};
        
        // Suggestions
        var suggestions = [
            'Explain quantum computing simply',
            'Write a Python function to sort a list',
            'What are the best practices for REST APIs?'
        ];
        
        // DOM Elements
        var elements = {};
        
        // Parse text with improved Markdown formatting
        function parseText(text) {
            if (!text) return '';
            
            // 1. Escape HTML first (security)
            text = text.replace(/&/g, '&amp;')
                       .replace(/</g, '&lt;')
                       .replace(/>/g, '&gt;');
            
            // 1.1 Restore safe <br> tags
            text = text.replace(/&lt;br\s*\/?&gt;/gim, '<br>');
            text = text.replace(/&lt;ul&gt;/gim, '<ul>');
            text = text.replace(/&lt;\/ul&gt;/gim, '</ul>');
            text = text.replace(/&lt;li&gt;/gim, '<li>');
            text = text.replace(/&lt;\/li&gt;/gim, '</li>');
            
            // 2. Code blocks (extract to prevent collision)
            var codeBlocks = [];
            text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, function(match, lang, code) {
                var id = 'CODEBLOCK_' + codeBlocks.length;
                codeBlocks.push({ lang: lang || 'code', code: code, match: match });
                return id;
            });

            // 3. Inline code (extract)
            var inlineCodes = [];
            text = text.replace(/`([^`]+)`/g, function(match, code) {
                var id = 'INLINECODE_' + inlineCodes.length;
                inlineCodes.push({ code: code });
                return id;
            });

            // 4. Headers (ordered from longest to shortest to avoid partial matches)
            text = text.replace(/(^|\s)######\s+(.*$)/gim, '$1<h6>$2</h6>');
            text = text.replace(/(^|\s)#####\s+(.*$)/gim, '$1<h5>$2</h5>');
            text = text.replace(/(^|\s)####\s+(.*$)/gim, '$1<h4>$2</h4>');
            text = text.replace(/(^|\s)###\s+(.*$)/gim, '$1<h3>$2</h3>');
            text = text.replace(/(^|\s)##\s+(.*$)/gim, '$1<h2>$2</h2>');
            text = text.replace(/(^|\s)#\s+(.*$)/gim, '$1<h1>$2</h1>');

            // 5. Bold, Italic, Bold+Italic
            // ***bolditalic***
            text = text.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
            // **bold**
            text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            // *italic*
            text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

            // 5.1 Horizontal Rule
            text = text.replace(/^---$/gm, '<hr>');

            // 6. Tables
            var lines = text.split('\n');
            var inTable = false;
            var tableHtml = '';
            var resultLines = [];
            
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i].trim();
                if (line.startsWith('|')) {
                    if (!inTable) {
                        inTable = true;
                        tableHtml = '<table><thead>';
                        var cols = line.split('|').filter(function(c) { return c.trim() !== ''; });
                        tableHtml += '<tr>';
                        for (var j = 0; j < cols.length; j++) {
                            tableHtml += '<th>' + cols[j].trim() + '</th>';
                        }
                        tableHtml += '</tr></thead><tbody>';
                    } else {
                        if (line.match(/\|[\s-]+\|/)) { continue; }
                        var cols = line.split('|').filter(function(c) { return c.trim() !== ''; });
                        tableHtml += '<tr>';
                        for (var j = 0; j < cols.length; j++) {
                            tableHtml += '<td>' + cols[j].trim() + '</td>';
                        }
                        tableHtml += '</tr>';
                    }
                } else {
                    if (inTable) {
                        inTable = false;
                        tableHtml += '</tbody></table>';
                        resultLines.push(tableHtml);
                        resultLines.push(line);
                    } else {
                        resultLines.push(line);
                    }
                }
            }
            if (inTable) {
                 tableHtml += '</tbody></table>';
                 resultLines.push(tableHtml);
            }
            text = resultLines.join('\n');

            // 7. Lists
            text = text.replace(/^\s*-\s+(.*)$/gim, '<ul><li>$1</li></ul>');
            text = text.replace(/<\/ul>\n<ul>/g, '');

            // 8. Restore Code Blocks
            codeBlocks.forEach(function(block, index) {
                var codeId = 'code-' + Math.random().toString(36).substr(2, 9);
                var html = '<div class="code-block-wrapper">' +
                    '<div class="code-block-header">' +
                    '<span class="code-lang">' + block.lang + '</span>' +
                    '<button class="copy-btn" onclick="copyCode(\'' + codeId + '\')" type="button">' +
                    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>' +
                    '<span>Copy</span></button></div>' +
                    '<pre class="code-block"><code id="' + codeId + '">' + block.code.trim() + '</code></pre></div>';
                text = text.replace('CODEBLOCK_' + index, html);
            });

            // 9. Restore Inline Code
            inlineCodes.forEach(function(block, index) {
                text = text.replace('INLINECODE_' + index, '<code>' + block.code + '</code>');
            });

            // 10. Simple paragraphs
            text = text.replace(/\n\n/g, '</p><p>');
            text = '<p>' + text.replace(/\n/g, '<br>') + '</p>';
            
            // Cleanup
            text = text.replace(/<p><div/g, '<div').replace(/div><\/p>/g, 'div>')
                       .replace(/<p><ul>/g, '<ul>').replace(/<\/ul><\/p>/g, '</ul>')
                       .replace(/<p><table>/g, '<table>').replace(/<\/table><\/p>/g, '</table>')
                       .replace(/<p><hr><\/p>/g, '<hr>');
            
            return text;
        }
        
        // Copy code (global)
        window.copyCode = function(codeId) {
            var codeEl = document.getElementById(codeId);
            if (!codeEl) return;
            navigator.clipboard.writeText(codeEl.textContent).then(function() {
                var btn = codeEl.closest('.code-block-wrapper').querySelector('.copy-btn');
                btn.classList.add('copied');
                btn.innerHTML = '<span>Copied!</span>';
                setTimeout(function() {
                    btn.classList.remove('copied');
                    btn.innerHTML = '<span>Copy</span>';
                }, 2000);
            });
        };
        
        // Copy Message Content
        window.copyMessage = function(btn) {
            var msgContent = btn.closest('.message-content-wrapper').querySelector('.message-content');
            var text = msgContent.innerText; // Get text content
            navigator.clipboard.writeText(text).then(function() {
                var original = btn.innerHTML;
                btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                setTimeout(function() { btn.innerHTML = original; }, 2000);
            });
        };

        // Regenerate Message
        window.regenerateMessage = function() {
            if (state.isStreaming || state.conversationHistory.length === 0) return;
            
            // 1. Remove last assistant message from History
            var lastMsg = state.conversationHistory[state.conversationHistory.length - 1];
            if (lastMsg.role === 'assistant') {
                state.conversationHistory.pop();
                // Remove from DOM
                var messages = elements.messagesContainer.querySelectorAll('.message.assistant');
                if (messages.length > 0) {
                    messages[messages.length - 1].remove();
                }
            }
            
            // 2. Trigger send (resends last user message which is now at top of stack, but logic usually requires input)
            // Ideally we need to grab the last user message text.
            var lastUserMsg = state.conversationHistory[state.conversationHistory.length - 1];
            if (lastUserMsg && lastUserMsg.role === 'user') {
                // We don't pop the user message, we just re-run the "send" logic but skip adding the user message again?
                // Actually sendMessage adds the user message. So we should NOT use sendMessage directly.
                // We need a lower level function or just manually trigger fetch.
                // EASIER: Pop user message too, put in input, click send.
                state.conversationHistory.pop();
                 var userMessages = elements.messagesContainer.querySelectorAll('.message.user');
                if (userMessages.length > 0) {
                    userMessages[userMessages.length - 1].remove();
                }
                elements.chatInput.value = lastUserMsg.content;
                sendMessage();
            }
        };

        // Edit Message
        window.editMessage = function(btn) {
             if (state.isStreaming) return;
             
             // Identify message index
             var messageDiv = btn.closest('.message');
             var allMessages = Array.from(elements.messagesContainer.children).filter(el => el.classList.contains('message'));
             var index = allMessages.indexOf(messageDiv);
             
             if (index === -1) return;
             
             // Get the prompt text
             var contentDiv = messageDiv.querySelector('.message-content');
             var text = contentDiv.innerText; // or store raw text somewhere
             
             // Populate input
             elements.chatInput.value = text;
             elements.chatInput.focus();
             elements.chatInput.style.height = 'auto';
             elements.chatInput.style.height = elements.chatInput.scrollHeight + 'px';
             
             // Truncate history and DOM
             // History index mapping is tricky because welcome screen is not in history.
             // We assume 1-to-1 mapping if welcome screen is gone.
             
             // Remove from DOM: this message and everything after
             for (var i = allMessages.length - 1; i >= index; i--) {
                 allMessages[i].remove();
             }
             
             // Remove from history:
             // History has {role, content}. 
             // If index is 0 (first user msg), remove everything.
             state.conversationHistory.splice(index);
        };
        
        // Generate Media (Image/Video)
        window.generateMedia = function(type) {
            var prompt = elements.chatInput.value.trim();
            if (!prompt || state.isStreaming) return;
            
            // Remove welcome
            var welcome = document.getElementById('welcomeScreen');
            if (welcome) welcome.remove();
            
            // Add user message
            addMessage('user', prompt);
            state.conversationHistory.push({ role: 'user', content: prompt });
            
            // Clear input
            elements.chatInput.value = '';
            elements.chatInput.style.height = 'auto';
            state.isStreaming = true;
            elements.sendBtn.disabled = true;
            elements.imageBtn.disabled = true;
            elements.videoBtn.disabled = true;
            
            // Add assistant placeholder
            var msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            msgDiv.innerHTML = '<div class="message-avatar">&#129302;</div>' +
                '<div class="message-content-wrapper">' +
                '<div class="message-content">' +
                '<div class="loading-dots"><span></span><span></span><span></span></div>' +
                '<span style="margin-left:8px; color:var(--text-secondary); font-size:0.9rem;">Generating ' + type + '... (this may take time)</span>' +
                '</div></div>';
            elements.messagesContainer.appendChild(msgDiv);
            elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
            
            // Fetch
            fetch('/zenith/generate_media', { 
                method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ type: type, prompt: prompt })
            })
            .then(function(res) { return res.json(); })
            .then(function(data) {
                // Remove placeholder content
                var contentDiv = msgDiv.querySelector('.message-content');
                if (data.error) {
                    contentDiv.innerHTML = '<p style="color:#ef4444;">Error: ' + data.error + '</p>';
                } else {
                    if (data.type === 'image') {
                        contentDiv.innerHTML = '<img src="' + data.url + '" style="max-width:100%; border-radius:8px; border:1px solid var(--glass-border); box-shadow:0 4px 12px rgba(0,0,0,0.3);" alt="Generated Image">';
                    } else if (data.type === 'video') {
                        contentDiv.innerHTML = '<video controls autoplay loop src="' + data.url + '" style="max-width:100%; border-radius:8px; border:1px solid var(--glass-border); box-shadow:0 4px 12px rgba(0,0,0,0.3);"></video>';
                    }
                    // Add to history (as simple text representation or skip)
                    state.conversationHistory.push({ role: 'assistant', content: '[' + type.toUpperCase() + ' GENERATED: ' + data.url + ']' });
                }
                state.isStreaming = false;
                elements.sendBtn.disabled = false;
                elements.imageBtn.disabled = false;
                elements.videoBtn.disabled = false;
                elements.chatInput.focus();
            })
            .catch(function(err) {
                var contentDiv = msgDiv.querySelector('.message-content');
                contentDiv.innerHTML = '<p style="color:#ef4444;">Error: ' + err.message + '</p>';
                state.isStreaming = false;
                elements.sendBtn.disabled = false;
                elements.imageBtn.disabled = false;
                elements.videoBtn.disabled = false;
            });
        };
        
        // Toggle thinking section (global)
        window.toggleThinking = function(id) {
            var content = document.getElementById('thinking-content-' + id);
            var toggle = document.getElementById('thinking-toggle-' + id);
            if (content && toggle) {
                content.classList.toggle('expanded');
                toggle.classList.toggle('expanded');
            }
        };
        
        // Build dropdown
        function buildDropdown() {
            var html = '';
            for (var i = 0; i < models.length; i++) {
                var m = models[i];
                var sel = i === state.currentModelIndex ? 'selected' : '';
                var badge = m.reasoning ? '<span class="reasoning-badge">REASONING</span>' : '';
                html += '<div class="model-option ' + sel + '" data-index="' + i + '">';
                html += '<div class="model-name">' + m.name + ' ' + badge + '</div>';
                html += '<div class="model-desc">' + m.desc + '</div>';
                html += '</div>';
            }
            elements.modelDropdown.innerHTML = html;
        }
        
        // Build suggestions
        function buildSuggestions() {
            var html = '';
            for (var i = 0; i < suggestions.length; i++) {
                html += '<div class="suggestion" data-text="' + suggestions[i] + '">' + suggestions[i] + '</div>';
            }
            elements.suggestionsContainer.innerHTML = html;
        }

        // Build Voice Dropdown
        function buildVoiceDropdown() {
            var html = '';
            for (var i = 0; i < voices.length; i++) {
                var v = voices[i];
                var sel = v.id === state.currentVoiceId ? 'selected' : '';
                html += '<div class="voice-option ' + sel + '" data-id="' + v.id + '">';
                html += '<div class="model-name">' + v.name + '</div>';
                html += '</div>';
            }
            elements.voiceDropdown.innerHTML = html;
        }

        // Select Voice
        function selectVoice(id) {
            state.currentVoiceId = id;
            var voice = voices.find(function(v) { return v.id === id; });
            if (voice) {
                elements.currentVoiceName.textContent = voice.name;
            }
            
            var options = elements.voiceDropdown.querySelectorAll('.voice-option');
            for (var i = 0; i < options.length; i++) {
                options[i].classList.toggle('selected', options[i].getAttribute('data-id') === id);
            }
            elements.voiceDropdown.classList.remove('show');
        }

        // Speak Message
        window.speakMessage = function(btn) {
            var wrapper = btn.closest('.message-content-wrapper');
            var responseDiv = wrapper.querySelector('.response-content');
            var text = '';
            
            if (responseDiv) {
                // For assistant messages, we only want to read the actual response, not the thinking
                text = responseDiv.innerText;
            } else {
                // For user messages or if response-content is not found
                var msgContent = wrapper.querySelector('.message-content');
                text = msgContent.innerText;
            }
            
            if (text) playTTS(text);
        };

        // Speak Input
        function speakInput() {
            var text = elements.chatInput.value.trim();
            if (text) playTTS(text);
        }

        // Play TTS
        function playTTS(text) {
            fetch('/zenith/generate_audio', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text: text, voice_id: state.currentVoiceId })
            })
            .then(function(res) {
                if (!res.ok) throw new Error('TTS failed');
                return res.blob();
            })
            .then(function(blob) {
                var url = URL.createObjectURL(blob);
                var audio = new Audio(url);
                audio.play();
            })
            .catch(function(err) { console.error('TTS Error:', err); });
        }
        
        // Update File Preview
        function updateFilePreview() {
            elements.filePreview.innerHTML = '';
            if (state.files.length === 0) return;
            
            for (var i = 0; i < state.files.length; i++) {
                var file = state.files[i];
                var chip = document.createElement('div');
                chip.className = 'file-chip';
                chip.innerHTML = '<span>' + file.name + '</span><span class="file-remove" onclick="removeFile(' + i + ')">&times;</span>';
                elements.filePreview.appendChild(chip);
            }
        }
        
        window.removeFile = function(index) {
            state.files.splice(index, 1);
            updateFilePreview();
        };
        
        // Select model
        function selectModel(idx) {
            state.currentModelIndex = idx;
            var model = models[idx];
            elements.currentModelName.textContent = model.name;
            
            // Reasoning Toggle Logic
            if (model.reasoning) {
                elements.reasoningToggle.disabled = false;
            } else {
                elements.reasoningToggle.disabled = true;
                elements.reasoningToggle.classList.remove('active');
                state.showReasoning = false; // Reset if switching to non-reasoning
            }

            // File Upload Logic (Gemini/Multimodal)
            var isMultimodal = model.provider === 'google' || model.id.includes('gemini') || model.id.includes('kimi-k2.5') || model.id.includes('mistral');
            elements.attachBtn.style.display = isMultimodal ? 'block' : 'none';
            if (!isMultimodal) {
                state.files = [];
                updateFilePreview();
            }
            
            var options = elements.modelDropdown.querySelectorAll('.model-option');
            for (var i = 0; i < options.length; i++) {
                options[i].classList.toggle('selected', i === idx);
            }
            elements.modelDropdown.classList.remove('show');
        }
        
        // Add message to UI
        function addMessage(role, content, files) {
            var div = document.createElement('div');
            div.className = 'message ' + role;
            var avatar = role === 'user' ? '&#128100;' : '&#129302;';
            
            var actionsHtml = '';
            if (role === 'user') {
                actionsHtml = '<div class="message-actions">' +
                    '<button class="action-btn" onclick="editMessage(this)" title="Edit">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>' +
                    '</button></div>';
            } else {
                 actionsHtml = '<div class="message-actions">' +
                    '<button class="action-btn" onclick="copyMessage(this)" title="Copy">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>' +
                    '</button>' +
                    '<button class="action-btn" onclick="regenerateMessage()" title="Regenerate">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>' +
                    '</button>' +
                    '<button class="action-btn" onclick="speakMessage(this)" title="Speak">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>' +
                    '</button></div>';
            }
            
            var filesHtml = '';
            if (files && files.length > 0) {
                filesHtml = '<div class="message-files">';
                for (var i = 0; i < files.length; i++) {
                    var f = files[i];
                    filesHtml += '<div class="file-attachment"><span class="file-icon">&#128206;</span><span>' + f.name + '</span></div>';
                }
                filesHtml += '</div>';
            }

            div.innerHTML = '<div class="message-avatar">' + avatar + '</div>' +
                           '<div class="message-content-wrapper">' +
                           filesHtml + 
                           '<div class="message-content">' + parseText(content) + '</div>' +
                           actionsHtml + '</div>';
            
            elements.messagesContainer.appendChild(div);
            elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
        }
        
        // Send message
        function sendMessage() {
            var message = elements.chatInput.value.trim();
            if ((!message && state.files.length === 0) || state.isStreaming) return;
            
            // Remove welcome
            var welcome = document.getElementById('welcomeScreen');
            if (welcome) welcome.remove();
            
            // Capture files before clearing
            var currentFiles = state.files.slice();
            
            // Add user message with files
            addMessage('user', message, currentFiles);
            state.conversationHistory.push({ role: 'user', content: message });
            
            // Clear input
            elements.chatInput.value = '';
            elements.chatInput.style.height = 'auto';
            state.isStreaming = true;
            elements.sendBtn.disabled = true;
            
            // Create assistant message container
            var thinkingId = 'think-' + Date.now();
            var msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            // Pre-build structure with actions
             var actionsHtml = '<div class="message-actions">' +
                    '<button class="action-btn" onclick="copyMessage(this)" title="Copy">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>' +
                    '</button>' +
                    '<button class="action-btn" onclick="regenerateMessage()" title="Regenerate">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>' +
                    '</button>' +
                    '<button class="action-btn" onclick="speakMessage(this)" title="Speak">' +
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>' +
                    '</button></div>';
            
            msgDiv.innerHTML = '<div class="message-avatar">&#129302;</div>' +
                '<div class="message-content-wrapper">' +
                '<div class="message-content">' +
                '<div class="thinking-section" id="thinking-section-' + thinkingId + '" style="display:none;">' +
                '<div class="thinking-header" onclick="toggleThinking(\'' + thinkingId + '\')">' +
                '<div class="thinking-title"><span class="spinner"></span><span id="thinking-status-' + thinkingId + '">Thinking...</span></div>' +
                '<button class="thinking-toggle" id="thinking-toggle-' + thinkingId + '" type="button">' +
                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>' +
                '</button></div>' +
                '<div class="thinking-content" id="thinking-content-' + thinkingId + '"><div class="thinking-content-inner" id="thinking-inner-' + thinkingId + '"></div></div></div>' +
                '<div class="response-content" id="response-' + thinkingId + '">' +
                '<div class="loading-dots"><span></span><span></span><span></span></div>' +
                '</div></div>' + actionsHtml + '</div>';
            
            elements.messagesContainer.appendChild(msgDiv);
            elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
            
            var thinkingSection = document.getElementById('thinking-section-' + thinkingId);
            var thinkingInner = document.getElementById('thinking-inner-' + thinkingId);
            var thinkingStatus = document.getElementById('thinking-status-' + thinkingId);
            var responseDiv = document.getElementById('response-' + thinkingId);
            
            // Prepare request
            var formData = new FormData();
            formData.append('message', message);
            formData.append('model_index', state.currentModelIndex);
            formData.append('show_reasoning', state.showReasoning);
            formData.append('deep_research', state.deepResearch);
            formData.append('history', JSON.stringify(state.conversationHistory.slice(-20)));
            
            // Send files
            for (var i = 0; i < state.files.length; i++) {
                formData.append('files', state.files[i]);
            }
            // Clear files after sending
            state.files = [];
            updateFilePreview();
            
            // Fetch
            fetch('/zenith/chat', { method: 'POST', body: formData })
            .then(function(response) {
                var reader = response.body.getReader();
                var decoder = new TextDecoder();
                var reasoning = '';
                var content = '';
                var accumulatedContent = '';
                var hasContent = false;
                
                function read() {
                    reader.read().then(function(result) {
                        if (result.done) {
                            // Mark thinking complete
                            if (reasoning || accumulatedContent.includes('</think>')) {
                                var titleDiv = thinkingSection.querySelector('.thinking-title');
                                titleDiv.innerHTML = '<span class="check">&#10004;</span><span>Thinking Complete</span>';
                            }
                            if (content) {
                                state.conversationHistory.push({ role: 'assistant', content: content });
                            }
                            state.isStreaming = false;
                            elements.sendBtn.disabled = false;
                            elements.chatInput.focus();
                            return;
                        }
                        
                        var chunk = decoder.decode(result.value);
                        var lines = chunk.split('\n');
                        
                        for (var i = 0; i < lines.length; i++) {
                            var line = lines[i];
                            if (line.indexOf('data: ') !== 0) continue;
                            var data = line.substring(6);
                            if (data === '[DONE]') continue;
                            
                            try {
                                var parsed = JSON.parse(data);
                                if (parsed.type === 'reasoning' && state.showReasoning) {
                                    thinkingSection.style.display = 'block';
                                    reasoning += parsed.content;
                                    thinkingInner.innerHTML = parseText(reasoning);
                                } else if (parsed.type === 'content') {
                                    accumulatedContent += parsed.content;
                                    
                                    // Handle <think> tags (MiniMax style)
                                    var thinkStart = accumulatedContent.indexOf('<think>');
                                    if (thinkStart !== -1) {
                                        var thinkEnd = accumulatedContent.indexOf('</think>');
                                        if (thinkEnd !== -1) {
                                            // Thinking complete
                                            var thinkPart = accumulatedContent.substring(thinkStart + 7, thinkEnd);
                                            var contentPart = accumulatedContent.substring(0, thinkStart) + accumulatedContent.substring(thinkEnd + 8);
                                            
                                            // Update thinking UI
                                            if (state.showReasoning) {
                                                thinkingSection.style.display = 'block';
                                                thinkingInner.innerHTML = parseText(thinkPart);
                                                var titleDiv = thinkingSection.querySelector('.thinking-title');
                                                if (!titleDiv.innerHTML.includes('Thinking Complete')) {
                                                     titleDiv.innerHTML = '<span class="check">&#10004;</span><span>Thinking Complete</span>';
                                                }
                                            }
                                            
                                            // Update Content UI
                                            content = contentPart; 
                                            if (content) {
                                                if (!hasContent) {
                                                    hasContent = true;
                                                    responseDiv.innerHTML = '';
                                                }
                                                responseDiv.innerHTML = parseText(content);
                                            }
                                        } else {
                                            // Still thinking
                                            var thinkPart = accumulatedContent.substring(thinkStart + 7);
                                            var preThink = accumulatedContent.substring(0, thinkStart);
                                            
                                            if (state.showReasoning) {
                                                thinkingSection.style.display = 'block';
                                                thinkingInner.innerHTML = parseText(thinkPart);
                                            }
                                            
                                            if (preThink) {
                                                 if (!hasContent) {
                                                    hasContent = true;
                                                    responseDiv.innerHTML = '';
                                                }
                                                responseDiv.innerHTML = parseText(preThink);
                                            }
                                        }
                                    } else {
                                        // Standard content
                                        if (!hasContent) {
                                            hasContent = true;
                                            responseDiv.innerHTML = '';
                                            if (reasoning) {
                                                var titleDiv = thinkingSection.querySelector('.thinking-title');
                                                titleDiv.innerHTML = '<span class="check">&#10004;</span><span>Thinking Complete</span>';
                                            }
                                        }
                                        content = accumulatedContent;
                                        responseDiv.innerHTML = parseText(content);
                                    }
                                } else if (parsed.type === 'error') {
                                    responseDiv.innerHTML = '<p style="color:#ef4444;">Error: ' + parsed.content + '</p>';
                                }
                                elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
                            } catch(e) {}
                        }
                        read();
                    }).catch(function(err) {
                        responseDiv.innerHTML = '<p style="color:#ef4444;">Error: ' + err.message + '</p>';
                        state.isStreaming = false;
                        elements.sendBtn.disabled = false;
                    });
                }
                read();
            }).catch(function(err) {
                responseDiv.innerHTML = '<p style="color:#ef4444;">Error: ' + err.message + '</p>';
                state.isStreaming = false;
                elements.sendBtn.disabled = false;
            });
        }
        
        // Clear chat
        function clearChat() {
            state.conversationHistory = [];
            state.files = [];
            updateFilePreview();
            elements.messagesContainer.innerHTML = '<div class="welcome" id="welcomeScreen">' +
                '<div class="welcome-icon">&#128640;</div>' +
                '<h2>Welcome to ZYLO ZENITH</h2>' +
                '<p>Experience powerful AI models with reasoning capabilities, code generation, and more.</p>' +
                '<div class="suggestions" id="suggestionsContainer"></div></div>';
            elements.suggestionsContainer = document.getElementById('suggestionsContainer');
            buildSuggestions();
        }
        
        // Initialize
        function init() {
            // Cache elements
            elements.modelBtn = document.getElementById('modelBtn');
            elements.modelDropdown = document.getElementById('modelDropdown');
            elements.voiceBtn = document.getElementById('voiceBtn');
            elements.voiceDropdown = document.getElementById('voiceDropdown');
            elements.currentModelName = document.getElementById('currentModelName');
            elements.currentVoiceName = document.getElementById('currentVoiceName');
            elements.reasoningToggle = document.getElementById('reasoningToggle');
            elements.clearBtn = document.getElementById('clearBtn');
            elements.sendBtn = document.getElementById('sendBtn');
            elements.chatInput = document.getElementById('chatInput');
            elements.messagesContainer = document.getElementById('messagesContainer');
            elements.suggestionsContainer = document.getElementById('suggestionsContainer');
            elements.attachBtn = document.getElementById('attachBtn');
            elements.fileInput = document.getElementById('fileInput');
            elements.filePreview = document.getElementById('filePreview');
            elements.imageBtn = document.getElementById('imageBtn');
            elements.videoBtn = document.getElementById('videoBtn');
            elements.speakInputBtn = document.getElementById('speakInputBtn');
            elements.deepResearchBtn = document.getElementById('deepResearchBtn');
            elements.toolsToggle = document.getElementById('toolsToggle');
            elements.toolsBubbles = document.getElementById('toolsBubbles');
            elements.toolsIcon = document.getElementById('toolsIcon');
            elements.closeToolsIcon = document.getElementById('closeToolsIcon');
            elements.overlay = document.getElementById('dropdownOverlay');
            
            // Build UI
            buildDropdown();
            buildVoiceDropdown();
            buildSuggestions();
            selectModel(0); // Initialize with first model
            selectVoice('Magpie-Multilingual.EN-US.Leo.Neutral'); // Initialize with default voice

            // Event: Tools Toggle
            elements.toolsToggle.onclick = function(e) {
                e.stopPropagation();
                var show = !elements.toolsBubbles.classList.contains('show');
                elements.toolsBubbles.classList.toggle('show', show);
                this.classList.toggle('active', show);
                elements.toolsIcon.style.display = show ? 'none' : 'block';
                elements.closeToolsIcon.style.display = show ? 'block' : 'none';
            };

            // Event: Deep Research Toggle
            elements.deepResearchBtn.onclick = function(e) {
                e.stopPropagation();
                state.deepResearch = !state.deepResearch;
                this.classList.toggle('active', state.deepResearch);
                console.log('Deep Research:', state.deepResearch);
            };

            // Close tools when clicking bubble (except deepResearchBtn)
            var bubbles = elements.toolsBubbles.querySelectorAll('.bubble:not(#deepResearchBtn)');
            bubbles.forEach(function(b) {
                b.addEventListener('click', function() {
                    elements.toolsBubbles.classList.remove('show');
                    elements.toolsToggle.classList.remove('active');
                    elements.toolsIcon.style.display = 'block';
                    elements.closeToolsIcon.style.display = 'none';
                });
            });
            
            // Event: Model button
            elements.modelBtn.onclick = function(e) {
                e.stopPropagation();
                var show = !elements.modelDropdown.classList.contains('show');
                
                // Reset all
                elements.modelDropdown.classList.remove('show');
                elements.voiceDropdown.classList.remove('show');
                elements.overlay.classList.remove('show');
                document.body.classList.remove('dropdown-open');
                
                if (show) {
                    elements.modelDropdown.classList.add('show');
                    if (window.innerWidth <= 768) {
                        elements.overlay.classList.add('show');
                        document.body.classList.add('dropdown-open');
                    }
                }
            };
            
            // Event: Voice button
            elements.voiceBtn.onclick = function(e) {
                e.stopPropagation();
                var show = !elements.voiceDropdown.classList.contains('show');
                
                // Reset all
                elements.modelDropdown.classList.remove('show');
                elements.voiceDropdown.classList.remove('show');
                elements.overlay.classList.remove('show');
                document.body.classList.remove('dropdown-open');
                
                if (show) {
                    elements.voiceDropdown.classList.add('show');
                    if (window.innerWidth <= 768) {
                        elements.overlay.classList.add('show');
                        document.body.classList.add('dropdown-open');
                    }
                }
            };
            
            // Event: Model selection
            elements.modelDropdown.onclick = function(e) {
                var target = e.target;
                while (target && !target.classList.contains('model-option')) {
                    target = target.parentElement;
                }
                if (target) {
                    selectModel(parseInt(target.getAttribute('data-index'), 10));
                    elements.overlay.classList.remove('show');
                    elements.modelDropdown.classList.remove('show');
                    document.body.classList.remove('dropdown-open');
                }
            };

            // Event: Voice selection
            elements.voiceDropdown.onclick = function(e) {
                var target = e.target;
                while (target && !target.classList.contains('voice-option')) {
                    target = target.parentElement;
                }
                if (target) {
                    selectVoice(target.getAttribute('data-id'));
                    elements.overlay.classList.remove('show');
                    elements.voiceDropdown.classList.remove('show');
                    document.body.classList.remove('dropdown-open');
                }
            };
            
            // Event: Click outside dropdown
            document.onclick = function(e) {
                if (!elements.modelDropdown.contains(e.target) && !elements.modelBtn.contains(e.target)) {
                    elements.modelDropdown.classList.remove('show');
                }
                if (elements.voiceDropdown && !elements.voiceDropdown.contains(e.target) && !elements.voiceBtn.contains(e.target)) {
                    elements.voiceDropdown.classList.remove('show');
                }
                if (elements.toolsBubbles && !elements.toolsBubbles.contains(e.target) && !elements.toolsToggle.contains(e.target)) {
                    elements.toolsBubbles.classList.remove('show');
                    elements.toolsToggle.classList.remove('active');
                    elements.toolsIcon.style.display = 'block';
                    elements.closeToolsIcon.style.display = 'none';
                }
                if (!elements.modelDropdown.classList.contains('show') && !elements.voiceDropdown.classList.contains('show')) {
                    elements.overlay.classList.remove('show');
                    document.body.classList.remove('dropdown-open');
                }
            };
            
            elements.overlay.onclick = function() {
                elements.modelDropdown.classList.remove('show');
                elements.voiceDropdown.classList.remove('show');
                elements.overlay.classList.remove('show');
                document.body.classList.remove('dropdown-open');
            };
            
            // Event: Reasoning toggle
            elements.reasoningToggle.onclick = function() {
                if (this.disabled) return;
                state.showReasoning = !state.showReasoning;
                this.classList.toggle('active', state.showReasoning);
            };
            
            // Event: Clear
            elements.clearBtn.onclick = clearChat;
            
            // Event: Send
            elements.sendBtn.onclick = sendMessage;
            
            // Event: Enter key
            elements.chatInput.onkeydown = function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            };
            
            // Event: Auto-resize
            elements.chatInput.oninput = function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 200) + 'px';
            };
            
            // Event: File Input
            elements.attachBtn.onclick = function() {
                elements.fileInput.click();
            };
            
            elements.fileInput.onchange = function() {
                for (var i = 0; i < this.files.length; i++) {
                    state.files.push(this.files[i]);
                }
                updateFilePreview();
                this.value = ''; // Reset input
            };
            
            // Media Buttons
            elements.imageBtn.onclick = function() {
                generateMedia('image');
            };
            
            elements.videoBtn.onclick = function() {
                generateMedia('video');
            };
            
            elements.speakInputBtn.onclick = speakInput;
            
            // Event: Suggestion click
            elements.messagesContainer.onclick = function(e) {
                if (e.target.classList.contains('suggestion')) {
                    elements.chatInput.value = e.target.getAttribute('data-text');
                    sendMessage();
                }
            };
            
            console.log('ZYLO ZENITH initialized with', models.length, 'models');
        }
        
        // Run on DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    })();
    </script>
</body>
</html>
'''


@zenith_bp.route('/zenith')
def index():
    """Serve the main chat interface."""
    models_json = json.dumps(MODELS)
    voices_json = json.dumps(VOICES)
    return render_template_string(HTML_TEMPLATE, models_json=models_json, voices_json=voices_json)


@zenith_bp.route('/zenith/chat', methods=['POST'])
def chat():
    """Handle chat messages and stream responses."""
    message = request.form.get('message', '')
    model_index = int(request.form.get('model_index', 0))
    show_reasoning = request.form.get('show_reasoning', 'true') == 'true'
    deep_research_enabled = request.form.get('deep_research', 'false') == 'true'
    history = json.loads(request.form.get('history', '[]'))
    
    # Read files immediately into memory to avoid closed file errors in stream_with_context
    uploaded_files = []
    for file in request.files.getlist('files'):
        uploaded_files.append({
            "data": file.read(),
            "mime_type": file.content_type or 'application/octet-stream'
        })
        
    model_config = MODELS[model_index]

    def generate():
        try:
            if deep_research_enabled:
                system_prompt = (
                    "You are ZYLO ZENITH in DEEP RESEARCH mode. "
                    "You have access to advanced web scraping via ScraperAPI.\n"
                    "Rules:\n"
                    "1. For any query that requires detailed, factual, or up-to-date information, you MUST use the search tool.\n"
                    "2. To search, output ONLY: [SEARCH: query]\n"
                    "3. Do NOT explain your search. Do NOT include any other text or reasoning before the search tag.\n"
                    "4. If the user provides a URL, it has already been scraped for you; analyze the provided [SCRAPED CONTENT] thoroughly.\n"
                    "5. Respond in English only."
                )
            else:
                system_prompt = (
                    "You are a high-performance AI assistant. "
                    "Rules:\n"
                    "1. If you need real-time information or are unsure, you MUST use the search tool.\n"
                    "2. To search, output ONLY: [SEARCH: query]\n"
                    "3. Do NOT explain why you are searching. Do NOT include any other text.\n"
                    "4. Respond in English only."
                )
            
            messages = history
            if not any(m["role"] == "system" for m in messages):
                messages = [{"role": "system", "content": system_prompt}] + history

            # Direct URL Scraping for Deep Research
            if deep_research_enabled and message:
                url_match = re.search(r'https?://[^\s]+', message)
                if url_match:
                    target_url = url_match.group(0)
                    yield "data: " + json.dumps({"type": "content", "content": f"\n\U0001F517 *Deep Research: Scraping content from {target_url}...*\n"}) + "\n\n"
                    scraped_content = fetch_url_content(target_url)
                    if scraped_content:
                        url_info = f"\n\n[SCRAPED CONTENT FROM {target_url}]:\n{scraped_content}\n\n"
                        # Update BOTH history and the current user message being processed
                        for msg in messages:
                            if msg["role"] == "user":
                                if isinstance(msg["content"], str):
                                    if target_url in msg["content"]:
                                        msg["content"] += url_info
                                elif isinstance(msg["content"], list):
                                    for part in msg["content"]:
                                        if part.get("type") == "text" and target_url in part.get("text", ""):
                                            part["text"] += url_info
                                            break
                        time.sleep(1) # Small pause for UI visibility

            # Prepare multimodal messages for NVIDIA if needed
            nvidia_messages = messages
            if uploaded_files:
                nvidia_messages = [m.copy() for m in messages]
                for i in range(len(nvidia_messages) - 1, -1, -1):
                    if nvidia_messages[i]["role"] == "user":
                        text_content = nvidia_messages[i]["content"]

                        # Check if this is a model that expects HTML img tags (Kimi) or other NVIDIA models (use structured format)
                        is_html_img_model = model_config["id"] in ["moonshotai/kimi-k2.5"]

                        if is_html_img_model:
                            # Kimi models expect HTML <img> tags in content string
                            new_content = text_content
                            for file_info in uploaded_files:
                                if file_info["mime_type"].startswith("image/"):
                                    b64_image = base64.b64encode(file_info["data"]).decode('utf-8')
                                    new_content += f" <img src=\"data:{file_info['mime_type']};base64,{b64_image}\" />"
                            nvidia_messages[i]["content"] = new_content
                        else:
                            # Other NVIDIA models expect structured content array
                            new_content = [{"type": "text", "text": text_content}]
                            for file_info in uploaded_files:
                                if file_info["mime_type"].startswith("image/"):
                                    b64_image = base64.b64encode(file_info["data"]).decode('utf-8')
                                    new_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{file_info['mime_type']};base64,{b64_image}"}
                                    })
                            nvidia_messages[i]["content"] = new_content
                        break

            stream = None
            is_fallback = False
            provider_used = model_config["provider"]

            # --- PRIMARY ATTEMPT ---
            if model_config["provider"] == "nvidia":
                try:
                    kwargs = {
                        "model": model_config["id"],
                        "messages": nvidia_messages,
                        "temperature": model_config.get("temperature", 0.7),
                        "top_p": model_config.get("top_p", 0.95),
                        "max_tokens": model_config.get("max_tokens", 8192),
                        "stream": True,
                        "timeout": 30
                    }
                    if model_config.get("extra_body"):
                        kwargs["extra_body"] = model_config["extra_body"]

                    # Special timeout handling for Kimi models
                    is_kimi_model = model_config["id"] == "moonshotai/kimi-k2.5"
                    if is_kimi_model:
                        # Try primary Kimi model with 15s timeout
                        try:
                            kwargs["timeout"] = 15
                            stream = nvidia_client.chat.completions.create(**kwargs)
                        except Exception as e:
                            print(f"Kimi primary (moonshotai/kimi-k2.5) failed or timed out: {e}. Switching to fallback Kimi model.")
                            # Fallback to moonshotai/kimi-k2-instruct-0905 (which doesn't support images)
                            try:
                                fallback_kwargs = kwargs.copy()
                                fallback_kwargs["model"] = "moonshotai/kimi-k2-instruct-0905"
                                fallback_kwargs["timeout"] = 30  # Use longer timeout for fallback

                                # Remove image data for fallback model (since it doesn't support multimodal)
                                if isinstance(fallback_kwargs["messages"], list):
                                    for msg in fallback_kwargs["messages"]:
                                        if msg["role"] == "user" and isinstance(msg["content"], str):
                                            # Remove HTML img tags from content string
                                            msg["content"] = re.sub(r'\s*<img\s+src="data:image/[^>]+>', '', msg["content"])
                                            msg["content"] = msg["content"].strip()

                                stream = nvidia_client.chat.completions.create(**fallback_kwargs)
                                print("Successfully switched to fallback Kimi model: moonshotai/kimi-k2-instruct-0905")
                            except Exception as fallback_e:
                                print(f"Both Kimi models failed: {fallback_e}. Switching to general fallback.")
                                stream = None
                                is_fallback = True
                    else:
                        # Regular NVIDIA models
                        stream = nvidia_client.chat.completions.create(**kwargs)
                except Exception as e:
                    print(f"NVIDIA primary failed: {e}. Switching to fallback.")
                    is_fallback = True

            elif model_config["provider"] == "google":
                if not HAS_GEMINI or gemini_client is None:
                    is_fallback = True
                else:
                    try:
                        chat_history_str = ""
                        for msg in messages:
                            if msg["role"] == "system": chat_history_str += "System: " + msg["content"] + "\n"
                            elif msg["role"] == "user": chat_history_str += "You: " + msg["content"] + "\n"
                            elif msg["role"] == "assistant": chat_history_str += "Assistant: " + msg["content"] + "\n"
                        contents = [chat_history_str]
                        if uploaded_files:
                            from google.genai import types
                            for file_info in uploaded_files:
                                contents.append(types.Part.from_bytes(data=file_info["data"], mime_type=file_info["mime_type"]))
                        stream = gemini_client.models.generate_content_stream(
                            model=model_config["id"].replace("google/", ""),
                            contents=contents
                        )
                    except Exception as e:
                        print(f"Gemini primary failed: {e}. Switching to fallback.")
                        is_fallback = True

            # --- FALLBACK LOGIC ---
            if is_fallback or (not stream and model_config["provider"] == "openrouter"):
                # Special fallback for GLM-4.7 to Cerebras
                if model_config["id"] == "z-ai/glm4.7" and cerebras_client:
                    try:
                        print("Falling back to Cerebras for GLM-4.7")
                        stream = cerebras_client.chat.completions.create(
                            model="zai-glm-4.7",
                            messages=messages,
                            stream=True,
                            temperature=1,
                            max_tokens=16384,
                            timeout=30
                        )
                        provider_used = "cerebras"
                    except Exception as ce:
                        print(f"Cerebras fallback failed: {ce}")
                
                # General fallback to OpenRouter
                if stream is None and openrouter_client:
                    print("Falling back to OpenRouter")
                    try:
                        stream = openrouter_client.chat.completions.create(
                            model="tngtech/deepseek-r1t2-chimera:free",
                            messages=messages,
                            temperature=0.1,
                            stream=True,
                            timeout=30
                        )
                        provider_used = "openrouter"
                    except Exception as e:
                        print(f"OpenRouter fallback failed: {e}")

            if not stream:
                yield "data: " + json.dumps({"type": "error", "content": "All providers failed or timed out. Please try again."}) + "\n\n"
                return

            # --- STREAMING LOOP ---
            full_response = ""
            search_detected = False
            in_think_block = False
            buffer = ""

            # Handle Gemini differently due to its unique stream object
            if provider_used == "google":
                try:
                    for chunk in stream:
                        content = chunk.text
                        if content:
                            yield "data: " + json.dumps({"type": "content", "content": content}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                except Exception as e:
                    # If it fails during streaming, we can't really fall back easily if we already yielded
                    yield "data: " + json.dumps({"type": "error", "content": f"Gemini stream error: {e}"}) + "\n\n"
                    return
            
            # Standard OpenAI-style streaming for NVIDIA, Cerebras, OpenRouter
            for chunk in stream:
                if not getattr(chunk, "choices", None) or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                # 1. Immediate reasoning streaming (NVIDIA/OpenRouter)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield "data: " + json.dumps({"type": "reasoning", "content": reasoning}) + "\n\n"
                
                # 2. Content processing
                content = getattr(delta, "content", None)
                if content:
                    full_response += content
                    buffer += content
                    
                    # Detect and handle <think> blocks in real-time
                    if not in_think_block and "<think>" in buffer:
                        in_think_block = True
                        parts = buffer.split("<think>", 1)
                        if parts[0]: # Pre-think content
                            yield "data: " + json.dumps({"type": "content", "content": parts[0]}) + "\n\n"
                        buffer = parts[1] # Rest is reasoning
                    
                    if in_think_block:
                        if "</think>" in buffer:
                            parts = buffer.split("</think>", 1)
                            if parts[0]:
                                yield "data: " + json.dumps({"type": "reasoning", "content": parts[0]}) + "\n\n"
                            buffer = parts[1]
                            in_think_block = False
                        else:
                            # Stream the current buffer as reasoning and clear it
                            yield "data: " + json.dumps({"type": "reasoning", "content": buffer}) + "\n\n"
                            buffer = ""
                            continue

                    # Search detection
                    if not in_think_block:
                        if "[SEARCH:" in full_response and not search_detected:
                            if "]" in full_response:
                                search_detected = True
                                match = re.search(r"\[SEARCH:(.*?)\]", full_response)
                                if match:
                                    query = match.group(1).strip()
                                    search_type = "Deep Research" if deep_research_enabled else "Search"
                                    yield "data: " + json.dumps({"type": "content", "content": f"\n\U0001F50D *{search_type}ing for: {query}*...\n"}) + "\n\n"
                                    
                                    if deep_research_enabled:
                                        search_results = deep_search(query)
                                    else:
                                        search_results = web_search(query)
                                    
                                    messages.append({"role": "assistant", "content": f"[{search_type.upper()}: {query}]"})
                                    messages.append({"role": "user", "content": f"Search Results:\n{search_results}\n\nPlease synthesize the final answer."})
                                    
                                    if provider_used == "cerebras" and cerebras_client:
                                        second_stream = cerebras_client.chat.completions.create(
                                            model="zai-glm-4.7",
                                            messages=messages,
                                            stream=True,
                                            timeout=13
                                        )
                                    elif provider_used == "nvidia" and nvidia_client:
                                        second_stream = nvidia_client.chat.completions.create(
                                            model=model_config["id"],
                                            messages=messages,
                                            stream=True,
                                            temperature=0.3,
                                            timeout=13
                                        )
                                    elif openrouter_client:
                                        second_stream = openrouter_client.chat.completions.create(
                                            model="tngtech/deepseek-r1t2-chimera:free",
                                            messages=messages,
                                            stream=True,
                                            timeout=13
                                        )
                                    
                                    if second_stream:
                                        for s_chunk in second_stream:
                                            if not getattr(s_chunk, "choices", None) or not s_chunk.choices:
                                                continue
                                            s_delta = s_chunk.choices[0].delta
                                            s_reasoning = getattr(s_delta, "reasoning_content", None)
                                            if s_reasoning:
                                                yield "data: " + json.dumps({"type": "reasoning", "content": s_reasoning}) + "\n\n"
                                            s_content = getattr(s_delta, "content", None)
                                            if s_content:
                                                yield "data: " + json.dumps({"type": "content", "content": s_content}) + "\n\n"
                                    break
                        else:
                            # Stream buffer if it doesn't look like a search tag is starting
                            if buffer:
                                # Hold buffer only if it contains a partial [SEARCH:
                                if "[" in buffer and "]" not in buffer:
                                    if "[SEARCH:".startswith(buffer) or buffer.startswith("[SEARCH:"):
                                        pass # Keep buffering
                                    else:
                                        yield "data: " + json.dumps({"type": "content", "content": buffer}) + "\n\n"
                                        buffer = ""
                                else:
                                    yield "data: " + json.dumps({"type": "content", "content": buffer}) + "\n\n"
                                    buffer = ""

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "content": str(e)}) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@zenith_bp.route('/zenith/generate_media', methods=['POST'])
def generate_media():
    """Handle image and video generation."""
    if not HAS_ZHIPU or zhipu_client is None:
        return jsonify({"error": "ZhipuAI not available"})

    data = request.json
    action_type = data.get('type')
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"})

    try:
        # IMAGE: CogView-3-Flash (Free/Fast)
        if action_type == 'image':
            response = zhipu_client.images.generations(
                model="cogview-3-flash", 
                prompt=prompt,
                quality="standard"
            )
            if response.data:
                return jsonify({"url": response.data[0].url, "type": "image"})
            return jsonify({"error": "No image data returned"})

        # VIDEO: CogVideoX-Flash (Free/Fast)
        elif action_type == 'video':
            response = zhipu_client.videos.generations(
                model="cogvideox-flash", 
                prompt=prompt
            )
            task_id = response.id
            
            # Polling for completion
            for _ in range(40):
                time.sleep(5)
                status_res = zhipu_client.videos.retrieve_videos_result(id=task_id)
                if status_res.task_status == "SUCCESS":
                    if status_res.video_result and len(status_res.video_result) > 0:
                        return jsonify({"url": status_res.video_result[0].url, "type": "video"})
                    return jsonify({"error": "Video success but no result data"})
                if status_res.task_status == "FAIL":
                    return jsonify({"error": "Video generation failed"})
            
            return jsonify({"error": "Video synthesis timed out. Try again."})
            
    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid action type"})


@zenith_bp.route('/zenith/generate_audio', methods=['POST'])
def generate_audio():
    """Handle TTS generation using NVIDIA Magpie (exact voice.py logic) or gTTS fallback."""
    data = request.json
    text = data.get('text', '')
    voice_id = data.get('voice_id', 'Magpie-Multilingual.EN-US.Leo.Neutral')
    
    if not text:
        return jsonify({"error": "No text provided"})

    # 1. Try NVIDIA Magpie TTS via talk.py (exact logic from voice.py)
    if voice_id.startswith("Magpie"):
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_file = tmp.name
            
            cmd = [
                "python3",
                TTS_TALK_SCRIPT,
                "--server", TTS_SERVER,
                "--use-ssl",
                "--metadata", "function-id", TTS_FUNCTION_ID,
                "--metadata", "authorization", f"Bearer {TTS_API_KEY}",
                "--language-code", "en-US",
                "--voice", voice_id,
                "--text", text,
                "--output", output_file
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(output_file):
                with open(output_file, "rb") as f:
                    audio_data = f.read()
                os.unlink(output_file) # Clean up
                return Response(audio_data, mimetype="audio/wav")
            else:
                print(f"NVIDIA TTS failed: {result.stderr}")
        except Exception as e:
            print(f"NVIDIA TTS Subprocess Error: {e}")

    # 2. Fallback to gTTS
    if not HAS_GTTS:
        return jsonify({"error": "gTTS not installed and NVIDIA TTS failed"})

    # Find voice config for gTTS fallback
    voice_config = next((v for v in VOICES if v['id'] == voice_id), VOICES[0])
    
    try:
        tts = gTTS(text=text, lang=voice_config['lang'], tld=voice_config['tld'])
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return Response(fp.read(), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # For standalone testing
    app = Flask(__name__)
    app.register_blueprint(zenith_bp)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

