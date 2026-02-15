from flask import Flask, render_template_string, jsonify, request, Response
import requests
import os
import subprocess
import tempfile
from io import BytesIO
from openai import OpenAI

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

app = Flask(__name__)

# --- CONFIGURATION ---
API_KEY = "428ee5ac3e354f85ba1104055261102"
BASE_URL = "http://api.weatherapi.com/v1/current.json"
QUERY = "Bolpur"
REFRESH_SECONDS = 30
TTS_API_KEY = os.getenv("NVIDIA_TTS_API_KEY", "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")
TTS_FUNCTION_ID = os.getenv("NVIDIA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
TTS_SERVER = os.getenv("NVIDIA_TTS_SERVER", "grpc.nvcf.nvidia.com:443")
TTS_TALK_SCRIPT = os.getenv("TTS_TALK_SCRIPT", "/home/ujan/python-clients/scripts/tts/talk.py")
ENERGETIC_VOICE_ID = "Magpie-Multilingual.EN-US.Aria.Happy"

# NVIDIA LLM CONFIG
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# Server-side Cache for LLM Summaries
WEATHER_STATE_CACHE = {}

VOICES = [
    {"id": "Magpie-Multilingual.EN-US.Leo.Neutral", "lang": "en", "tld": "us"},
    {"id": "Magpie-Multilingual.EN-US.Mia.Neutral", "lang": "en", "tld": "co.uk"},
    {"id": "Magpie-Multilingual.EN-US.Jason.Calm", "lang": "en", "tld": "ie"},
    {"id": "Magpie-Multilingual.EN-US.Aria.Happy", "lang": "en", "tld": "com.au"},
    {"id": "Magpie-Multilingual.EN-US.Leo.Angry", "lang": "en", "tld": "ca"},
    {"id": "Magpie-Multilingual.EN-US.Mia.Sad", "lang": "en", "tld": "co.in"},
]

# --- HELPERS ---
def safe_https_icon(url):
    if not url: return None
    return ("https:" + url) if url.startswith("//") else url

def human_label(key: str) -> str:
    # Clean up keys for display (e.g., "precip_mm" -> "Precip Mm")
    key = key.replace("_", " ").replace("-", " ")
    return " ".join([w.capitalize() if w.upper() not in ("EPA", "DEFRA") else w.upper() for w in key.split()])

# Air Quality Mapping
AQ_ORDER = ["co", "gb-defra-index", "no2", "o3", "pm10", "pm2_5", "so2", "us-epa-index"]
AQ_LABELS = {
    "co": "CO", "gb-defra-index": "GB Defra", "no2": "NO2", "o3": "O3",
    "pm10": "PM10", "pm2_5": "PM2.5", "so2": "SO2", "us-epa-index": "US EPA"
}

def get_us_aqi(pm25):
    """Calculate US AQI value from PM2.5 concentration."""
    if pm25 is None: return 0
    if pm25 <= 12.0:
        return round((50 / 12.0) * pm25)
    elif pm25 <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4:
        return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    elif pm25 <= 350.4:
        return round(((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301)
    elif pm25 <= 500.4:
        return round(((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401)
    return 500

def format_payload(raw: dict) -> dict:
    loc = raw.get("location", {})
    cur = raw.get("current", {})
    
    # Condition Icon
    condition = cur.get("condition", {}) or {}
    if "icon" in condition:
        condition["icon"] = safe_https_icon(condition["icon"])

    # Location Block
    location_block = {
        "City": loc.get("name"),
        "Region": loc.get("region"),
        "Country": loc.get("country"),
        "Time": loc.get("localtime").split(" ")[1] if "localtime" in loc else "--",
        "Date": loc.get("localtime").split(" ")[0] if "localtime" in loc else "--",
    }

    # Air Quality
    raw_aq = cur.get("air_quality", {}) or {}
    aq_block = {}
    for key in AQ_ORDER:
        if key in raw_aq:
            label = AQ_LABELS.get(key, human_label(key))
            aq_block[label] = raw_aq.get(key)

    # Calculate US AQI Value (0-500 scale)
    pm25_val = raw_aq.get("pm2_5")
    aqi_val = get_us_aqi(pm25_val)
    
    # Translate US EPA Index to descriptive text for LLM context
    epa_index = raw_aq.get("us-epa-index")
    epa_labels = {
        1: "Good",
        2: "Moderate",
        3: "Unhealthy (Sens.)",
        4: "Unhealthy",
        5: "Very Unhealthy",
        6: "Hazardous"
    }
    aqi_status = epa_labels.get(epa_index, "Unknown") if epa_index else "N/A"

    # Top Metrics (Highlights)
    heat_index = cur.get("heatindex_c")
    top_metrics = {
        "Temperature": cur.get("temp_c"),
        "Feels Like": cur.get("feelslike_c") if heat_index is None else None,
        "Heat Index": heat_index if heat_index is not None else None,
        "Precipitation": f"{cur.get('precip_mm')} mm" if cur.get('precip_mm') is not None else None,
        "Humidity": f"{cur.get('humidity')}%",
        "UV Index": cur.get("uv"),
        "AQI": aqi_val, # Numerical AQI value
        "AQI_Status": aqi_status, # Internal status for LLM
        "Visibility": f"{cur.get('vis_km')} km",
        "Pressure": f"{cur.get('pressure_mb')} mb",
    }

    # Wind Block
    wind_block = {
        "Speed": f"{cur.get('wind_kph')} kph",
        "Dir": cur.get("wind_dir"),
        "Degree": f"{cur.get('wind_degree')}°",
        "Gusts": f"{cur.get('gust_kph')} kph",
    }

    # Other Metrics (The "Long List" - filtering out used keys)
    skip_keys = {
        "condition", "air_quality", "temp_c", "temp_f", "feelslike_c", "feelslike_f",
        "wind_kph", "wind_mph", "wind_degree", "wind_dir", "gust_kph", "gust_mph", 
        "vis_km", "vis_miles", "last_updated_epoch", "last_updated", "humidity", 
        "pressure_mb", "pressure_in", "cloud", "is_day",
        "precip_mm", "precip_in", "heatindex_c", "heatindex_f"
    }

    other_block = {}
    # Explicitly add some useful ones often buried
    if "cloud" in cur: other_block["Cloud Cover"] = f"{cur.get('cloud')}%"
    
    for k, v in cur.items():
        if k in skip_keys or k in other_block: continue
        if k.endswith("_f"):
            continue
        if isinstance(v, (dict, list)): continue
        if k.endswith("_c"):
            other_block[human_label(k[:-2])] = f"{v}°"
        else:
            other_block[human_label(k)] = v

    location_block["Condition"] = condition
    location_block["Air Quality"] = aq_block

    return {
        "Location": location_block,
        "TopMetrics": top_metrics,
        "Wind": wind_block,
        "Other": other_block
    }

def generate_llm_summary(data: dict) -> str:
    """Generate a conversational, intelligent summary using Mistral 675b."""
    try:
        loc = data.get("Location", {})
        city = loc.get("City") or "your location"
        
        # Create a condensed data string for the LLM
        metrics = {
            "condition": (loc.get("Condition") or {}).get("text"),
            "temp": data.get("TopMetrics", {}).get("Temperature"),
            "feels_like": data.get("TopMetrics", {}).get("Heat Index") or data.get("TopMetrics", {}).get("Feels Like"),
            "humidity": data.get("TopMetrics", {}).get("Humidity"),
            "aqi_value": data.get("TopMetrics", {}).get("AQI"),
            "aqi_status": data.get("TopMetrics", {}).get("AQI_Status")
        }
        
        prompt = (
            f"You are ZYLO, a premium digital assistant. Provide a brief, elegant weather summary for {city}. "
            "Focus on the essentials: how it feels, significant conditions, or important air quality notes. "
            "Keep it short, flowing paragraph. Avoid robotic lists; speak like a sophisticated concierge. "
            "IMPORTANT: Use only plain text. Do not use ANY Markdown formatting like asterisks (*), underscores (_), or hashes (#). "
            f"Current Data: {metrics}"
        )
        
        resp = nvidia_client.chat.completions.create(
            model="mistralai/mistral-large-3-675b-instruct-2512",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        summary = resp.choices[0].message.content.strip()
        
        # Strip Markdown artifacts and formatting characters
        import re
        summary = re.sub(r'[\*\#_]+', '', summary)
        
        # Remove quotes if present
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        return summary.strip()
    except Exception as e:
        print(f"LLM Summary Error: {e}")
        return build_weather_summary_text(data)

def fetch_weather(query=None):
    q = (query or QUERY).strip().lower()
    params = {"key": API_KEY, "q": q, "aqi": "yes"}
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    payload = format_payload(r.json())
    
    # Significant metrics for change detection
    loc = payload.get("Location", {})
    metrics = {
        "temp": payload.get("TopMetrics", {}).get("Temperature"),
        "cond": (loc.get("Condition") or {}).get("text"),
        "aqi": loc.get("Air Quality", {}).get("US EPA"),
        "city": loc.get("City")
    }
    
    cached = WEATHER_STATE_CACHE.get(q)
    if cached and cached['metrics'] == metrics:
        # Data hasn't changed significantly, use cached summary
        payload["SpeechSummary"] = cached['summary']
    else:
        # Data changed or no cache, generate new LLM summary
        new_summary = generate_llm_summary(payload)
        WEATHER_STATE_CACHE[q] = {
            "metrics": metrics,
            "summary": new_summary
        }
        payload["SpeechSummary"] = new_summary

    payload["Summary"] = build_weather_open_summary(payload)
    return payload

def build_weather_open_summary(data: dict) -> str:
    loc = data.get("Location", {}) or {}
    top = data.get("TopMetrics", {}) or {}
    wind = data.get("Wind", {}) or {}
    aq = loc.get("Air Quality", {}) or {}

    city = loc.get("City") or "Unknown city"
    region = loc.get("Region")
    condition = ((loc.get("Condition") or {}).get("text")) or "Unknown condition"
    place = f"{city}, {region}" if region else city

    temp = top.get("Temperature")
    feels = top.get("Heat Index") if top.get("Heat Index") is not None else top.get("Feels Like")
    humidity = top.get("Humidity")
    wind_speed = wind.get("Speed")
    wind_dir = wind.get("Dir")
    us_epa = aq.get("US EPA")

    lines = [f"{condition} in {place}."]

    temp_line_parts = []
    if temp is not None:
        temp_line_parts.append(f"{temp}° now")
    if feels is not None:
        temp_line_parts.append(f"feels like {feels}°")
    if temp_line_parts:
        lines.append(", ".join(temp_line_parts) + ".")

    detail_parts = []
    if humidity:
        detail_parts.append(f"Humidity {humidity}")
    if wind_speed:
        wind_text = f"Wind {wind_speed}"
        if wind_dir:
            wind_text += f" {wind_dir}"
        detail_parts.append(wind_text)
    if us_epa is not None:
        detail_parts.append(f"US EPA AQI {us_epa}")
    if detail_parts:
        lines.append(" • ".join(detail_parts) + ".")

    return "\n".join(lines)

def _is_numeric_value(value) -> bool:
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False

def _format_speech_value(key: str, value) -> str:
    text = str(value).strip()
    text = text.replace("%", " percent")
    text = text.replace("°", " degrees")
    text = text.replace(" kph", " kilometers per hour")
    text = text.replace(" km", " kilometers")
    text = text.replace(" mm", " millimeters")
    text = text.replace(" mb", " millibars")
    text = text.replace(" w/m2", " watts per square meter")
    text = text.replace(" W/m2", " watts per square meter")
    text = text.replace(" w/m²", " watts per square meter")
    text = text.replace(" W/m²", " watts per square meter")

    if any(u in text for u in (
        "percent",
        "degrees",
        "kilometers per hour",
        "kilometers",
        "millimeters",
        "millibars",
        "index",
        "watts per square meter",
    )):
        return text

    if not _is_numeric_value(value):
        return text

    key_l = (key or "").strip().lower()

    if key_l in {"temperature", "feels like", "heat index", "windchill", "dewpoint", "dew point"}:
        return f"{text} degrees celsius"
    if key_l in {"humidity", "cloud cover"}:
        return f"{text} percent"
    if key_l in {"precipitation"}:
        return f"{text} millimeters"
    if key_l in {"visibility"}:
        return f"{text} kilometers"
    if key_l in {"pressure"}:
        return f"{text} millibars"
    if key_l in {"speed", "gusts"}:
        return f"{text} kilometers per hour"
    if key_l in {"degree"}:
        return f"{text} degrees"
    if key_l in {"uv", "uv index", "us epa", "gb defra"}:
        return f"{text} index"
    if key_l in {"co", "no2", "o3", "pm10", "pm2.5", "so2"}:
        return f"{text} micrograms per cubic meter"
    if key_l in {"short rad", "diff rad", "dni", "gti"}:
        return f"{text} watts per square meter"

    return text

def build_weather_summary_text(data: dict) -> str:
    parts = []
    loc = data.get("Location", {})
    cond = loc.get("Condition", {}) or {}

    city = loc.get("City") or "Unknown city"
    region = loc.get("Region")
    country = loc.get("Country") or "Unknown country"
    location_bits = [city]
    if region:
        location_bits.append(region)
    if country:
        location_bits.append(country)
    parts.append(f"Weather report for {', '.join(location_bits)}.")

    date_str = loc.get("Date") or "--"
    time_str = loc.get("Time") or "--"
    parts.append(f"Local time is {date_str} {time_str}.")

    condition_text = cond.get("text")
    if condition_text:
        parts.append(f"Current condition is {condition_text}.")

    top = data.get("TopMetrics", {}) or {}
    top_bits = []
    for key, value in top.items():
        if value is None:
            continue
        top_bits.append(f"{key} {_format_speech_value(key, value)}")
    if top_bits:
        parts.append("Highlights: " + "; ".join(top_bits) + ".")

    wind = data.get("Wind", {}) or {}
    wind_bits = []
    for key, value in wind.items():
        if value in (None, ""):
            continue
        wind_bits.append(f"{key} {_format_speech_value(key, value)}")
    if wind_bits:
        parts.append("Wind details: " + "; ".join(wind_bits) + ".")

    aq = loc.get("Air Quality", {}) or {}
    aq_bits = []
    for key, value in aq.items():
        if value is None:
            continue
        aq_bits.append(f"{key} {_format_speech_value(key, value)}")
    if aq_bits:
        parts.append("Air quality: " + "; ".join(aq_bits) + ".")

    other = data.get("Other", {}) or {}
    other_bits = []
    for key, value in other.items():
        if value in (None, ""):
            continue
        other_bits.append(f"{key} {_format_speech_value(key, value)}")
    if other_bits:
        parts.append("Detailed metrics: " + "; ".join(other_bits) + ".")

    return " ".join(parts).strip()

def synthesize_speech_with_nvidia(text: str, voice_id: str = ENERGETIC_VOICE_ID) -> bytes:
    if not text:
        raise ValueError("No text provided for speech synthesis.")
    if not TTS_API_KEY:
        raise RuntimeError("NVIDIA_TTS_API_KEY is not configured.")
    if not os.path.exists(TTS_TALK_SCRIPT):
        raise RuntimeError(f"TTS talk.py script not found at {TTS_TALK_SCRIPT}")

    output_file = None
    try:
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
            "--output", output_file,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not os.path.exists(output_file):
            stderr = (result.stderr or "").strip() or "Unknown NVIDIA TTS error."
            raise RuntimeError(stderr)

        with open(output_file, "rb") as f:
            audio_data = f.read()
        # Some failures produce an empty/silent WAV header only.
        if len(audio_data) <= 44:
            raise RuntimeError("NVIDIA TTS produced empty audio.")
        return audio_data
    finally:
        if output_file and os.path.exists(output_file):
            os.unlink(output_file)

def synthesize_speech_with_gtts(text: str, voice_id: str = ENERGETIC_VOICE_ID) -> bytes:
    if not HAS_GTTS:
        raise RuntimeError("gTTS fallback is not available.")
    voice = next((v for v in VOICES if v["id"] == voice_id), VOICES[0])
    tts = gTTS(text=text, lang=voice["lang"], tld=voice["tld"])
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def synthesize_speech(text: str, voice_id: str = ENERGETIC_VOICE_ID) -> tuple[bytes, str]:
    try:
        return synthesize_speech_with_nvidia(text, voice_id=voice_id), "audio/wav"
    except Exception:
        audio = synthesize_speech_with_gtts(text, voice_id=voice_id)
        return audio, "audio/mpeg"

# --- HTML/CSS ---
HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZYLO FEELS</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
/* --- VARIABLES --- */
:root {
    --bg-dark: #09090b;
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --glass-shine: rgba(255, 255, 255, 0.2);
    --text-main: #ffffff;
    --text-muted: rgba(255, 255, 255, 0.6);
    --accent-grad: linear-gradient(135deg, #60a5fa, #c084fc);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg-dark);
    /* Subtle mesh gradient background */
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(96, 165, 250, 0.15), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(192, 132, 252, 0.15), transparent 25%);
    color: var(--text-main);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 20px;
}

.dashboard {
    width: 100%;
    max-width: 1000px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-top: 20px;
}

/* --- CARDS --- */
.card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

h3 {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    margin-bottom: 16px;
    font-weight: 600;
}

/* --- HEADER (Front Portion) --- */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 40px;
    position: relative;
    overflow: hidden;
}

/* Header Left: Temp & Feels Like */
.header-main {
    display: flex;
    flex-direction: column;
}
.temp-row {
    display: flex;
    align-items: flex-start;
    line-height: 1;
}
.temp-val {
    font-size: 84px;
    font-weight: 700;
    letter-spacing: -2px;
    background: linear-gradient(to bottom, #fff, #ccc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.temp-unit {
    font-size: 28px;
    font-weight: 400;
    margin-top: 12px;
    margin-left: 4px;
    color: var(--text-muted);
}
.feels-like {
    font-size: 18px;
    color: var(--text-muted);
    margin-top: 8px;
    font-weight: 300;
}

/* Header Right: Condition & Location */
.header-meta {
    text-align: right;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}
.cond-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(255,255,255,0.05);
    padding: 8px 20px 8px 12px;
    border-radius: 50px;
    border: 1px solid var(--glass-border);
    margin-bottom: 16px;
}
.cond-icon { width: 48px; height: 48px; }
.cond-text { font-size: 18px; font-weight: 600; color: #fff; }

.location-text { font-size: 24px; font-weight: 500; }
.time-text { font-size: 14px; color: var(--text-muted); margin-top: 4px; }
.controls-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 14px;
}
.summary-title {
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.summary-box {
    display: none; /* Hidden as per request */
    margin-top: 10px;
    width: min(360px, 100%);
    text-align: left;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    background: rgba(255,255,255,0.03);
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-muted);
    white-space: pre-wrap;
}
.summary-box.empty {
    font-style: italic;
}
.play-btn {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    border: 1px solid var(--glass-border);
    background: rgba(255,255,255,0.06);
    color: #fff;
    cursor: pointer;
    font-size: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.play-btn.ready {
    background: #22c55e;
    border-color: #22c55e;
    color: #0b0b0d;
}
.play-btn.connected {
    background: #16a34a;
    border-color: #16a34a;
    color: #0b0b0d;
}
.play-btn.error {
    background: #ef4444;
    border-color: #ef4444;
    color: #0b0b0d;
}
.play-btn.loading {
    color: transparent;
    position: relative;
}
.play-btn.loading::after {
    content: "";
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255,255,255,0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
.play-btn[disabled] {
    opacity: 0.6;
    cursor: not-allowed;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* --- MAIN GRID CONTENT --- */
.grid-layout {
    display: grid;
    grid-template-columns: 2fr 1fr; /* Left (Wider), Right (Narrower) */
    gap: 20px;
}

/* --- PILLS (Highlights) --- */
.pills-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
}
.pill {
    background: rgba(255,255,255,0.03);
    padding: 16px;
    border-radius: 16px;
    border: 1px solid transparent;
    transition: 0.2s;
}
.pill:hover { background: rgba(255,255,255,0.06); border-color: var(--glass-border); }
.pill-label { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.pill-val { font-size: 18px; font-weight: 600; }
#highlights.pills-container {
    display: flex;
    flex-wrap: nowrap;
    gap: 10px;
    overflow-x: auto;
    scrollbar-width: none; /* Firefox */
}
#highlights.pills-container::-webkit-scrollbar { display: none; } /* Chrome/Safari */

#highlights .pill {
    flex: 1 0 0;
    min-width: 0;
    padding: 12px 8px;
    text-align: center;
}
#highlights .pill-label { font-size: 10px; }
#highlights .pill-val { font-size: 15px; white-space: nowrap; }

/* --- COMPACT LISTS (Wind/AQ) --- */
.compact-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.list-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    font-size: 16px;
}
.list-row:last-child { margin-bottom: 0; }
.k { color: var(--text-muted); }
.v { font-weight: 600; }
/* Make Air Quality values feel closer to pill sizing */
#aq .k { font-size: 13px; letter-spacing: 0.4px; text-transform: uppercase; }
#aq .v { font-size: 18px; }
#aq .list-row {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}

/* --- OTHER STATS GRID (The Fix) --- */
/* Turning the long list into a horizontal grid of boxes */
.other-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); /* Compact cards */
    gap: 12px;
}
.mini-card {
    background: rgba(255,255,255,0.02);
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}
.mini-card .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }
.mini-card .val { font-size: 15px; font-weight: 500; color: #fff; word-break: break-word;}

/* --- RESPONSIVE --- */
@media (max-width: 768px) {
    .header { flex-direction: column; text-align: center; gap: 20px; padding: 30px 20px; }
    .header-meta { align-items: center; text-align: center; }
    .header-main { align-items: center; }
    .grid-layout { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
    body { padding: 12px; }
    .dashboard { gap: 14px; margin-top: 12px; }
    .card { padding: 18px; border-radius: 18px; }
    .header { padding: 26px 18px; }
    .temp-val { font-size: 64px; }
    .temp-unit { font-size: 22px; margin-top: 10px; }
    .feels-like { font-size: 16px; }
    .cond-badge { padding: 6px 14px 6px 10px; }
    .cond-text { font-size: 16px; }
    .location-text { font-size: 20px; }
    .time-text { font-size: 12px; }
    h3 { font-size: 12px; }
    .pills-container { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
    .pill { padding: 14px; }
    .pill-label { font-size: 11px; }
    .pill-val { font-size: 16px; }
    .other-grid { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }
    .list-row { font-size: 15px; }
}

/* --- LOCATION MODAL --- */
.modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(6px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}
.modal {
    width: min(440px, 92vw);
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.45);
}
.modal h4 {
    font-size: 16px;
    letter-spacing: 0.4px;
    margin-bottom: 12px;
}
.modal-input {
    width: 100%;
    padding: 12px 14px;
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    background: rgba(255,255,255,0.05);
    color: var(--text-main);
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    outline: none;
}
.modal-actions {
    display: flex;
    gap: 10px;
    justify-content: flex-end;
    margin-top: 12px;
}
.btn {
    padding: 8px 14px;
    border-radius: 10px;
    border: 1px solid var(--glass-border);
    background: rgba(255,255,255,0.06);
    color: var(--text-main);
    cursor: pointer;
    font-size: 13px;
}
.btn.primary {
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    color: #0b0b0d;
    font-weight: 600;
    border: none;
}
.location-text {
    cursor: pointer;
}
</style>
</head>
<body>

<div class="dashboard">

    <!-- 1. PREMIUM HEADER -->
    <div class="card header">
        <div class="header-main">
            <div class="temp-row">
                <span id="temp" class="temp-val">--</span>
                <span class="temp-unit">°</span>
            </div>
            <div class="feels-like">Feels like <span id="feels">--</span>°</div>
        </div>
        
        <div class="header-meta">
            <div class="cond-badge">
                <img id="icon" class="cond-icon" src="" alt="">
                <span id="condition" class="cond-text">--</span>
            </div>
            <div id="city" class="location-text">--</div>
            <div id="datetime" class="time-text">--</div>
            <div class="controls-row">
                <div class="summary-title">Summary</div>
                <button id="play-btn" class="play-btn" aria-label="Speak weather details">▶</button>
            </div>
            <div id="summary-box" class="summary-box empty">Summary unavailable.</div>
        </div>
    </div>

    <!-- 2. MAIN STATS AREA -->
    <div class="grid-layout">
        
        <!-- Left: Highlights & Other Stats -->
        <div style="display:flex; flex-direction:column; gap:20px;">
            <!-- Highlights -->
            <div class="card">
                <h3>Current Highlights</h3>
                <div id="highlights" class="pills-container"></div>
            </div>

            <!-- Wind (Horizontal under highlights) -->
            <div class="card">
                <h3>Wind</h3>
                <div id="wind" class="pills-container"></div>
            </div>

            <!-- THE FIX: Other Stats as a Grid -->
            <div class="card">
                <h3>Detailed Metrics</h3>
                <div id="other" class="other-grid">
                    <!-- Javascript will populate grid boxes here -->
                </div>
            </div>
        </div>

        <!-- Right: Wind & Air Quality (Sidebars) -->
        <div style="display:flex; flex-direction:column; gap:20px;">
            <div class="card">
                <h3>Air Quality</h3>
                <div id="aq" class="compact-list"></div>
            </div>
        </div>
    </div>

</div>

<!-- Location Modal -->
<div id="locModal" class="modal-backdrop" aria-hidden="true">
    <div class="modal">
        <h4>Change location</h4>
        <input id="locInput" class="modal-input" type="text" placeholder="e.g., Bolpur, India">
        <div class="modal-actions">
            <button id="locCancel" class="btn">Cancel</button>
            <button id="locSave" class="btn primary">Update</button>
        </div>
    </div>
</div>

<script>
function el(id) { return document.getElementById(id); }

function renderPill(label, value) {
    return `
    <div class="pill">
        <div class="pill-label">${label}</div>
        <div class="pill-val">${value}</div>
    </div>`;
}

function renderRow(label, value) {
    return `<div class="list-row"><span class="k">${label}</span><span class="v">${value}</span></div>`;
}

function renderMiniCard(label, value) {
    // This creates the side-wise distribution for 'Other' stats
    return `
    <div class="mini-card">
        <div class="label">${label}</div>
        <div class="val">${value}</div>
    </div>`;
}

function updateUI(data) {
    // Header
    el('temp').innerText = data.TopMetrics['Temperature'];
    const headerFeel = data.TopMetrics['Heat Index'] ?? data.TopMetrics['Feels Like'];
    el('feels').innerText = headerFeel ?? '--';
    el('condition').innerText = data.Location.Condition.text;
    el('icon').src = data.Location.Condition.icon;
    el('city').innerText = `${data.Location.City}, ${data.Location.Country}`;
    el('datetime').innerText = `${data.Location.Date} • ${data.Location.Time}`;

    // Highlights (Pills)
    let hlHtml = '';
    const hlKeys = ['Precipitation', 'Humidity', 'AQI', 'Visibility', 'Pressure'];
    hlKeys.forEach(k => {
        if (data.TopMetrics[k] !== null && data.TopMetrics[k] !== undefined) {
            hlHtml += renderPill(k, data.TopMetrics[k]);
        }
    });
    el('highlights').innerHTML = hlHtml;

    // Wind (Horizontal)
    let wHtml = '';
    for(const [k,v] of Object.entries(data.Wind)) wHtml += renderPill(k, v);
    el('wind').innerHTML = wHtml;

    // Air Quality
    let aqHtml = '';
    for(const [k,v] of Object.entries(data.Location['Air Quality'])) aqHtml += renderRow(k,v);
    el('aq').innerHTML = aqHtml;

    // Other Stats (Grid Layout)
    let otherHtml = '';
    // Add Precipitation/Cloud first if they exist in Other
    for(const [k,v] of Object.entries(data.Other)) {
        otherHtml += renderMiniCard(k, v);
    }
    el('other').innerHTML = otherHtml;

    const newSpeechText = (data.SpeechSummary || '').trim();
    
    // Only prepare/refresh audio if the summary text has actually changed
    if (newSpeechText && newSpeechText !== speechSummaryText) {
        speechSummaryText = newSpeechText;
        prepareSpeechAudio(speechSummaryText);
    }

    lastWeatherData = data;
    setSummaryText(data.Summary || buildSummaryText(data));
}

let currentQuery = '';
let lastWeatherData = null;
let summaryText = '';
let speechSummaryText = '';
let speechAudio = null;
let speechLoading = false;
const ENERGETIC_VOICE_ID = 'Magpie-Multilingual.EN-US.Aria.Happy';

async function prepareSpeechAudio(text) {
    if (speechLoading || !text) return;
    
    speechLoading = true;
    setPlayButtonState('loading');
    try {
        const blob = await requestTTSAudio(text);
        resetSpeechAudio();
        const url = URL.createObjectURL(blob);
        speechAudio = new Audio(url);
        speechAudio.addEventListener('ended', () => setPlayButtonState('idle'));
        speechAudio.addEventListener('error', () => setPlayButtonState('error'));
        setPlayButtonState('ready'); // Ready to play
    } catch (e) {
        console.error('Pre-fetch Error:', e);
        setPlayButtonState('error');
    } finally {
        speechLoading = false;
    }
}

function setSummaryText(text) {
    summaryText = (text || '').trim();
    const box = el('summary-box');
    if (!box) return;
    if (!summaryText) {
        box.innerText = 'Summary unavailable.';
        box.classList.add('empty');
        return;
    }
    box.innerText = summaryText;
    box.classList.remove('empty');
}

function formatSpeechValue(key, value) {
    const raw = String(value).trim();
    let text = raw
        .replaceAll('%', ' percent')
        .replaceAll('°', ' degrees')
        .replaceAll(' kph', ' kilometers per hour')
        .replaceAll(' km', ' kilometers')
        .replaceAll(' mm', ' millimeters')
        .replaceAll(' mb', ' millibars')
        .replaceAll(' w/m2', ' watts per square meter')
        .replaceAll(' W/m2', ' watts per square meter')
        .replaceAll(' w/m²', ' watts per square meter')
        .replaceAll(' W/m²', ' watts per square meter');

    if (
        text.includes('percent') ||
        text.includes('degrees') ||
        text.includes('kilometers per hour') ||
        text.includes('kilometers') ||
        text.includes('millimeters') ||
        text.includes('millibars') ||
        text.includes('index') ||
        text.includes('watts per square meter')
    ) {
        return text;
    }

    const numeric = !Number.isNaN(Number(raw));
    if (!numeric) return text;

    const k = String(key || '').trim().toLowerCase();
    if (['temperature', 'feels like', 'heat index', 'windchill', 'dewpoint', 'dew point'].includes(k)) return `${text} degrees celsius`;
    if (['humidity', 'cloud cover'].includes(k)) return `${text} percent`;
    if (k === 'precipitation') return `${text} millimeters`;
    if (k === 'visibility') return `${text} kilometers`;
    if (k === 'pressure') return `${text} millibars`;
    if (['speed', 'gusts'].includes(k)) return `${text} kilometers per hour`;
    if (k === 'degree') return `${text} degrees`;
    if (['uv', 'uv index', 'us epa', 'gb defra'].includes(k)) return `${text} index`;
    if (['co', 'no2', 'o3', 'pm10', 'pm2.5', 'so2'].includes(k)) return `${text} micrograms per cubic meter`;
    if (['short rad', 'diff rad', 'dni', 'gti'].includes(k)) return `${text} watts per square meter`;
    return text;
}

function buildSummaryText(data) {
    if (!data) return '';
    const parts = [];
    const loc = data.Location || {};
    const cond = (loc.Condition || {}).text;

    const locationBits = [loc.City, loc.Region, loc.Country].filter(Boolean);
    if (locationBits.length) parts.push(`Weather report for ${locationBits.join(', ')}.`);
    if (loc.Date || loc.Time) parts.push(`Local time is ${loc.Date || '--'} ${loc.Time || '--'}.`);
    if (cond) parts.push(`Current condition is ${cond}.`);

    const sections = [
        ['Highlights', data.TopMetrics || {}],
        ['Wind details', data.Wind || {}],
        ['Air quality', (loc['Air Quality'] || {})],
        ['Detailed metrics', data.Other || {}],
    ];
    sections.forEach(([title, entries]) => {
        const values = [];
        Object.entries(entries).forEach(([k, v]) => {
            if (v === null || v === undefined || v === '') return;
            values.push(`${k} ${formatSpeechValue(k, v)}`);
        });
        if (values.length) parts.push(`${title}: ${values.join('; ')}.`);
    });

    return parts.join(' ').trim();
}

async function requestTTSAudio(text) {
    const endpoints = ['/zenith/generate_audio', '/api/speak'];
    let lastError = null;

    for (const endpoint of endpoints) {
        try {
            const r = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    voice_id: ENERGETIC_VOICE_ID,
                }),
            });
            if (!r.ok) {
                let errMsg = `${endpoint} failed (${r.status})`;
                try {
                    const err = await r.json();
                    if (err?.error) errMsg = err.error;
                } catch (_) {}
                throw new Error(errMsg);
            }
            return await r.blob();
        } catch (e) {
            lastError = e;
        }
    }

    throw lastError || new Error('All TTS endpoints failed.');
}

function setPlayButtonState(state) {
    const playBtn = el('play-btn');
    if (!playBtn) return;
    playBtn.classList.remove('loading', 'connected', 'error', 'ready');
    playBtn.removeAttribute('disabled');

    if (state === 'loading') {
        playBtn.classList.add('loading');
        playBtn.setAttribute('disabled', 'true');
        playBtn.innerText = '▶';
        return;
    }
    if (state === 'playing') {
        playBtn.classList.add('connected');
        playBtn.innerText = '❚❚';
        return;
    }
    if (state === 'ready') {
        playBtn.classList.add('ready');
        playBtn.innerText = '▶';
        return;
    }
    if (state === 'error') {
        playBtn.classList.add('error');
        playBtn.innerText = '!';
        return;
    }
    playBtn.innerText = '▶';
}

function resetSpeechAudio() {
    if (!speechAudio) return;
    speechAudio.pause();
    if (speechAudio.src && speechAudio.src.startsWith('blob:')) {
        URL.revokeObjectURL(speechAudio.src);
    }
    speechAudio = null;
}

async function speakSummary() {
    if (speechLoading) return;

    if (speechAudio && !speechAudio.paused) {
        speechAudio.pause();
        setPlayButtonState('ready');
        return;
    }
    
    if (speechAudio) {
        try {
            await speechAudio.play();
            setPlayButtonState('playing');
            return;
        } catch (e) {
            console.error('Play Error:', e);
            // If play fails (e.g. user hasn't interacted), we might need to re-fetch or try again
            setPlayButtonState('error');
        }
    }

    // If no pre-fetched audio, fetch it now
    const textToSpeak = (speechSummaryText || summaryText || (lastWeatherData ? buildSummaryText(lastWeatherData) : '')).trim();
    if (!textToSpeak) {
        setPlayButtonState('error');
        return;
    }

    speechLoading = true;
    setPlayButtonState('loading');
    try {
        const blob = await requestTTSAudio(textToSpeak);
        resetSpeechAudio();
        const url = URL.createObjectURL(blob);
        speechAudio = new Audio(url);
        speechAudio.addEventListener('ended', () => setPlayButtonState('idle'));
        speechAudio.addEventListener('error', () => setPlayButtonState('error'));
        await speechAudio.play();
        setPlayButtonState('playing');
    } catch (e) {
        console.error(e);
        setPlayButtonState('error');
    } finally {
        speechLoading = false;
    }
}

function loadData() {
    const url = currentQuery ? `/api/weather?q=${encodeURIComponent(currentQuery)}` : '/api/weather';
    fetch(url)
        .then(r => r.json())
        .then(d => {
            if (d && d.error) throw new Error(d.error);
            updateUI(d);
        })
        .catch(e => console.error(e));
}

function openModal() {
    const modal = el('locModal');
    const input = el('locInput');
    input.value = el('city').innerText || 'Bolpur, India';
    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');
    setTimeout(() => input.focus(), 0);
}
function closeModal() {
    const modal = el('locModal');
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
}
function saveLocation() {
    const input = el('locInput');
    const next = (input.value || '').trim();
    if (!next) return;
    currentQuery = next;
    closeModal();
    loadData();
}

el('city').addEventListener('click', openModal);
el('locCancel').addEventListener('click', closeModal);
el('locSave').addEventListener('click', saveLocation);
el('locInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') saveLocation();
    if (e.key === 'Escape') closeModal();
});
el('locModal').addEventListener('click', (e) => {
    if (e.target === el('locModal')) closeModal();
});

el('play-btn').addEventListener('click', speakSummary);
loadData();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML, refresh=REFRESH_SECONDS)

@app.route("/api/weather")
def api_weather():
    try:
        query = request.args.get("q") or None
        return jsonify(fetch_weather(query=query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/speak", methods=["POST"])
@app.route("/zenith/generate_audio", methods=["POST"])
def api_speak():
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        voice_id = (payload.get("voice_id") or ENERGETIC_VOICE_ID).strip() or ENERGETIC_VOICE_ID
        if not text:
            data = payload.get("weather")
            if not data:
                query = request.args.get("q") or None
                data = fetch_weather(query=query)
            text = (data.get("SpeechSummary") or build_weather_summary_text(data)).strip()
        audio, mimetype = synthesize_speech(text, voice_id=voice_id)
        return Response(audio, mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    app.run(debug=False, port=5000)
