from flask import Flask, render_template, request, session, redirect, url_for, flash, g, jsonify
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import traceback
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re
import requests
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
import pathlib
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
from difflib import get_close_matches
import uuid

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

try:
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

load_dotenv()
# Configure Gemini API key if present; track availability
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("✓ Gemini API configured successfully")
    except Exception as e:
        print("✗ Warning: failed to configure Gemini client:", e)
        GEMINI_AVAILABLE = False
else:
    print("✗ Warning: GEMINI_API_KEY not set. Gemini placement lookups disabled.")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY') or 'dev-secret-key'

# ───────────────────────────────────────────────
# Configuration & Model Loading
# ───────────────────────────────────────────────
MODEL_DIR = "saved_models"
DB_PATH = os.path.join('instance', 'app.sqlite')
DB_URI = f"sqlite:///{pathlib.Path(DB_PATH).resolve().as_posix()}"
LANGCHAIN_CHAT_TABLE = 'langchain_chat_messages'

try:
    ensemble_model   = joblib.load(os.path.join(MODEL_DIR, 'ensemble_model.pkl'))
    scaler           = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le_dict          = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    target_le        = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
    feature_columns  = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))

# Ensure instance dir exists and DB is initialized
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db():
    if 'db' not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cur = db.cursor()
    # users table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')
    # predictions history
    cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        input_json TEXT,
        prediction TEXT,
        top_predictions_json TEXT,
        fees_info_json TEXT,
        placements_info_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    # chatbot session metadata (LangChain-style session grouping)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_key TEXT NOT NULL,
        user_id INTEGER,
        session_key TEXT UNIQUE NOT NULL,
        title TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_message_at TEXT,
        is_active INTEGER NOT NULL DEFAULT 1
    )''')
    # fallback storage in case LangChain package is unavailable
    cur.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_key TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')
    db.commit()

@app.teardown_appcontext
def teardown_db(exception):
    close_db()

# Create DB schema immediately (safe without app context)
def init_db_schema():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_json TEXT,
            prediction TEXT,
            top_predictions_json TEXT,
            fees_info_json TEXT,
            placements_info_json TEXT,
            created_at TEXT NOT NULL
        )''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_key TEXT NOT NULL,
            user_id INTEGER,
            session_key TEXT UNIQUE NOT NULL,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_message_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        )''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        conn.commit()
        conn.close()
        print('✓ Database schema ensured')
    except Exception as e:
        print('✗ Database schema initialization failed:', e)

# ensure DB schema exists now
init_db_schema()

def normalize_college_key(name):
    """Normalize strings for matching by removing non-alphanumeric chars and lowercasing."""
    return re.sub(r'[^a-z0-9]+', '', str(name or '').lower())

# Load college dataset for fee & placement lookups and predictions
CSV_PATH = "complete_engineering_colleges_dataset.csv"
GENERATED_DETAILS_CSV_PATH = "college_details_generated.csv"

try:
    colleges_df = pd.read_csv(CSV_PATH)
    if not colleges_df.empty and 'college_name' in colleges_df.columns:
        # Globally exclude unwanted branches from the entire application
        if 'branch' in colleges_df.columns:
            colleges_df = colleges_df[~colleges_df['branch'].astype(str).str.lower().isin(['aerospace', 'biotech'])]
        
        colleges_df['_name_lower'] = colleges_df['college_name'].astype(str).str.lower()
except Exception:
    colleges_df = pd.DataFrame()
    print("Warning: could not load college dataset for fees/placements lookup")

try:
    generated_details_df = pd.read_csv(GENERATED_DETAILS_CSV_PATH)
    if not generated_details_df.empty and 'college_name' in generated_details_df.columns:
        generated_details_df['_name_lower'] = generated_details_df['college_name'].astype(str).str.lower()
        generated_details_df['_name_key'] = generated_details_df['college_name'].astype(str).map(normalize_college_key)
        GENERATED_CHOICES = generated_details_df['college_name'].astype(str).tolist()
    else:
        GENERATED_CHOICES = []
except Exception:
    generated_details_df = pd.DataFrame()
    GENERATED_CHOICES = []
    print("Warning: could not load generated college details dataset")

# Placements cache helper
PLACEMENTS_CACHE = os.path.join(MODEL_DIR, 'placements_cache.json')
COLLEGE_DETAILS_CACHE = os.path.join(MODEL_DIR, 'college_details_cache.json')
COLLEGE_DETAILS_CACHE_VERSION = 2
COLLEGE_DETAILS_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

PREFERRED_SOURCE_DOMAINS = [
    'shiksha.com',
    'collegedunia.com',
    'careers360.com',
    'collegepravesh.com',
    'collegesearch.in',
    '.edu',
    '.ac.in'
]

SOURCE_SIGNAL_KEYWORDS = [
    'placement', 'placements', 'package', 'ctc', 'recruiter',
    'fees', 'tuition', 'hostel', 'mess', 'affiliation', 'naac', 'nirf'
]

REQUEST_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    )
}

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _domain_from_url(url):
    try:
        host = (urlparse(url).netloc or '').lower().strip()
        if host.startswith('www.'):
            host = host[4:]
        return host
    except Exception:
        return ''

def _source_score(item):
    title = normalize_spaces(item.get('title', '')).lower()
    snippet = normalize_spaces(item.get('snippet', '')).lower()
    url = item.get('url', '')
    domain = _domain_from_url(url)

    score = 0

    # 🔥 PRIORITY BOOST
    if 'shiksha.com' in domain:
        score += 35
    elif 'collegedunia.com' in domain:
        score += 35
    elif 'careers360.com' in domain:
        score += 30
    elif '.edu' in domain or '.ac.in' in domain:
        score += 40

    # official keyword
    if 'official' in title or 'official' in snippet:
        score += 10

    # keyword boost
    for kw in SOURCE_SIGNAL_KEYWORDS:
        if kw in title:
            score += 5
        if kw in snippet:
            score += 3

    # penalty
    if any(x in domain for x in ['youtube', 'instagram', 'facebook']):
        score -= 25

    return score

def _is_stale(ts_value, ttl_seconds):
    if not ts_value:
        return True
    try:
        ts = datetime.fromisoformat(str(ts_value).replace('Z', '+00:00'))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
        return age_seconds > ttl_seconds
    except Exception:
        return True

def _extract_relevant_excerpt(raw_text, max_chars=3000):
    text = normalize_spaces(raw_text)

    if not text:
        return None

    lines = re.split(r'[.!?]', text)

    filtered = []

    for line in lines:
        line_lower = line.lower()

        if any(k in line_lower for k in SOURCE_SIGNAL_KEYWORDS):
            filtered.append(line.strip())

    if not filtered:
        return text[:max_chars]

    final_text = '. '.join(filtered)

    return final_text[:max_chars]

def load_json_cache(cache_path):
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_json_cache(cache_path, cache_data):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_placements_cache():
    return load_json_cache(PLACEMENTS_CACHE)

def save_placements_cache(cache):
    save_json_cache(PLACEMENTS_CACHE, cache)

def load_college_details_cache():
    return load_json_cache(COLLEGE_DETAILS_CACHE)

def save_college_details_cache(cache):
    save_json_cache(COLLEGE_DETAILS_CACHE, cache)

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()

def _get_best_generated_match(college_name):
    if generated_details_df.empty or 'college_name' not in generated_details_df.columns:
        return None

    target = normalize_spaces(college_name)
    if not target:
        return None

    target_key = normalize_college_key(target)

    if '_name_key' in generated_details_df.columns:
        exact = generated_details_df[generated_details_df['_name_key'] == target_key]
        if not exact.empty:
            return exact.iloc[0]

    if '_name_lower' in generated_details_df.columns:
        contains = generated_details_df[generated_details_df['_name_lower'].str.contains(target.lower(), na=False, regex=False)]
        if not contains.empty:
            return contains.iloc[0]

    if RAPIDFUZZ_AVAILABLE and getattr(sys.modules[__name__], 'GENERATED_CHOICES', None):
        result = rf_process.extractOne(target, GENERATED_CHOICES, scorer=rf_fuzz.token_sort_ratio)
        if result and len(result) >= 2 and result[1] >= 75:
            row = generated_details_df[generated_details_df['college_name'] == result[0]]
            if not row.empty:
                return row.iloc[0]
    elif getattr(sys.modules[__name__], 'GENERATED_CHOICES', None):
        close = get_close_matches(target, GENERATED_CHOICES, n=1, cutoff=0.72)
        if close:
            row = generated_details_df[generated_details_df['college_name'] == close[0]]
            if not row.empty:
                return row.iloc[0]

    return None

def _num_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r'\d+(?:\.\d+)?', str(value).replace(',', ''))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None

def get_college_details_from_generated_dataset(college_name):
    row = _get_best_generated_match(college_name)
    if row is None:
        return None

    placement_rate = _num_or_none(row.get('placement_rate_percent'))
    avg_pkg = _num_or_none(row.get('average_package_lpa'))
    high_pkg = _num_or_none(row.get('highest_package_lpa'))
    overall_fees = _num_or_none(row.get('overall_fees'))
    hostel_fees = _num_or_none(row.get('hostel_fees'))

    summary_parts = []
    if placement_rate is not None:
        summary_parts.append(f"Placement rate around {placement_rate:.1f}%")
    if avg_pkg is not None:
        summary_parts.append(f"average package {avg_pkg:.2f} LPA")
    if high_pkg is not None:
        summary_parts.append(f"highest package {high_pkg:.2f} LPA")

    return {
        'college_details': {
            'official_website': None,
            'location': normalize_spaces(row.get('location')) or None,
            'college_type': normalize_spaces(row.get('college_type')) or None,
            'approval_or_affiliation': normalize_spaces(row.get('affiliation')) or None
        },
        'placement_details': {
            'summary': ', '.join(summary_parts) if summary_parts else None,
            'placement_rate_percent': placement_rate,
            'average_package_lpa': avg_pkg,
            'highest_package_lpa': high_pkg,
            'top_recruiters': []
        },
        'fees_details': {
            'tuition_fee_annual_inr': None,
            'overall_program_fee_inr': overall_fees,
            'hostel_fee_annual_inr': hostel_fees,
            'notes': 'From generated dataset (college_details_generated.csv).'
        },
        'sources': [],
        'source_label': 'Generated dataset',
        'fetched_at': utc_now_iso()
    }

def merge_detail_payloads(primary, secondary):
    if not primary:
        return secondary
    if not secondary:
        return primary

    def pick(a, b):
        return a if a not in (None, '', [], {}) else b

    merged = {
        'college_details': {},
        'placement_details': {},
        'fees_details': {},
        'sources': (primary.get('sources') or []) + (secondary.get('sources') or []),
        'source_label': 'Generated dataset + Web/Gemini',
        'fetched_at': utc_now_iso()
    }

    for key in ['official_website', 'location', 'college_type', 'approval_or_affiliation']:
        merged['college_details'][key] = pick(primary.get('college_details', {}).get(key), secondary.get('college_details', {}).get(key))

    for key in ['summary', 'placement_rate_percent', 'average_package_lpa', 'highest_package_lpa', 'top_recruiters']:
        merged['placement_details'][key] = pick(primary.get('placement_details', {}).get(key), secondary.get('placement_details', {}).get(key))

    for key in ['tuition_fee_annual_inr', 'overall_program_fee_inr', 'hostel_fee_annual_inr', 'notes']:
        merged['fees_details'][key] = pick(primary.get('fees_details', {}).get(key), secondary.get('fees_details', {}).get(key))

    return merged

def build_college_full_details(college_name):
    """Build complete college details payload for UI/API usage."""
    generated_details = get_college_details_from_generated_dataset(college_name)
    web_details = fetch_college_details_from_web(college_name)
    merged_details = merge_detail_payloads(generated_details, web_details)

    college_details = (merged_details or {}).get('college_details') or {}
    placement_details = (merged_details or {}).get('placement_details') or {}
    fees_details = (merged_details or {}).get('fees_details') or {}
    detail_sources = (merged_details or {}).get('sources') or []
    details_source_label = (merged_details or {}).get('source_label')
    details_fetched_at = (merged_details or {}).get('fetched_at')

    placements_info = placement_details.get('summary') if placement_details else None
    placements_source = details_source_label if placements_info else None

    if not colleges_df.empty and '_name_lower' in colleges_df.columns:
        target_lower = str(college_name).lower()
        mask_exact = colleges_df['_name_lower'] == target_lower
        mask_contains = colleges_df['_name_lower'].str.contains(target_lower, na=False, regex=False)
        matched = colleges_df[mask_exact]
        if matched.empty:
            matched = colleges_df[mask_contains]

        if not matched.empty:
            fees_mean = matched['college_fees'].dropna().mean()
            if fees_details.get('tuition_fee_annual_inr') is None and not np.isnan(fees_mean):
                fees_details['tuition_fee_annual_inr'] = float(fees_mean)

            if 'placements' in matched.columns and placements_info is None:
                placements_vals = matched['placements'].dropna()
                if not placements_vals.empty:
                    try:
                        placements_mean = float(placements_vals.astype(float).mean())
                        placements_info = f"Average placement rate: {placements_mean:.1f}%"
                        placement_details['summary'] = placements_info
                        placement_details['placement_rate_percent'] = placements_mean
                    except Exception:
                        placements_info = str(placements_vals.iloc[0])
                        placement_details['summary'] = placements_info
                    placements_source = 'Dataset'

    if placements_info is None:
        gemini_summary = fetch_placements_with_gemini(college_name)
        if gemini_summary:
            placements_info = gemini_summary
            placement_details['summary'] = gemini_summary
            placements_source = 'Gemini (approx)'

    return {
        'college': college_name,
        'college_details': college_details,
        'placement_details': placement_details,
        'fees_details': fees_details,
        'detail_sources': detail_sources,
        'details_source_label': details_source_label,
        'details_fetched_at': details_fetched_at,
        'placements_info': placements_info,
        'placements_source': placements_source
    }

def _resolve_duckduckgo_link(href):
    if not href:
        return None
    if href.startswith('http://') or href.startswith('https://'):
        return href
    if href.startswith('/l/?'):
        try:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if 'uddg' in qs and qs['uddg']:
                return unquote(qs['uddg'][0])
        except Exception:
            return None
    return None

def search_web_sources(college_name, max_results=6):
    url = 'https://duckduckgo.com/html/'
    results = []
    seen = set()

    # 🔥 UPDATED QUERIES (more targeted)
    queries = [
        f"site:shiksha.com {college_name} fees placements hostel",
        f"site:collegedunia.com {college_name} fees placements hostel",
        f"site:careers360.com {college_name} fees placements hostel",
        f"{college_name} official website placements fees hostel"
    ]

    try:
        for query in queries:
            resp = requests.get(url, params={'q': query}, headers=REQUEST_HEADERS, timeout=10)

            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            for row in soup.select('.result'):
                link = row.select_one('a.result__a')
                snippet_el = row.select_one('.result__snippet')

                if not link:
                    continue

                href = _resolve_duckduckgo_link(link.get('href'))
                if not href or href in seen:
                    continue

                seen.add(href)

                title = normalize_spaces(link.get_text())
                snippet = normalize_spaces(snippet_el.get_text() if snippet_el else '')

                results.append({
                    'title': title,
                    'url': href,
                    'snippet': snippet
                })

        # 🔥 better ranking
        results = sorted(results, key=_source_score, reverse=True)

        return results[:max_results]

    except Exception as e:
        print("Search error:", e)
        return []

def fetch_page_excerpt(url, max_chars=5000):
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'lxml')
        for tag in soup(['script', 'style', 'noscript', 'svg']):
            tag.decompose()
        text = normalize_spaces(soup.get_text(' ', strip=True))
        if not text:
            return None
        relevant = _extract_relevant_excerpt(text, max_chars=max_chars)
        return relevant
    except Exception:
        # Fallback parser in case lxml is unavailable at runtime.
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'noscript', 'svg']):
                tag.decompose()
            text = normalize_spaces(soup.get_text(' ', strip=True))
            return _extract_relevant_excerpt(text, max_chars=max_chars)
        except Exception:
            return None

def _is_useful_detail_payload(payload):
    if not payload:
        return False
    college = payload.get('college_details') or {}
    placement = payload.get('placement_details') or {}
    fees = payload.get('fees_details') or {}

    checks = [
        bool(college.get('official_website')),
        bool(college.get('location')),
        bool(placement.get('summary')),
        placement.get('placement_rate_percent') is not None,
        fees.get('tuition_fee_annual_inr') is not None,
        fees.get('hostel_fee_annual_inr') is not None,
    ]
    return sum(1 for c in checks if c) >= 2

def _enforce_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).replace(',', '').strip()
    match = re.search(r'\d+(?:\.\d+)?', raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None

def extract_json_object(text):
    payload = str(text or '').strip()
    if not payload:
        return None
    if payload.startswith('```'):
        payload = re.sub(r'^```(?:json)?\s*', '', payload)
        payload = re.sub(r'\s*```$', '', payload)
    try:
        return json.loads(payload)
    except Exception:
        pass

    match = re.search(r'\{.*\}', payload, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def fetch_college_details_from_web(college_name):
    """Return structured web-sourced college info with cache.

    Fields: college_details, placement_details, fees_details, sources.
    """
    key = normalize_spaces(college_name)
    if not key:
        return None

    cache = load_college_details_cache()
    cached = cache.get(key)
    if cached:
        cached_version = int(cached.get('cache_version', 1))
        cached_ts = cached.get('fetched_at')
        if cached_version >= COLLEGE_DETAILS_CACHE_VERSION and not _is_stale(cached_ts, COLLEGE_DETAILS_CACHE_TTL_SECONDS):
            return cached

    sources = search_web_sources(key, max_results=6)
    if not sources:
        return None

    enriched_sources = []
    for item in sources[:4]:
        excerpt = fetch_page_excerpt(item['url'])
        if not excerpt:
            continue
        enriched_sources.append({
            'title': item.get('title'),
            'url': item.get('url'),
            'snippet': item.get('snippet'),
            'excerpt': excerpt
        })

    if not enriched_sources:
        return None

    if not GEMINI_AVAILABLE:
        # No LLM configured; still return sources for transparency.
        payload = {
            'college_details': {
                'official_website': None,
                'location': None,
                'college_type': None,
                'approval_or_affiliation': None
            },
            'placement_details': {
                'summary': None,
                'placement_rate_percent': None,
                'average_package_lpa': None,
                'highest_package_lpa': None,
                'top_recruiters': []
            },
            'fees_details': {
                'tuition_fee_annual_inr': None,
                'overall_program_fee_inr': None,
                'hostel_fee_annual_inr': None,
                'notes': None
            },
            'sources': [{'title': s.get('title'), 'url': s.get('url')} for s in enriched_sources],
            'source_label': 'Web search',
            'cache_version': COLLEGE_DETAILS_CACHE_VERSION,
            'fetched_at': utc_now_iso()
        }
        cache[key] = payload
        save_college_details_cache(cache)
        return payload

    prompt = (
         f"""
You are a strict data extraction engine.

Extract ONLY factual information from the given sources.
Do NOT guess or assume anything.

Return ONLY valid JSON.

{{
  "college_details": {{
    "official_website": string|null,
    "location": string|null,
    "college_type": string|null,
    "approval_or_affiliation": string|null
  }},
  "placement_details": {{
    "summary": string|null,
    "placement_rate_percent": number|null,
    "average_package_lpa": number|null,
    "highest_package_lpa": number|null,
    "top_recruiters": [string]
  }},
  "fees_details": {{
    "tuition_fee_annual_inr": number|null,
    "overall_program_fee_inr": number|null,
    "hostel_fee_annual_inr": number|null
  }},
  "sources": [{{"title": string, "url": string}}]
}}

College: {key}

Sources:
{json.dumps(enriched_sources, ensure_ascii=False)}
"""
    )

    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        parsed = extract_json_object(response.text if response else '')
        if not parsed:
            return None

        payload = {
            'college_details': parsed.get('college_details') or {},
            'placement_details': parsed.get('placement_details') or {},
            'fees_details': parsed.get('fees_details') or {},
            'sources': parsed.get('sources') or [{'title': s.get('title'), 'url': s.get('url')} for s in enriched_sources],
            'source_label': 'Web + Gemini extraction',
            'cache_version': COLLEGE_DETAILS_CACHE_VERSION,
            'fetched_at': utc_now_iso()
        }

        payload['placement_details']['placement_rate_percent'] = _enforce_numeric(payload['placement_details'].get('placement_rate_percent'))
        payload['placement_details']['average_package_lpa'] = _enforce_numeric(payload['placement_details'].get('average_package_lpa'))
        payload['placement_details']['highest_package_lpa'] = _enforce_numeric(payload['placement_details'].get('highest_package_lpa'))
        payload['fees_details']['tuition_fee_annual_inr'] = _enforce_numeric(payload['fees_details'].get('tuition_fee_annual_inr'))
        payload['fees_details']['overall_program_fee_inr'] = _enforce_numeric(payload['fees_details'].get('overall_program_fee_inr'))
        payload['fees_details']['hostel_fee_annual_inr'] = _enforce_numeric(payload['fees_details'].get('hostel_fee_annual_inr'))

        if not _is_useful_detail_payload(payload):
            return None

        cache[key] = payload
        save_college_details_cache(cache)
        return payload
    except Exception:
        return None

def fetch_placements_with_gemini(college_name):
    """Fetch a short placement summary for a college using Gemini (cached).

    Returns a string summary or None if not available.
    """
    if not college_name:
        return None

    cache = load_placements_cache()
    key = str(college_name).strip()
    if key in cache:
        return cache[key].get('summary')

    # require Gemini availability configured at startup
    if not GEMINI_AVAILABLE:
        print("Gemini API key not available; skipping external lookup for:", college_name)
        return None

    prompt = (
        f"You are a concise, factual assistant. Provide a short (1-3 sentence) summary of placement "
        f"statistics for the Indian engineering college named \"{college_name}\". Include approximate "
        "placement rate (percent), top recruiters, and average/highest package if known. "
        "If you do not know or data is not available, reply with the single word: unknown. "
        "Do not fabricate data; be explicit when information is approximate."
    )

    try:
        print(f"  [Gemini] Fetching placements for: {college_name}")
        text = None

        # Try with latest Gemini model
        try:
            model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                text = response.text
                print(f"  [Gemini] ✓ Got response from gemini-2.5-flash-lite")
        except Exception as e:
            print(f"  [Gemini] Models/gemini-2.5-flash failed: {e}")

        # Fallback to older Gemini model
        if not text:
            try:
                model = genai.GenerativeModel('models/gemini-2.5-flash')
                response = model.generate_content(prompt)
                if response and hasattr(response, 'text') and response.text:
                    text = response.text
                    print(f"  [Gemini] ✓ Got response from gemini-2.5-flash")
            except Exception as e:
                print(f"  [Gemini] gemini-2.5-flash failed: {e}")

        # Final fallback: stringify anything we got
        if not text:
            print(f"  [Gemini] ✗ No text extracted from response")
            return None

        summary = str(text).strip()
        # standardize unknown-like replies
        if not summary or summary.lower() in ('unknown', "i don't know", 'not available', 'no data', 'no information'):
            print(f"  [Gemini] ✗ Got 'unknown' response")
            return None

        print(f"✓ Placement fetched for {college_name}: {summary[:50]}...")

        # store in cache with timestamp
        cache[key] = {'summary': summary, 'ts': utc_now_iso()}
        save_placements_cache(cache)
        return summary

    except Exception as e:
        print(f"✗ Gemini placement lookup error for {college_name}: {e}")
        traceback.print_exc()
        return None

# District name normalization / fallback
DISTRICT_MAP = {
    'tiruchirappalli': 'Trichy',
    'Tiruchirappalli': 'Trichy',
    'Trichy': 'Trichy',
    'tiruchy': 'Trichy',
    'Tiruchy': 'Trichy',
    # add more known variations if needed after checking encoder classes
}

# Keep legacy function for backward compatibility (history page)
def predict_single_student(input_dict, top_n=3):
    user_cut_off = float(input_dict.get('cut_off', 0))
    user_branch = str(input_dict.get('branch', '')).strip()
    user_category = str(input_dict.get('category', '')).strip()
    user_district = str(input_dict.get('district', '')).strip()
    if colleges_df.empty:
        return "Dataset not found", []
    mask = (colleges_df['category'].str.upper() == user_category.upper()) & \
           (colleges_df['branch'].str.upper() == user_branch.upper())
    filtered = colleges_df[mask].copy()
    if filtered.empty:
        mask = (colleges_df['category'].str.upper() == user_category.upper())
        filtered = colleges_df[mask].copy()
    if filtered.empty:
        return "No Matching College Found", []
    filtered['gap'] = user_cut_off - filtered['cut_off']
    filtered['abs_gap'] = filtered['gap'].abs()
    realistic = filtered[filtered['gap'] >= -15.0].copy()
    if realistic.empty:
        realistic = filtered.copy()
    if user_district and user_district.lower() not in ['all', 'others', 'any']:
        realistic = realistic[realistic['district'].str.lower() == user_district.lower()]
        if realistic.empty:
            return f"No Matching College Found in {user_district}", []
    realistic = realistic.sort_values(by=['abs_gap'], ascending=[True])
    realistic = realistic.drop_duplicates(subset=['college_name'], keep='first')
    top_colleges_df = realistic.head(top_n)
    results = []
    for _, row in top_colleges_df.iterrows():
        gap = float(row['gap'])
        if gap >= 2: prob = min(0.80, 0.75 + (gap - 2) * 0.01)
        elif gap >= 0: prob = 0.68 + (gap * 0.035)
        elif gap >= -2: prob = 0.55 + ((gap + 2) * 0.065)
        else: prob = max(0.15, 0.55 + (gap * 0.05))
        results.append((str(row['college_name']), prob))
    best = results[0][0] if results else "No Match"
    return best, results


# ──────────────────────────────────────────────────────────────
# CollegeDP-style Prediction Engine
# ──────────────────────────────────────────────────────────────

def _compute_branch_chance(user_cutoff, branch_min, branch_avg, branch_max):
    """Compute admission chance % for a single branch given user cutoff and branch stats."""
    if user_cutoff >= branch_max:
        # Well above the highest cutoff — very high chance
        overshoot = user_cutoff - branch_max
        chance = min(99, 85 + overshoot * 0.5)
    elif user_cutoff >= branch_avg:
        # Between avg and max — moderate-high chance
        span = max(branch_max - branch_avg, 0.1)
        ratio = (user_cutoff - branch_avg) / span
        chance = 55 + ratio * 30  # 55% to 85%
    elif user_cutoff >= branch_min:
        # Between min and avg — moderate-low chance
        span = max(branch_avg - branch_min, 0.1)
        ratio = (user_cutoff - branch_min) / span
        chance = 25 + ratio * 30  # 25% to 55%
    else:
        # Below minimum — low chance
        deficit = branch_min - user_cutoff
        chance = max(2, 25 - deficit * 1.5)
    return round(chance)


def _chance_level(chance_pct):
    """Map chance percentage to a level string."""
    if chance_pct >= 60:
        return 'high'
    elif chance_pct >= 30:
        return 'moderate'
    return 'low'


def predict_colleges(user_cutoff, category, district=None, sort_by='best_match', page=1, per_page=25, max_fee=None, search_query=None):
    """
    CollegeDP-style prediction engine.

    Returns:
        dict with keys: colleges (list), total, page, per_page, total_pages
    """
    if colleges_df.empty:
        return {'colleges': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

    # 1. Filter by category
    cat_upper = str(category).strip().upper()
    filtered = colleges_df[colleges_df['category'].str.upper() == cat_upper].copy()
    if filtered.empty:
        return {'colleges': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

    # 1b. Exclude unwanted branches (now handled globally on load, but keeping safe guard)

    # 2. Optional district filter
    if district and str(district).strip().lower() not in ('', 'any', 'all', 'any district'):
        dist_lower = str(district).strip().lower()
        filtered = filtered[filtered['district'].str.lower() == dist_lower]
        if filtered.empty:
            return {'colleges': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

    # 2a. Optional search name filter (for filtering results)
    if search_query and str(search_query).strip():
        sq = str(search_query).strip().lower()
        filtered = filtered[filtered['college_name'].str.lower().str.contains(sq, na=False)]
        if filtered.empty:
            return {'colleges': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

    # 2b. Optional budget filter — exclude colleges where ALL fees exceed budget
    if max_fee and max_fee > 0 and 'college_fees' in filtered.columns:
        # Keep rows where fee is within budget (or fee is unknown)
        fee_mask = (filtered['college_fees'].isna()) | (filtered['college_fees'] <= max_fee)
        budget_filtered = filtered[fee_mask]
        # Only apply if at least some results survive
        if not budget_filtered.empty:
            filtered = budget_filtered
    # 3. Group by college, then by branch — compute min/avg/max
    grouped = filtered.groupby(['college_name', 'branch']).agg(
        cut_min=('cut_off', 'min'),
        cut_avg=('cut_off', 'mean'),
        cut_max=('cut_off', 'max'),
        fee_min=('college_fees', 'min'),
        fee_max=('college_fees', 'max'),
        district=('district', 'first'),
    ).reset_index()

    # 4. Compute per-branch chance
    grouped['chance'] = grouped.apply(
        lambda r: _compute_branch_chance(user_cutoff, r['cut_min'], r['cut_avg'], r['cut_max']),
        axis=1
    )
    grouped['chance_level'] = grouped['chance'].apply(_chance_level)
    grouped['cut_avg'] = grouped['cut_avg'].round(2)

    # 5. Build college-level aggregation
    college_agg = grouped.groupby('college_name').agg(
        best_chance=('chance', 'max'),
        avg_chance=('chance', 'mean'),
        avg_cutoff=('cut_avg', 'mean'),
        fee_min=('fee_min', 'min'),
        fee_max=('fee_max', 'max'),
        district=('district', 'first'),
        branch_count=('branch', 'count'),
    ).reset_index()

    # Determine overall chance level for each college (from best branch)
    college_agg['chance_level'] = college_agg['best_chance'].apply(_chance_level)

    # 6. Sorting
    if sort_by == 'cutoff_high':
        college_agg = college_agg.sort_values('avg_cutoff', ascending=False)
    elif sort_by == 'cutoff_low':
        college_agg = college_agg.sort_values('avg_cutoff', ascending=True)
    elif sort_by == 'fees_low':
        college_agg = college_agg.sort_values('fee_min', ascending=True)
    elif sort_by == 'fees_high':
        college_agg = college_agg.sort_values('fee_max', ascending=False)
    else:  # best_match
        college_agg = college_agg.sort_values('best_chance', ascending=False)

    total = len(college_agg)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    page_colleges = college_agg.iloc[start:end]

    # 7. Build response
    results = []
    for rank_idx, (_, col_row) in enumerate(page_colleges.iterrows(), start=start + 1):
        cname = col_row['college_name']
        # Get branches for this college
        branch_rows = grouped[grouped['college_name'] == cname].sort_values('chance', ascending=False)
        branches = []
        for _, br in branch_rows.iterrows():
            branches.append({
                'name': br['branch'],
                'min': float(br['cut_min']),
                'avg': float(br['cut_avg']),
                'max': float(br['cut_max']),
                'chance': int(br['chance']),
                'level': br['chance_level'],
            })

        # Try to get college type from generated_details
        college_type = 'Engineering College'
        gen = _get_best_generated_match(cname)
        if gen is not None:
            ct = gen.get('college_type')
            if ct and str(ct).strip() and str(ct).strip().lower() != 'nan':
                college_type = str(ct).strip()

        results.append({
            'rank': rank_idx,
            'name': cname,
            'district': str(col_row['district']),
            'college_type': college_type,
            'best_chance': int(col_row['best_chance']),
            'chance_level': col_row['chance_level'],
            'fee_min': float(col_row['fee_min']) if not pd.isna(col_row['fee_min']) else None,
            'fee_max': float(col_row['fee_max']) if not pd.isna(col_row['fee_max']) else None,
            'branches': branches,
        })

    return {
        'colleges': results,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
    }


ALL_DISTRICTS = sorted(colleges_df['district'].dropna().unique().tolist()) if not colleges_df.empty else []
ALL_CATEGORIES = sorted(colleges_df['category'].dropna().unique().tolist()) if not colleges_df.empty else ['OC', 'BC', 'MBC', 'SC', 'ST']
ALL_BRANCHES = sorted(colleges_df['branch'].dropna().unique().tolist()) if not colleges_df.empty else []
ALL_COLLEGE_NAMES = sorted(colleges_df['college_name'].dropna().unique().tolist()) if not colleges_df.empty else []


@app.route('/predict', methods=['GET'], endpoint='predict')
def predict_page():
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to access prediction.', 'warning')
        return redirect(url_for('login'))

    return render_template('index.html',
                           all_districts=ALL_DISTRICTS,
                           all_categories=ALL_CATEGORIES,
                           all_colleges=ALL_COLLEGE_NAMES,
                           gemini_available=GEMINI_AVAILABLE)


@app.route('/compare', methods=['GET'])
def compare_page():
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to access comparison.', 'warning')
        return redirect(url_for('login'))
    return render_template('compare.html',
                           all_colleges=ALL_COLLEGE_NAMES)


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """JSON API for comparing up to 4 colleges side-by-side."""
    try:
        payload = request.get_json(silent=True) or {}
        college_names = payload.get('colleges', [])
        if not college_names or len(college_names) < 2:
            return jsonify({'ok': False, 'error': 'Select at least 2 colleges'}), 400
        if len(college_names) > 4:
            college_names = college_names[:4]

        results = []
        for cname in college_names:
            cname = str(cname).strip()
            if not cname:
                continue

            entry = {
                'name': cname,
                'location': None,
                'college_type': None,
                'affiliation': None,
                'placement_rate': None,
                'avg_package': None,
                'highest_package': None,
                'fee_avg': None,
                'fee_min': None,
                'fee_max': None,
                'hostel_fee': None,
                'overall_fee': None,
                'branches': [],
                'categories': [],
            }

            # 1. From generated_details_df
            gen = _get_best_generated_match(cname)
            if gen is not None:
                entry['location'] = normalize_spaces(gen.get('location')) or None
                entry['college_type'] = normalize_spaces(gen.get('college_type')) or None
                entry['affiliation'] = normalize_spaces(gen.get('affiliation')) or None
                entry['placement_rate'] = _num_or_none(gen.get('placement_rate_percent'))
                entry['avg_package'] = _num_or_none(gen.get('average_package_lpa'))
                entry['highest_package'] = _num_or_none(gen.get('highest_package_lpa'))
                entry['overall_fee'] = _num_or_none(gen.get('overall_fees'))
                entry['hostel_fee'] = _num_or_none(gen.get('hostel_fees'))

            # 2. From main colleges_df (cutoff dataset)
            if not colleges_df.empty:
                mask_exact = colleges_df['college_name'].str.lower() == cname.lower()
                matched = colleges_df[mask_exact]
                if matched.empty:
                    mask_contains = colleges_df['college_name'].str.lower().str.contains(
                        cname.lower().split(' - ')[0], na=False, regex=False
                    )
                    matched = colleges_df[mask_contains]

                if not matched.empty:
                    fees = matched['college_fees'].dropna()
                    if not fees.empty:
                        entry['fee_min'] = float(fees.min())
                        entry['fee_max'] = float(fees.max())
                        entry['fee_avg'] = round(float(fees.mean()))

                    if entry['location'] is None:
                        entry['location'] = str(matched['district'].iloc[0])

                    entry['branches'] = sorted(matched['branch'].dropna().unique().tolist())
                    entry['categories'] = sorted(matched['category'].dropna().unique().tolist())

            results.append(entry)

        return jsonify({'ok': True, 'colleges': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for CollegeDP-style college prediction."""
    try:
        payload = request.get_json(silent=True) or {}
        maths = float(payload.get('maths', 0))
        physics = float(payload.get('physics', 0))
        chemistry = float(payload.get('chemistry', 0))
        category = str(payload.get('category', 'OC')).strip()
        district = str(payload.get('district', '')).strip()
        sort_by = str(payload.get('sort_by', 'best_match')).strip()
        page = int(payload.get('page', 1))
        search_query = str(payload.get('search_query', '')).strip()

        # Validate marks
        for mark, name in [(maths, 'Maths'), (physics, 'Physics'), (chemistry, 'Chemistry')]:
            if mark < 0 or mark > 100:
                return jsonify({'ok': False, 'error': f'{name} marks must be between 0 and 100'}), 400

        # TNEA cutoff formula
        cutoff = maths + (physics / 2) + (chemistry / 2)

        result = predict_colleges(
            user_cutoff=cutoff,
            category=category,
            district=district if district and district.lower() not in ('any district', 'any', '') else None,
            sort_by=sort_by,
            page=page,
            per_page=25,
            max_fee=float(payload.get('max_fee', 0)) if payload.get('max_fee') else None,
            search_query=search_query
        )

        result['ok'] = True
        result['cutoff'] = round(cutoff, 2)
        result['category'] = category

        # Save prediction to DB
        try:
            user_id = session.get('user_id')
            if user_id:
                db = get_db()
                input_data = {'maths': maths, 'physics': physics, 'chemistry': chemistry,
                              'cutoff': round(cutoff, 2), 'category': category, 'district': district or 'Any'}
                top_colleges = result['colleges'][:5]
                top_names = [c['name'] for c in top_colleges]
                # Save top colleges with their fee and chance info
                top_json = json.dumps([{
                    'name': c['name'],
                    'district': c.get('district', ''),
                    'chance_level': c.get('chance_level', ''),
                    'fee_min': c.get('fee_min'),
                    'fee_max': c.get('fee_max'),
                } for c in top_colleges])
                db.execute(
                    "INSERT INTO predictions (user_id, input_json, prediction, top_predictions_json, fees_info_json, placements_info_json, created_at) VALUES (?,?,?,?,?,?,?)",
                    (int(user_id), json.dumps(input_data),
                     top_names[0] if top_names else 'No colleges found',
                     top_json, None, None, utc_now_iso())
                )
                db.commit()
        except Exception:
            traceback.print_exc()

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/college-details', methods=['POST'])
def api_college_details():
    """Fetch detailed info for a specific college."""
    try:
        payload = request.get_json(silent=True) or {}
        college_name = str(payload.get('name', '')).strip()
        if not college_name:
            return jsonify({'ok': False, 'error': 'College name required'}), 400

        details = {}

        # 1. Generated dataset details (placements, college type, etc.)
        gen = get_college_details_from_generated_dataset(college_name)
        if gen:
            details['college_info'] = gen.get('college_details') or {}
            details['placements'] = gen.get('placement_details') or {}
            details['fees_details'] = gen.get('fees_details') or {}
        else:
            details['college_info'] = {}
            details['placements'] = {}
            details['fees_details'] = {}

        # 2. Dataset stats (aggregated from CSV)
        if not colleges_df.empty:
            mask_exact = colleges_df['college_name'].str.lower() == college_name.lower()
            matched = colleges_df[mask_exact]
            if matched.empty:
                # try fuzzy contains
                mask_contains = colleges_df['college_name'].str.lower().str.contains(
                    college_name.lower().split(' - ')[0], na=False, regex=False
                )
                matched = colleges_df[mask_contains]

            if not matched.empty:
                details['district'] = str(matched['district'].iloc[0])
                fees = matched['college_fees'].dropna()
                if not fees.empty:
                    details['fee_min'] = float(fees.min())
                    details['fee_max'] = float(fees.max())
                    details['fee_avg'] = round(float(fees.mean()))
                    if not details['fees_details'].get('tuition_fee_annual_inr'):
                        details['fees_details']['tuition_fee_annual_inr'] = details['fee_avg']

                # Branches available
                branches = sorted(matched['branch'].unique().tolist())
                details['branches_available'] = branches

                # Categories available
                cats = sorted(matched['category'].unique().tolist())
                details['categories_available'] = cats

        # 3. Web-enriched details (Gemini)
        try:
            web = fetch_college_details_from_web(college_name)
            if web:
                merged = merge_detail_payloads(gen, web)
                if merged:
                    if merged.get('college_details'):
                        details['college_info'] = {**details.get('college_info', {}), **{k: v for k, v in merged['college_details'].items() if v}}
                    if merged.get('placement_details'):
                        details['placements'] = {**details.get('placements', {}), **{k: v for k, v in merged['placement_details'].items() if v}}
                    if merged.get('fees_details'):
                        details['fees_details'] = {**details.get('fees_details', {}), **{k: v for k, v in merged['fees_details'].items() if v}}
        except Exception:
            traceback.print_exc()

        details['ok'] = True
        details['name'] = college_name
        return jsonify(details)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


def get_chat_owner_key():
    user_id = session.get('user_id')
    if user_id:
        return f'user:{int(user_id)}'

    guest_key = session.get('guest_chat_key')
    if not guest_key:
        guest_key = uuid.uuid4().hex
        session['guest_chat_key'] = guest_key
    return f'guest:{guest_key}'

def _session_row_to_dict(row):
    return {
        'id': row['id'],
        'owner_key': row['owner_key'],
        'user_id': row['user_id'],
        'session_key': row['session_key'],
        'title': row['title'] or 'New Chat',
        'created_at': row['created_at'],
        'updated_at': row['updated_at'],
        'last_message_at': row['last_message_at'],
        'is_active': bool(row['is_active'])
    }

def _get_langchain_message_history(session_key):
    if not LANGCHAIN_AVAILABLE:
        return None
    try:
        return SQLChatMessageHistory(
            session_id=session_key,
            connection_string=DB_URI,
            table_name=LANGCHAIN_CHAT_TABLE
        )
    except Exception:
        return None

def list_chat_sessions(owner_key, limit=20):
    db = get_db()
    rows = db.execute(
        'SELECT * FROM chat_sessions WHERE owner_key = ? ORDER BY updated_at DESC LIMIT ?',
        (owner_key, int(limit))
    ).fetchall()
    return [_session_row_to_dict(r) for r in rows]

def get_chat_session(owner_key, session_key):
    db = get_db()
    row = db.execute(
        'SELECT * FROM chat_sessions WHERE owner_key = ? AND session_key = ? LIMIT 1',
        (owner_key, session_key)
    ).fetchone()
    return _session_row_to_dict(row) if row else None

def create_chat_session(owner_key, user_id=None, title='New Chat'):
    db = get_db()
    now = utc_now_iso()
    session_key = f'{owner_key}:{uuid.uuid4().hex[:12]}'

    db.execute('UPDATE chat_sessions SET is_active = 0 WHERE owner_key = ?', (owner_key,))
    db.execute(
        'INSERT INTO chat_sessions (owner_key, user_id, session_key, title, created_at, updated_at, last_message_at, is_active) VALUES (?,?,?,?,?,?,?,1)',
        (owner_key, int(user_id) if user_id else None, session_key, title, now, now, None)
    )
    db.commit()
    return get_chat_session(owner_key, session_key)

def get_or_create_active_chat_session(owner_key, user_id=None):
    db = get_db()
    row = db.execute(
        'SELECT * FROM chat_sessions WHERE owner_key = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 1',
        (owner_key,)
    ).fetchone()
    if row:
        return _session_row_to_dict(row)
    return create_chat_session(owner_key, user_id=user_id)

def set_active_chat_session(owner_key, session_key):
    db = get_db()
    db.execute('UPDATE chat_sessions SET is_active = 0 WHERE owner_key = ?', (owner_key,))
    db.execute(
        'UPDATE chat_sessions SET is_active = 1, updated_at = ? WHERE owner_key = ? AND session_key = ?',
        (utc_now_iso(), owner_key, session_key)
    )
    db.commit()

def update_chat_session_after_message(owner_key, session_key, title_hint=None):
    db = get_db()
    now = utc_now_iso()
    row = db.execute(
        'SELECT title FROM chat_sessions WHERE owner_key = ? AND session_key = ? LIMIT 1',
        (owner_key, session_key)
    ).fetchone()
    if not row:
        return

    new_title = None
    current_title = (row['title'] or '').strip()
    if title_hint and (not current_title or current_title.lower() == 'new chat'):
        new_title = normalize_spaces(title_hint)[:60]

    if new_title:
        db.execute(
            'UPDATE chat_sessions SET title = ?, updated_at = ?, last_message_at = ? WHERE owner_key = ? AND session_key = ?',
            (new_title, now, now, owner_key, session_key)
        )
    else:
        db.execute(
            'UPDATE chat_sessions SET updated_at = ?, last_message_at = ? WHERE owner_key = ? AND session_key = ?',
            (now, now, owner_key, session_key)
        )
    db.commit()

def load_chat_messages(owner_key, session_key, limit=150):
    if not get_chat_session(owner_key, session_key):
        return []

    history = _get_langchain_message_history(session_key)
    if history is not None:
        out = []
        for msg in history.messages[-int(limit):]:
            msg_type = str(getattr(msg, 'type', '')).lower()
            role = 'assistant'
            if msg_type == 'human':
                role = 'user'
            elif msg_type == 'ai':
                role = 'assistant'
            out.append({'role': role, 'content': str(getattr(msg, 'content', ''))})
        return out

    db = get_db()
    rows = db.execute(
        'SELECT role, content FROM chat_messages WHERE session_key = ? ORDER BY id ASC LIMIT ?',
        (session_key, int(limit))
    ).fetchall()
    return [{'role': r['role'], 'content': r['content']} for r in rows]

def persist_chat_turn(owner_key, session_key, user_message, ai_message):
    if not get_chat_session(owner_key, session_key):
        return

    history = _get_langchain_message_history(session_key)
    if history is not None:
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)
    else:
        db = get_db()
        now = utc_now_iso()
        db.execute(
            'INSERT INTO chat_messages (session_key, role, content, created_at) VALUES (?,?,?,?)',
            (session_key, 'user', user_message, now)
        )
        db.execute(
            'INSERT INTO chat_messages (session_key, role, content, created_at) VALUES (?,?,?,?)',
            (session_key, 'assistant', ai_message, now)
        )
        db.commit()

    update_chat_session_after_message(owner_key, session_key, title_hint=user_message)

def extract_rag_context(user_message):
    try:
        if generated_details_df.empty or not RAPIDFUZZ_AVAILABLE:
            return ""
        if len(user_message.strip()) < 5:
            return ""
        
        from rapidfuzz import utils as rf_utils
        names = generated_details_df['college_name'].dropna().tolist()
        
        # Use partial_ratio directly on sanitized strings
        best_match = rf_process.extractOne(
            user_message, 
            names, 
            processor=rf_utils.default_process,
            scorer=rf_fuzz.partial_ratio
        )
        
        # Lowered threshold to 75 to catch queries like "fees for [college_name]" mapping to "[college_name] - [Location]"
        if best_match and best_match[1] >= 75:
            matched_college = best_match[0]
            row = generated_details_df[generated_details_df['college_name'] == matched_college].iloc[0]
            
            ctx = f"Data for {matched_college}:\n"
            ctx += f"- Location: {row.get('location', 'N/A')}\n"
            ctx += f"- Type: {row.get('college_type', 'N/A')}\n"
            ctx += f"- Affiliation: {row.get('affiliation', 'N/A')}\n"
            ctx += f"- Placement Rate: {row.get('placement_rate_percent', 'N/A')}%\n"
            ctx += f"- Average Package: {row.get('average_package_lpa', 'N/A')} LPA\n"
            ctx += f"- Highest Package: {row.get('highest_package_lpa', 'N/A')} LPA\n"
            
            annual_fee = row.get('overall_fees', 50000)
            overall_fee = int(annual_fee) * 4 if pd.notna(annual_fee) else 'N/A'
            ctx += f"- Tuition (Annual): Rs. {annual_fee}\n"
            ctx += f"- Overall Program Fee (4 years): Rs. {overall_fee}\n"
            ctx += f"- Hostel Fee: Rs. {row.get('hostel_fees', 'N/A')}\n"
            return f"\n\n[SYSTEM RAG CONTEXT - Use this factual data to answer the user:]\n{ctx}\n"
    except Exception as e:
        print(f"RAG extraction error: {e}")
    return ""

def get_chatbot_response(user_message, chat_history=None):
    """Return chatbot response from Gemini 2.5 Flash Lite with RAG."""
    if not GEMINI_AVAILABLE:
        return None, 'Gemini API is not configured on the server.'

    msg = normalize_spaces(user_message)
    if not msg:
        return None, 'Please enter a valid question.'

    history_text = ''
    if chat_history:
        turns = []
        for item in chat_history[-8:]:
            role = 'Student' if item.get('role') == 'user' else 'Assistant'
            turns.append(f"{role}: {normalize_spaces(item.get('content', ''))}")
        history_text = '\n'.join(turns)

    rag_context = extract_rag_context(msg)

    prompt = (
        'You are a strict, highly specialized academic counseling assistant exclusively for Tamil Nadu engineering admissions (TNEA) and colleges. '
        'You MUST refuse to answer any questions that are not related to Tamil Nadu engineering colleges, admissions, cutoff scores, branches, placements, fees, hostels, or counseling choices. '
        'If a user asks about general programming (like Python code), general knowledge, or anything off-topic, politely decline and state your specific purpose. '
        'Answer clearly and concisely. Prefer practical guidance for students. '
        'If unsure about specific college data, say what is uncertain instead of guessing.\n\n'
        f'Conversation context:\n{history_text}\n'
        f'{rag_context}\n'
        f'Student question: {msg}'
    )

    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        text = normalize_spaces(response.text if response and hasattr(response, 'text') else '')
        if not text:
            return None, 'No response from Gemini. Please try again.'
        return text, None
    except Exception as e:
        return None, f'Chatbot error: {e}'


@app.route('/chatbot', methods=['GET'])
def chatbot_page():
    if not session.get('user_id'):
        flash('Please log in to access AI chatbot.', 'warning')
        return redirect(url_for('login'))

    owner_key = get_chat_owner_key()
    user_id = session.get('user_id')
    active = get_or_create_active_chat_session(owner_key, user_id=user_id)
    sessions = list_chat_sessions(owner_key, limit=30)
    messages = load_chat_messages(owner_key, active['session_key']) if active else []

    return render_template(
        'chatbot.html',
        gemini_available=GEMINI_AVAILABLE,
        chat_sessions=sessions,
        active_session_key=active['session_key'] if active else None,
        initial_messages=messages,
        langchain_enabled=LANGCHAIN_AVAILABLE
    )


@app.route('/chatbot/new', methods=['POST'])
def chatbot_new_session():
    if not session.get('user_id'):
        return jsonify({'ok': False, 'error': 'Authentication required.'}), 401

    try:
        owner_key = get_chat_owner_key()
        user_id = session.get('user_id')
        new_session = create_chat_session(owner_key, user_id=user_id)
        return jsonify({'ok': True, 'session': new_session})
    except Exception:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': 'Unable to create chat session.'}), 500


@app.route('/chatbot/history/<session_key>', methods=['GET'])
def chatbot_history(session_key):
    if not session.get('user_id'):
        return jsonify({'ok': False, 'error': 'Authentication required.'}), 401

    owner_key = get_chat_owner_key()
    chat_session = get_chat_session(owner_key, session_key)
    if not chat_session:
        return jsonify({'ok': False, 'error': 'Chat session not found.'}), 404

    set_active_chat_session(owner_key, session_key)
    messages = load_chat_messages(owner_key, session_key)
    return jsonify({'ok': True, 'session': chat_session, 'messages': messages})


@app.route('/chatbot/ask', methods=['POST'])
def chatbot_ask():
    if not session.get('user_id'):
        return jsonify({'ok': False, 'error': 'Authentication required.'}), 401

    try:
        payload = request.get_json(silent=True) or {}
        owner_key = get_chat_owner_key()

        message = payload.get('message', '')
        session_key = payload.get('session_key')

        if session_key:
            selected = get_chat_session(owner_key, session_key)
            if not selected:
                return jsonify({'ok': False, 'error': 'Invalid chat session.'}), 404
            set_active_chat_session(owner_key, session_key)
            active = selected
        else:
            active = get_or_create_active_chat_session(owner_key, user_id=session.get('user_id'))
            session_key = active['session_key']

        history = load_chat_messages(owner_key, session_key)
        reply, err = get_chatbot_response(message, chat_history=history)
        if err:
            return jsonify({'ok': False, 'error': err}), 400

        persist_chat_turn(owner_key, session_key, normalize_spaces(message), reply)
        updated_session = get_chat_session(owner_key, session_key)

        return jsonify({'ok': True, 'reply': reply, 'session': updated_session})
    except Exception:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': 'Unable to process request right now.'}), 500



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password required', 'danger')
            return render_template('register.html')
        db = get_db()
        try:
            pw_hash = generate_password_hash(password)
            db.execute('INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)',
                       (username, email, pw_hash, utc_now_iso()))
            db.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken', 'danger')
        except Exception:
            flash('Registration failed', 'danger')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')



@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))
    db = get_db()
    # show last 5 predictions
    rows = db.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5', (int(user_id),)).fetchall()
    recent = []
    for r in rows:
        try:
            input_data = json.loads(r['input_json']) if r['input_json'] else {}
        except Exception:
            input_data = {}
        recent.append({
            'id': r['id'],
            'input': input_data,
            'prediction': r['prediction'],
            'created_at': r['created_at']
        })
    return render_template('dashboard.html', recent=recent)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))


@app.route('/history')
def history():
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to view history', 'warning')
        return redirect(url_for('login'))
    db = get_db()
    rows = db.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC', (int(user_id),)).fetchall()
    history = []
    for r in rows:
        try:
            input_data = json.loads(r['input_json']) if r['input_json'] else {}
        except Exception:
            input_data = {}
        try:
            top_raw = r['top_predictions_json']
            top = json.loads(top_raw) if top_raw else []
        except Exception:
            top = []
        history.append({
            'id': r['id'],
            'input': input_data,
            'prediction': r['prediction'] or '',
            'top_colleges': top if isinstance(top, list) and top and isinstance(top[0], dict) else [],
            'top_names': top if isinstance(top, list) and top and isinstance(top[0], str) else [],
            'created_at': r['created_at']
        })
    return render_template('history.html', history=history)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


# --- ADMIN MODULE ---
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

def save_colleges_dataset():
    global colleges_df
    try:
        # Do not save internal computation columns
        save_df = colleges_df.drop(columns=['_name_lower', 'gap', 'abs_gap', 'district_match'], errors='ignore')
        save_df.to_csv(CSV_PATH, index=False)
        return True
    except Exception as e:
        print("Error saving CSV:", e)
        return False

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            flash('Admin logged in successfully', 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid admin credentials', 'danger')
            return render_template('admin_login.html')
            
    if session.get('is_admin'):
        return render_template('admin.html')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('Admin logged out', 'info')
    return redirect(url_for('home'))

@app.route('/api/admin/colleges', methods=['GET'])
def admin_get_colleges():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    
    if colleges_df.empty:
        return jsonify({'ok': True, 'colleges': []})
    
    # Safely convert to json-compliant dicts avoiding numpy type errors
    json_str = colleges_df.to_json(orient='records')
    data = json.loads(json_str)
    
    for idx, row in enumerate(data):
        row['_id'] = idx
    
    return jsonify({'ok': True, 'colleges': data})

@app.route('/api/admin/college/add', methods=['POST'])
def admin_add_college():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    
    global colleges_df
    try:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'No data provided'}), 400
        
        new_row = {
            'cut_off': float(data.get('cut_off') or 0),
            'previous_year_cutoff': float(data.get('previous_year_cutoff') or 0),
            'rank': int(data.get('rank') or 0),
            'branch': str(data.get('branch', '')),
            'category': str(data.get('category', '')),
            'district': str(data.get('district', '')),
            'sports_quota': str(data.get('sports_quota', 'No')),
            'college_name': str(data.get('college_name', '')),
            'college_fees': float(data.get('college_fees') or 0),
            '_name_lower': str(data.get('college_name', '')).lower()
        }
        
        new_df = pd.DataFrame([new_row])
        colleges_df = pd.concat([colleges_df, new_df], ignore_index=True)
        save_colleges_dataset()
        
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/admin/college/update', methods=['POST'])
def admin_update_college():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
        
    global colleges_df
    try:
        data = request.get_json()
        idx = data.get('_id')
        if idx is None or idx < 0 or idx >= len(colleges_df):
            return jsonify({'ok': False, 'error': 'Invalid ID'}), 400
            
        colleges_df.at[idx, 'cut_off'] = float(data.get('cut_off') or 0)
        colleges_df.at[idx, 'previous_year_cutoff'] = float(data.get('previous_year_cutoff') or 0)
        colleges_df.at[idx, 'rank'] = int(data.get('rank') or 0)
        colleges_df.at[idx, 'branch'] = str(data.get('branch', ''))
        colleges_df.at[idx, 'category'] = str(data.get('category', ''))
        colleges_df.at[idx, 'district'] = str(data.get('district', ''))
        colleges_df.at[idx, 'sports_quota'] = str(data.get('sports_quota', 'No'))
        colleges_df.at[idx, 'college_name'] = str(data.get('college_name', ''))
        colleges_df.at[idx, 'college_fees'] = float(data.get('college_fees') or 0)
        colleges_df.at[idx, '_name_lower'] = str(data.get('college_name', '')).lower()
        
        save_colleges_dataset()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/admin/college/delete', methods=['POST'])
def admin_delete_college():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
        
    global colleges_df
    try:
        data = request.get_json()
        idx = data.get('_id')
        if idx is None or idx < 0 or idx >= len(colleges_df):
            return jsonify({'ok': False, 'error': 'Invalid ID'}), 400
            
        colleges_df = colleges_df.drop(idx).reset_index(drop=True)
        save_colleges_dataset()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)