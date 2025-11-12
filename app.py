"""
AI Text-to-SQL Pro Assistant - Production-Ready Version
Fixed session management and dataset replacement
"""

import os
import io
import csv
import sqlite3
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
import uuid

# Optional: Gemini LLM import
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional: openpyxl for Excel export
try:
    import openpyxl
except Exception:
    openpyxl = None

# -------------------------
# ENHANCED PROFESSIONAL STYLING
# -------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Theme-agnostic color palette - works in light and dark mode */
    :root {
        --primary: #FF6B35;
        --primary-dark: #E85D2A;
        --primary-light: #FF8C61;
        --secondary: #4ECDC4;
        --success: #2ECC71;
        --warning: #F39C12;
        --danger: #E74C3C;
        --accent: #9B59B6;
    }

    .main {
        padding: 1.5rem 2.5rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FDC830 100%);
        padding: 3.5rem 2rem;
        border-radius: 24px;
        color: #FFFFFF;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(255, 107, 53, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }

    .hero-card h1 {
        margin: 0;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        color: #FFFFFF;
    }

    .hero-card p {
        margin: 1.2rem 0 0 0;
        opacity: 0.95;
        font-size: 1.2rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        position: relative;
        z-index: 1;
        color: #FFFFFF;
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: 2px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        color: #FFFFFF !important;
        font-size: 1rem;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #E85D2A 0%, #E07B1E 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 107, 53, 0.5);
        border-color: #FF6B35;
    }

    .stButton>button:active {
        transform: translateY(0);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 16px;
        border: 2px solid rgba(255, 107, 53, 0.2);
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        color: inherit;
        background-color: transparent;
        transition: all 0.2s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 107, 53, 0.1);
        border-color: rgba(255, 107, 53, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
        border-color: #FF6B35 !important;
    }

    .stAlert {
        border-radius: 12px;
        border: 2px solid;
        padding: 1rem 1.25rem;
        font-weight: 500;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
        border-right: 3px solid #FF6B35;
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        color: #FFFFFF !important;
        width: 100%;
        border-radius: 10px;
        border: 2px solid transparent;
    }

    [data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #E85D2A 0%, #E07B1E 100%) !important;
        box-shadow: 0 6px 16px rgba(255, 107, 53, 0.5);
        border-color: #FDC830;
    }

    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid rgba(255, 107, 53, 0.3);
        padding: 0.75rem;
        transition: all 0.2s ease;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #FF6B35;
        box-shadow: 0 0 0 4px rgba(255, 107, 53, 0.15);
    }

    h1, h2, h3, h4 {
        font-weight: 700;
        color: inherit;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.75rem;
        border-radius: 16px;
        border-left: 5px solid #FF6B35;
        border: 2px solid rgba(255, 107, 53, 0.3);
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.15);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(255, 107, 53, 0.25);
        border-color: #FF6B35;
    }
    
    .stat-card h3 {
        margin: 0;
        font-size: 2rem;
        color: #FF6B35;
    }
    
    .stat-card p, .stat-card h4 {
        margin: 0.5rem 0 0 0;
        color: inherit;
        font-weight: 500;
    }
    
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 2px solid rgba(255, 107, 53, 0.2);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: rgba(255, 107, 53, 0.5);
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid rgba(255, 107, 53, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #FF6B35;
    }
    
    [data-testid="stMetricLabel"] {
        color: inherit;
        font-weight: 600;
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%) !important;
        border: 2px solid transparent;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #27AE60 0%, #229954 100%) !important;
        border-color: #2ECC71;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 12px;
        border: 2px solid rgba(255, 107, 53, 0.3);
    }
    
    code {
        background-color: rgba(255, 107, 53, 0.1) !important;
        color: #FF6B35 !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        border: 1px solid rgba(255, 107, 53, 0.2);
    }
    
    /* Info/Success/Warning/Error boxes - theme agnostic */
    .stSuccess {
        background-color: rgba(46, 204, 113, 0.15) !important;
        border-left: 4px solid #2ECC71 !important;
        color: inherit !important;
    }
    
    .stInfo {
        background-color: rgba(78, 205, 196, 0.15) !important;
        border-left: 4px solid #4ECDC4 !important;
        color: inherit !important;
    }
    
    .stWarning {
        background-color: rgba(243, 156, 18, 0.15) !important;
        border-left: 4px solid #F39C12 !important;
        color: inherit !important;
    }
    
    .stError {
        background-color: rgba(231, 76, 60, 0.15) !important;
        border-left: 4px solid #E74C3C !important;
        color: inherit !important;
    }
    
    /* Make metric containers more visible */
    [data-testid="metric-container"] {
        background: rgba(255, 107, 53, 0.05);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid rgba(255, 107, 53, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# CONFIG
# -------------------------
USER_DB_FILE = "user_data.db"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_DB_PREFIX = "uploaded_"

# -------------------------
# UTILITY: Clear Session Data
# -------------------------
def clear_session_data():
    """Clear all session data and caches when switching datasets"""
    import gc
    import time
    
    # Clear all uploaded database files
    for db_key, db_meta in list(st.session_state.get("databases", {}).items()):
        db_path = db_meta.get("path")
        
        # Close database connections
        if db_path and os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("VACUUM;")
            except:
                pass
        
        # Dispose SQLAlchemy engines
        if db_meta.get("engine"):
            try:
                db_meta["engine"].dispose()
            except:
                pass
    
    # Clear Streamlit cache
    st.cache_data.clear()
    
    # Force garbage collection
    gc.collect()
    time.sleep(0.2)
    
    # Reset session state
    st.session_state["databases"] = {}
    st.session_state["chat_history"] = []
    st.session_state["last_result"] = {"columns": [], "records": []}
    st.session_state["selected_db"] = None

# -------------------------
# AUTH FUNCTIONS
# -------------------------
def init_user_table():
    """Initialize user authentication database"""
    if not os.path.exists(USER_DB_FILE):
        conn = sqlite3.connect(USER_DB_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT
            );
        """)
        conn.commit()
        conn.close()

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username, password, display_name=None):
    """Register a new user"""
    if not username or not password:
        return False, "Username and password required."
    conn = sqlite3.connect(USER_DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, display_name) VALUES (?, ?, ?)",
                  (username, hash_password(password), display_name or username))
        conn.commit()
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    """Authenticate user login"""
    conn = sqlite3.connect(USER_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash, display_name FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row or hash_password(password) != row[0]:
        return False, "Invalid username or password."
    return True, row[1] or username

init_user_table()

# -------------------------
# GEMINI CONFIG
# -------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if genai is not None and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        genai = None

def generate_sql_with_gemini(question, schema, chat_history=None, db_type="sqlite"):
    """Generate SQL query using Gemini AI"""
    if genai is None:
        raise RuntimeError("LLM not configured. Please set GOOGLE_API_KEY environment variable.")
    if chat_history is None:
        chat_history = []

    context = "\n".join([
        f"Q: {h.get('question')} → SQL: {h.get('sql')}" 
        for h in chat_history[-3:]
        if h.get("sql")
    ])

    # Database-specific guidance
    db_guidelines = {
        "sqlite": (
            "Use only SQLite-compatible SQL. "
            "Avoid unsupported functions like GREATEST(), LEAST(), DATE_FORMAT(), NOW(). "
            "Use CASE WHEN statements and strftime() for dates."
        ),
        "mysql": (
            "Use MySQL-compatible syntax. Functions like GREATEST(), DATE_FORMAT(), and CONCAT() are supported."
        ),
        "postgresql": (
            "Use PostgreSQL syntax. Functions like GREATEST(), STRING_AGG(), and DATE_TRUNC() are supported."
        ),
        "mssql": (
            "Use SQL Server syntax. Use TOP instead of LIMIT. Use CONCAT or + for string concatenation."
        )
    }

    db_hint = db_guidelines.get(db_type.lower(), "Generate standard SQL syntax.")

    prompt = f"""You are an expert SQL query generator.

Database type: {db_type.upper()}

Schema:
{schema}

Guidelines:
{db_hint}

Previous queries context:
{context}

User question:
{question}

Generate ONLY a valid SELECT query. Do not include:
- Explanations or comments
- Markdown formatting
- Multiple queries
- Any text before or after the query

Return the SQL query directly."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        sql = (resp.text or "").strip()
        
        # Clean up response
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # Extract SELECT query
        if "select" in sql.lower():
            sql = sql[sql.lower().find("select"):]
        sql = sql.split(";")[0].strip()

        if not sql.lower().startswith("select"):
            raise ValueError("LLM did not produce a valid SELECT query.")

        return sql
    except Exception as e:
        raise RuntimeError(f"Failed to generate SQL: {str(e)}")


def generate_ai_explanation(question, df):
    """Generate AI-powered explanation of query results"""
    if genai is None:
        return None
    try:
        prompt = f"""Analyze this SQL query result and provide concise, actionable insights.

User Question: {question}

Data Preview:
{df.head(5).to_string(index=False)}

Dataset Info:
- Total Rows: {len(df)}
- Columns: {', '.join(df.columns.tolist())}

Provide:
1. Key findings (2-3 bullet points)
2. Notable patterns or trends
3. One actionable recommendation

Keep response under 150 words."""

        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

# -------------------------
# DB UTILS
# -------------------------
def _safe_filename(name: str) -> str:
    """Sanitize filename for database storage"""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)

def safe_read_excel(uploaded_file):
    """Safely read Excel file with multiple sheets"""
    excel = pd.ExcelFile(uploaded_file)
    result = {}
    for sheet in excel.sheet_names:
        df = excel.parse(sheet)
        result[sheet.strip().replace(" ", "_")] = df
    return result

def load_database_to_sqlite(uploaded_file, auto_normalize=False):
    """Load CSV/Excel/SQLite into SQLite DB"""
    import re

    orig_name = uploaded_file.name
    base = _safe_filename(orig_name.rsplit(".", 1)[0])
    target_db = f"{UPLOAD_DB_PREFIX}{base}.db"

    uploaded_file.seek(0, io.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    if size > MAX_UPLOAD_SIZE:
        raise ValueError("File too large (>50MB limit).")

    lower = orig_name.lower()
    conn = sqlite3.connect(target_db)

    try:
        # --- SQLite file ---
        if lower.endswith((".db", ".sqlite", ".sqlite3")):
            conn.close()
            with open(target_db, "wb") as f:
                f.write(uploaded_file.getbuffer())
            conn = sqlite3.connect(target_db)

        # --- CSV file ---
        elif lower.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if df.empty:
                raise ValueError("CSV file contains no data.")
            df.columns = [str(c).strip().replace(" ", "_").replace(".", "_") for c in df.columns]
            df.to_sql(base, conn, if_exists="replace", index=False)

        # --- Excel file ---
        elif lower.endswith((".xlsx", ".xls")):
            excel = pd.ExcelFile(uploaded_file)
            created_tables = []

            for sheet_name in excel.sheet_names:
                try:
                    df = None
                    for header_row in range(5):
                        temp = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
                        if not temp.empty and temp.shape[1] > 0:
                            if temp.columns.notna().sum() > 0:
                                df = temp
                                break

                    if df is None or df.empty:
                        continue

                    df.columns = [
                        f"Column_{i}" if str(c).startswith("Unnamed") or str(c).strip() == ""
                        else str(c).strip().replace(" ", "_").replace(".", "_")
                        for i, c in enumerate(df.columns)
                    ]

                    df.dropna(how="all", inplace=True)

                    if df.empty:
                        continue

                    clean_name = re.sub(r"[^A-Za-z0-9_]", "_", sheet_name.strip())
                    clean_name = re.sub(r"_+", "_", clean_name).strip("_") or base

                    counter = 1
                    original_name = clean_name
                    while clean_name in created_tables:
                        clean_name = f"{original_name}_{counter}"
                        counter += 1

                    df.to_sql(clean_name, conn, if_exists="replace", index=False)
                    created_tables.append(clean_name)
                    st.info(f"✅ Loaded sheet '{sheet_name}' → table '{clean_name}' ({len(df)} rows, {len(df.columns)} cols)")
                except Exception as e:
                    st.warning(f"⚠️ Skipped sheet '{sheet_name}': {str(e)}")

            if not created_tables:
                raise ValueError("Excel file contained no valid sheets with data.")

        else:
            raise ValueError(f"Unsupported file type: {orig_name}")

        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]

        conn.close()
        return target_db, sorted(set(tables))

    except Exception as e:
        conn.close()
        if os.path.exists(target_db):
            os.remove(target_db)
        raise e


def get_schema_text_from_path(db_path):
    """Extract schema info from SQLite DB"""
    import re
    try:
        with sqlite3.connect(db_path) as conn:
            all_tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                ).fetchall()
            ]

            clean_names = {}
            for t in all_tables:
                short = re.sub(r".*__\d+__", "", t)
                short = re.sub(r"^uploaded_|[_\W]+$", "", short)
                short = re.sub(r"^DataScience_Lab_Datasets_*", "", short)
                short = short.strip("_")
                if short and short not in clean_names.values():
                    clean_names[t] = short

            lines = []
            for full_name, clean_name in clean_names.items():
                try:
                    cols = [
                        f"{c[1]} ({c[2]})"
                        for c in conn.execute(f"PRAGMA table_info('{full_name}');").fetchall()
                    ]
                    if cols:
                        lines.append(f"Table: {clean_name}\nColumns: {', '.join(cols)}\n")
                except sqlite3.OperationalError:
                    continue

        return "\n".join(lines)

    except Exception as e:
        return f"Error reading schema: {str(e)}"


def get_schema_text_from_engine(engine):
    """Extract schema information from SQLAlchemy engine"""
    try:
        inspector = inspect(engine)
        lines = []
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            col_info = [f"{c['name']} ({c['type']})" for c in columns]
            lines.append(f"Table: {table_name}\nColumns: {', '.join(col_info)}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading schema: {str(e)}"


# -------------------------
# SAFE SQL VALIDATION
# -------------------------
DANGEROUS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "EXEC"]

def is_safe_sql(sql):
    """Validate SQL query safety - only SELECT queries allowed"""
    if not sql:
        return False
    up = sql.upper().strip()
    for kw in DANGEROUS:
        if kw in up:
            return False
    return up.startswith("SELECT")

# -------------------------
# TABLE LOADERS WITH CACHING
# -------------------------
@st.cache_data
def load_table_sample_path(db_path, table_name, n=100):
    """Load sample data from SQLite table"""
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM '{table_name}' LIMIT {n}", conn)
    except Exception as e:
        st.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_table_sample_engine(_engine, table_name, n=100):
    """Load sample data from database engine"""
    try:
        return pd.read_sql_table(table_name, _engine).head(n)
    except Exception as e:
        st.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_full_table_path(db_path, table_name, n=None):
    """Load full table data from SQLite"""
    try:
        with sqlite3.connect(db_path) as conn:
            sql = f"SELECT * FROM '{table_name}'" + (f" LIMIT {n}" if n else "")
            return pd.read_sql_query(sql, conn)
    except Exception as e:
        st.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_full_table_engine(_engine, table_name, n=None):
    """Load full table data from database engine"""
    try:
        df = pd.read_sql_table(table_name, _engine)
        return df.head(n) if n else df
    except Exception as e:
        st.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(
    page_title="AI Text-to-SQL Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# SESSION STATE INITIALIZATION
default_states = {
    "logged_in": False,
    "username": None,
    "display_name": None,
    "chat_history": [],
    "last_result": {"columns": [], "records": []},
    "databases": {},
    "selected_db": None
}

for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# -------------------------
# SIDEBAR AUTHENTICATION
# -------------------------
with st.sidebar:
    st.markdown("### 🔐 User Account")
    
    if not st.session_state.logged_in:
        auth_mode = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        username_input = st.text_input("👤 Username", key="auth_user", placeholder="Enter username")
        password_input = st.text_input("🔑 Password", type="password", key="auth_pw", placeholder="Enter password")
        
        if auth_mode == "Register":
            display_name_input = st.text_input("✨ Display Name (optional)", key="auth_display", placeholder="Your name")
            if st.button("📝 Create Account", use_container_width=True, key="register_btn"):
                ok, msg = register_user(username_input.strip(), password_input, display_name_input)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            if st.button("🚀 Sign In", use_container_width=True, key="login_btn", type="primary"):
                ok, msg = login_user(username_input.strip(), password_input)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username_input.strip()
                    st.session_state.display_name = msg
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <h3>👋</h3>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">{st.session_state.display_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Your Statistics")
        
        num_dbs = len(st.session_state.databases)
        num_queries = len(st.session_state.chat_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📦 Databases", num_dbs)
        with col2:
            st.metric("💬 Queries", num_queries)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            for key, default_value in default_states.items():
                st.session_state[key] = default_value
            st.rerun()

# -------------------------
# MAIN APPLICATION (AFTER LOGIN)
# -------------------------
if st.session_state.logged_in:
    st.markdown("""
    <div class="hero-card">
        <h1>🤖 AI Text-to-SQL Pro</h1>
        <p>Transform natural language into powerful SQL queries with intelligent AI assistance</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Database Connection Section ---
    with st.expander("📥 Data Source Management", expanded=True):
        st.markdown("#### 📂 Upload Files")
        
        col_upload, col_settings = st.columns([2, 1])
        
        with col_upload:
            uploaded_files = st.file_uploader(
                "Select files to upload",
                type=["db", "sqlite", "sqlite3", "csv", "xlsx", "xls"],
                accept_multiple_files=True,
                help="Supported: SQLite (.db, .sqlite), CSV, Excel (.xlsx, .xls)",
                label_visibility="collapsed"
            )
        
        with col_settings:
            auto_normalize = st.checkbox(
                "🔄 Auto-Normalize",
                value=False,
                help="Automatically normalize tables to 3NF (Coming Soon)"
            )
            st.info("**💡 Quick Info**\n\n✓ Max: 50MB\n✓ Multi-file support\n✓ Auto-encoding detection")
        
        # -------------------------
        # Handle uploaded files - CLEAR OLD DATA FIRST
        # -------------------------
        if uploaded_files:
            # 🧹 Step 1: Clear ALL previous session data
            clear_session_data()
            
            # 🆕 Step 2: Process newly uploaded files
            for f in uploaded_files:
                safe_key = os.path.splitext(f.name)[0]
                try:
                    with st.spinner(f"⏳ Processing {f.name}..."):
                        db_path, tables = load_database_to_sqlite(f, auto_normalize)
                    st.session_state["databases"][safe_key] = {
                        "type": "sqlite",
                        "path": db_path,
                        "tables": list(set(tables)),
                        "engine": None
                    }
                    st.success(f"✅ {f.name} loaded successfully ({len(set(tables))} valid tables)")
                except Exception as e:
                    st.error(f"❌ Failed to load {f.name}: {str(e)}")

        # -------------------------
        # Manage loaded databases
        # -------------------------
        if st.session_state["databases"]:
            st.markdown("### 🗑️ Manage Loaded Databases")

            import gc, time

            for db_key, db_meta in list(st.session_state["databases"].items()):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**📦 {db_key}** — {len(db_meta['tables'])} table(s)")
                with col2:
                    st.caption(f"({db_meta['type'].upper()})")
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_{db_key}", use_container_width=True):
                        try:
                            db_path = db_meta.get("path")

                            # Step 1: Clear caches and garbage collect
                            st.cache_data.clear()
                            gc.collect()
                            time.sleep(0.3)

                            # Step 2: Close SQLite connections
                            if db_path and os.path.exists(db_path):
                                try:
                                    with sqlite3.connect(db_path) as conn:
                                        conn.execute("VACUUM;")
                                    conn = None
                                except Exception:
                                    pass

                            # Step 3: Dispose SQLAlchemy engine
                            if db_meta.get("engine"):
                                try:
                                    db_meta["engine"].dispose()
                                except Exception:
                                    pass

                            # Step 4: Delete database file with retries
                            if db_path and os.path.exists(db_path):
                                for attempt in range(3):
                                    try:
                                        os.remove(db_path)
                                        break
                                    except PermissionError:
                                        time.sleep(0.5)
                                        continue
                                    except Exception:
                                        break

                            # Step 5: Clear this database from session
                            if db_key in st.session_state["databases"]:
                                del st.session_state["databases"][db_key]
                            
                            # Clear related data
                            st.session_state["chat_history"] = []
                            st.session_state["last_result"] = {"columns": [], "records": []}
                            st.session_state["selected_db"] = None

                            st.success(f"✅ {db_key} removed successfully.")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Failed to remove {db_key}: {str(e)}")

        # Database Server Connection
        st.markdown("---")
        st.markdown("#### 🔌 Connect to Database Server")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            db_type = st.selectbox("Database Type", ["Select", "MSSQL", "MySQL", "PostgreSQL"])
        
        with col2:
            if db_type != "Select":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    host = st.text_input("Host", placeholder="localhost")
                with col_b:
                    port = st.text_input("Port", placeholder="3306")
                with col_c:
                    db_name = st.text_input("Database Name")
                
                col_d, col_e, col_f = st.columns([1, 1, 1])
                with col_d:
                    user = st.text_input("Username")
                with col_e:
                    password = st.text_input("Password", type="password")
                with col_f:
                    st.write("")
                    st.write("")
                    if st.button("🔗 Connect", use_container_width=True):
                        with st.spinner("🔄 Connecting to database..."):
                            try:
                                if db_type == "MSSQL":
                                    conn_str = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server"
                                elif db_type == "MySQL":
                                    conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
                                elif db_type == "PostgreSQL":
                                    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
                                
                                engine = create_engine(conn_str)
                                tables = inspect(engine).get_table_names()
                                
                                st.session_state["databases"][f"{db_type}_{db_name}"] = {
                                    "type": db_type.lower(),
                                    "engine": engine,
                                    "tables": tables,
                                    "path": None
                                }
                                st.success(f"✅ Connected to {db_type} - {db_name} ({len(tables)} tables)")
                            except SQLAlchemyError as e:
                                st.error(f"❌ Connection failed: {str(e)}")

    # --- Select Active Database ---
    st.markdown("### 🎯 Active Database Selection")
    
    if st.session_state["databases"]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            db_choice = st.selectbox(
                "Choose your database",
                list(st.session_state["databases"].keys()),
                help="Select which database to query",
                label_visibility="collapsed"
            )
        
        selected_db_meta = st.session_state["databases"].get(db_choice)
        selected_db_type = selected_db_meta["type"] if selected_db_meta else None
        selected_db_engine = selected_db_meta.get("engine") if selected_db_meta else None
        selected_db_path = selected_db_meta.get("path") if selected_db_meta else None
        tables = selected_db_meta["tables"] if selected_db_meta else []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 Tables", len(tables))
        with col2:
            st.metric("🔧 Type", selected_db_type.upper() if selected_db_type else "N/A")
        with col3:
            total_queries = sum(1 for h in st.session_state.chat_history if h.get("ok"))
            st.metric("✅ Successful Queries", total_queries)
    else:
        st.info("👆 Please upload a file or connect to a database to get started")
        st.stop()

    st.markdown("---")

    # -------------------------
    # MAIN TABS
    # -------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Query Assistant",
        "📘 Schema Explorer",
        "📊 Data Insights",
        "🕘 Query History",
        "🎨 Visualizations",
        "🗺️ Dashboard"
    ])

    # ====================================
    # TAB 1: QUERY ASSISTANT
    # ====================================
    with tab1:
        st.markdown("### 💬 Ask Your Question")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., Show me top 10 customers by total revenue in 2024",
                label_visibility="collapsed",
                key="question_input"
            )
        with col2:
            use_ai = st.checkbox("✨ Smart Mode", value=True, help="Use AI to generate SQL automatically")
        
        manual_sql = st.text_area(
            "Or write SQL manually",
            height=120,
            placeholder="SELECT * FROM table_name LIMIT 10;",
            key="manual_sql_input"
        )

        if st.button("▶️ Execute Query", type="primary", use_container_width=True):
            chat_entry = {
                "question": question.strip(),
                "sql": None,
                "rows": None,
                "explanation": None,
                "ok": False
            }

            # Input validation
            import re
            
            def is_invalid_prompt(q):
                q_clean = q.strip().lower()
                if len(q_clean.split()) < 3:
                    return True
                if re.fullmatch(r"[^\w\s]{1,}", q_clean):
                    return True
                if len(q_clean) > 5 and len(set(q_clean.replace(" ", ""))) <= 2:
                    return True
                meaningless = ["hello", "test", "asdf", "qwer", "zxcv", "random", "abc", "xyz"]
                if any(word in q_clean for word in meaningless):
                    return True
                return False

            if not manual_sql.strip() and is_invalid_prompt(question):
                st.error("❌ Please enter a meaningful question with at least 3 words describing what data you want to see.")
                chat_entry["ok"] = False
                st.session_state.chat_history.append(chat_entry)
                st.stop()

            # Get schema context
            schema_context = (
                get_schema_text_from_path(selected_db_path) if selected_db_path
                else get_schema_text_from_engine(selected_db_engine) if selected_db_engine
                else "No schema available"
            )

            # Determine SQL to execute
            sql_to_run = None

            if manual_sql.strip():
                sql_to_run = manual_sql.strip()
            elif use_ai and genai:
                with st.spinner("✨ AI is generating your SQL query..."):
                    try:
                        sql_to_run = generate_sql_with_gemini(
                            question.strip(),
                            schema_context,
                            st.session_state.chat_history,
                            db_type=selected_db_type
                        )
                    except Exception as e:
                        st.warning(f"⚠️ AI generation failed: {str(e)}")
                        st.info("💡 Try writing SQL manually or check your GOOGLE_API_KEY")
            elif tables:
                # Default fallback query
                sql_to_run = f"SELECT * FROM '{tables[0]}' LIMIT 10"
                st.info(f"ℹ️ Using default query for table: {tables[0]}")

            if not sql_to_run:
                st.error("❌ No SQL to execute. Please enable Smart Mode or write manual SQL.")
                chat_entry["ok"] = False
                st.session_state.chat_history.append(chat_entry)
                st.stop()

            # Validate SQL safety
            if not is_safe_sql(sql_to_run):
                st.error("❌ Only SELECT queries are allowed for security reasons.")
                chat_entry["ok"] = False
                st.session_state.chat_history.append(chat_entry)
                st.stop()

            chat_entry["sql"] = sql_to_run
            
            st.markdown("#### 📝 Generated SQL Query")
            st.code(sql_to_run, language="sql")

            # Execute query
            try:
                with st.spinner("⚡ Executing query..."):
                    if selected_db_type == "sqlite" and selected_db_path:
                        with sqlite3.connect(selected_db_path) as conn:
                            df = pd.read_sql_query(sql_to_run, conn)
                    elif selected_db_engine:
                        df = pd.read_sql_query(sql_to_run, selected_db_engine)
                    else:
                        df = pd.DataFrame()

                if df.empty:
                    st.warning("⚠️ Query returned no results. Try modifying your question.")
                    chat_entry["ok"] = False
                else:
                    chat_entry["rows"] = len(df)
                    chat_entry["ok"] = True
                    
                    st.markdown("#### 📊 Query Results")
                    st.success(f"✅ Query successful! Retrieved {len(df)} rows")
                    st.dataframe(df, use_container_width=True, height=400)

                    # AI Explanation
                    if genai and use_ai and question.strip():
                        st.markdown("#### 💡 AI Insights & Analysis")
                        with st.spinner("✨ AI is analyzing your results..."):
                            explanation = generate_ai_explanation(question.strip(), df)
                            if explanation:
                                chat_entry["explanation"] = explanation
                                st.info(explanation)

                    # Save results for later use
                    st.session_state.last_result = {
                        "columns": df.columns.tolist(),
                        "records": df.to_dict(orient="records")
                    }
                    
                    # Download option
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download as CSV",
                            csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        if openpyxl:
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Results')
                            excel_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Download as Excel",
                                excel_buffer,
                                file_name="query_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        else:
                            st.info("📦 Install openpyxl for Excel export")

            except Exception as e:
                st.error(f"❌ Query execution failed: {str(e)}")
                st.info("💡 Tips: Check table names, column names, and SQL syntax")
                chat_entry["ok"] = False

            st.session_state.chat_history.append(chat_entry)

    # ====================================
    # TAB 2: SCHEMA EXPLORER
    # ====================================
    # ====================================
    # TAB 2: SCHEMA EXPLORER
    # ====================================
    # ====================================
    # TAB 2: SCHEMA EXPLORER
    # ====================================
    # ====================================
    # TAB 2: SCHEMA EXPLORER
    # ====================================
    with tab2:
        st.markdown("### 📘 Database Schema Overview")
        
        if selected_db_type == "sqlite" and selected_db_path:
            schema_text = get_schema_text_from_path(selected_db_path)
        elif selected_db_engine:
            schema_text = get_schema_text_from_engine(selected_db_engine)
        else:
            st.info("ℹ️ Select a database to view schema information")
            schema_text = ""
        
        if schema_text:
            st.markdown("#### 📋 Complete Schema Summary")
            st.code(schema_text, language="sql")
            
            st.markdown("---")
            st.markdown("### 📊 Detailed Table Information")
            
            for t in tables:
                with st.expander(f"🗂️ Table: **{t}**", expanded=False):
                    try:
                        if selected_db_type == "sqlite" and selected_db_path:
                            with sqlite3.connect(selected_db_path) as conn:
                                # Get column info
                                cols_info = conn.execute(f"PRAGMA table_info('{t}');").fetchall()
                                
                                if cols_info:
                                    schema_data = []
                                    
                                    for col in cols_info:
                                        try:
                                            if len(col) >= 6:
                                                col_name = col[1]
                                                col_type = col[2]
                                                not_null = col[3]
                                                default_val = col[4]
                                            else:
                                                col_name = str(col[1]) if len(col) > 1 else "unknown"
                                                col_type = str(col[2]) if len(col) > 2 else "TEXT"
                                                not_null = col[3] if len(col) > 3 else 0
                                                default_val = col[4] if len(col) > 4 else None
                                            
                                            schema_data.append({
                                                "Column Name": col_name,
                                                "Data Type": col_type if col_type else "TEXT",
                                                "Not Null": "Yes" if (not_null == 1 or not_null is True) else "No",
                                                "Default Value": str(default_val) if default_val else ""
                                            })
                                        except Exception as col_err:
                                            st.error(f"Error parsing column: {col_err}")
                                    
                                    st.markdown("**📊 Column Schema**")
                                    df_schema = pd.DataFrame(schema_data)
                                    st.dataframe(df_schema, use_container_width=True, hide_index=True)
                                    
                                    # Show metrics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("📝 Total Columns", len(schema_data))
                                    with col2:
                                        try:
                                            row_count = conn.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
                                            st.metric("📊 Total Rows", f"{row_count:,}")
                                        except:
                                            st.metric("📊 Total Rows", "N/A")
                                    
                                    # Show data type distribution
                                    st.markdown("---")
                                    st.markdown("**🔢 Data Type Distribution**")
                                    
                                    type_counts = {}
                                    for item in schema_data:
                                        dtype = item["Data Type"]
                                        type_counts[dtype] = type_counts.get(dtype, 0) + 1
                                    
                                    for dtype, count in sorted(type_counts.items()):
                                        st.caption(f"• **{dtype}**: {count} column(s)")
                                    
                                    # Sample data preview
                                    st.markdown("---")
                                    st.markdown("**📄 Sample Data (First 5 Rows)**")
                                    try:
                                        sample_df = pd.read_sql_query(f"SELECT * FROM '{t}' LIMIT 5", conn)
                                        st.dataframe(sample_df, use_container_width=True, hide_index=True)
                                    except Exception as e:
                                        st.error(f"Could not load sample data: {e}")
                                else:
                                    st.warning("⚠️ Could not retrieve schema information")
                        
                        elif selected_db_engine:
                            inspector = inspect(selected_db_engine)
                            
                            # Get columns
                            columns = inspector.get_columns(t)
                            
                            # Build schema dataframe
                            schema_data = []
                            for col in columns:
                                schema_data.append({
                                    "Column Name": col['name'],
                                    "Data Type": str(col['type']),
                                    "Nullable": "Yes" if col.get('nullable', True) else "No",
                                    "Default Value": str(col.get('default', '')) if col.get('default') else ""
                                })
                            
                            st.markdown("**📊 Column Schema**")
                            df_schema = pd.DataFrame(schema_data)
                            st.dataframe(df_schema, use_container_width=True, hide_index=True)
                            
                            # Show metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("📝 Total Columns", len(schema_data))
                            with col2:
                                try:
                                    count_query = f"SELECT COUNT(*) as cnt FROM {t}"
                                    row_count = pd.read_sql_query(count_query, selected_db_engine).iloc[0]['cnt']
                                    st.metric("📊 Total Rows", f"{row_count:,}")
                                except:
                                    st.metric("📊 Total Rows", "N/A")
                            
                            # Show data type distribution
                            st.markdown("---")
                            st.markdown("**🔢 Data Type Distribution**")
                            
                            type_counts = {}
                            for item in schema_data:
                                dtype = item["Data Type"]
                                type_counts[dtype] = type_counts.get(dtype, 0) + 1
                            
                            for dtype, count in sorted(type_counts.items()):
                                st.caption(f"• **{dtype}**: {count} column(s)")
                            
                            # Sample data preview
                            st.markdown("---")
                            st.markdown("**📄 Sample Data (First 5 Rows)**")
                            try:
                                sample_df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 5", selected_db_engine)
                                st.dataframe(sample_df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Could not load sample data: {e}")
                    
                    except Exception as e:
                        st.error(f"❌ Error loading table info: {str(e)}")
                        import traceback
                        with st.expander("Show Error Details"):
                            st.code(traceback.format_exc())
    # ====================================
    # TAB 3: DATA INSIGHTS
    # ====================================
    with tab3:
        st.markdown("### 📊 Data Quality & Statistical Analysis")
        
        if tables:
            for idx, t in enumerate(tables):
                with st.expander(f"📊 Analysis: **{t}**", expanded=(len(tables) == 1)):
                    if selected_db_type == "sqlite" and selected_db_path:
                        df_analysis = load_table_sample_path(selected_db_path, t, n=500)
                    else:
                        df_analysis = load_table_sample_engine(selected_db_engine, t, n=500)
                    
                    if df_analysis.empty:
                        st.info("ℹ️ No data available in this table")
                        continue
                    
                    st.markdown("#### 📄 Sample Data Preview")
                    st.dataframe(df_analysis.head(20), use_container_width=True)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Summary Statistics")
                        stats = df_analysis.describe(include="all").transpose()
                        st.dataframe(stats, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### ⚠️ Data Quality Check")
                        
                        # Missing values
                        missing = df_analysis.isna().sum()
                        missing_pct = (missing / len(df_analysis) * 100).round(2)
                        quality_df = pd.DataFrame({
                            "Column": missing.index,
                            "Missing": missing.values,
                            "Missing %": missing_pct.values
                        })
                        quality_df = quality_df[quality_df["Missing"] > 0]
                        
                        if quality_df.empty:
                            st.success("✅ Perfect! No missing values found")
                        else:
                            st.dataframe(quality_df, use_container_width=True, hide_index=True)
                            st.warning(f"⚠️ Found missing values in {len(quality_df)} column(s)")
                        
                        # Data types
                        st.markdown("**📋 Data Types**")
                        dtype_counts = df_analysis.dtypes.value_counts()
                        for dtype, count in dtype_counts.items():
                            st.caption(f"• {dtype}: {count} columns")
                    
                    # Numeric columns analysis
                    numeric_cols = df_analysis.select_dtypes(include='number').columns.tolist()
                    if numeric_cols:
                        st.markdown("---")
                        st.markdown("#### 📊 Numeric Columns Distribution")
                        
                        for col_idx, col in enumerate(numeric_cols[:3]):
                            fig = px.histogram(
                                df_analysis,
                                x=col,
                                title=f"Distribution of {col}",
                                color_discrete_sequence=['#3b82f6']
                            )
                            fig.update_layout(height=300, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True, key=f"hist_tab3_{idx}_{col_idx}_{col}_{uuid.uuid4().hex[:8]}")
        else:
            st.info("📊 No tables available for analysis")

    # ====================================
    # TAB 4: QUERY HISTORY
    # ====================================
    with tab4:
        st.markdown("### 🕘 Query Execution History")
        
        history = st.session_state.get("chat_history", [])
        
        if not history:
            st.info("📝 No queries executed yet. Start by asking questions in the Query Assistant tab!")
        else:
            # Summary metrics
            total = len(history)
            successful = sum(1 for h in history if h.get("ok"))
            failed = total - successful
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Queries", total)
            with col2:
                st.metric("✅ Successful", successful)
            with col3:
                st.metric("❌ Failed", failed)
            with col4:
                success_rate = (successful / total * 100) if total > 0 else 0
                st.metric("📈 Success Rate", f"{success_rate:.1f}%")
            
            st.markdown("---")
            
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()
            
            st.markdown("---")
            
            # Display history (most recent first)
            for i, h in enumerate(reversed(history), 1):
                status_icon = "✅" if h.get("ok") else "❌"
                question_preview = h.get('question', 'No question')[:70]
                
                with st.expander(
                    f"{status_icon} Query #{total - i + 1}: {question_preview}...",
                    expanded=(i == 1)
                ):
                    st.markdown(f"**❓ Question:**")
                    st.write(h.get('question', 'N/A'))
                    
                    if h.get("sql"):
                        st.markdown("**📝 Generated SQL:**")
                        st.code(h.get("sql"), language="sql")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if h.get("rows") is not None:
                            st.metric("📊 Rows Returned", h.get("rows"))
                    with col2:
                        status_text = "✅ Success" if h.get("ok") else "❌ Failed"
                        st.metric("Status", status_text)
                    
                    if h.get("explanation"):
                        st.markdown("**💡 AI Analysis:**")
                        st.info(h.get("explanation"))

    # ====================================
    # TAB 5: VISUALIZATIONS
    # ====================================
    with tab5:
        st.markdown("### 🎨 Data Visualization Studio")
        
        last = st.session_state.get("last_result", {"columns": [], "records": []})
        last_df = pd.DataFrame(last["records"]) if last["columns"] and last["records"] else None
        
        table_options = []
        if last_df is not None and not last_df.empty:
            table_options.append("-- Last Query Result --")
        if tables:
            table_options += tables
        
        if not table_options:
            st.info("📊 Execute a query or upload data to create visualizations")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                table_choice = st.selectbox("📋 Select Data Source", table_options)
            
            # Load data
            if table_choice == "-- Last Query Result --":
                df_vis = last_df
                st.success(f"✅ Using last query result ({len(df_vis)} rows, {len(df_vis.columns)} columns)")
            else:
                with col2:
                    row_limit = st.number_input(
                        "Row Limit",
                        min_value=10,
                        max_value=10000,
                        value=500,
                        step=100
                    )
                
                with st.spinner("⏳ Loading data..."):
                    if selected_db_type == "sqlite" and selected_db_path:
                        df_vis = load_full_table_path(selected_db_path, table_choice, n=row_limit)
                    else:
                        df_vis = load_full_table_engine(selected_db_engine, table_choice, n=row_limit)
            
            if df_vis is not None and not df_vis.empty:
                numeric_cols = df_vis.select_dtypes(include="number").columns.tolist()
                all_cols = df_vis.columns.tolist()
                
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns found for visualization")
                else:
                    st.markdown("---")
                    st.markdown("#### 📊 Chart Configuration")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        chart_type = st.selectbox(
                            "Chart Type",
                            ["Scatter", "Line", "Bar", "Pie", "Histogram", "Box", "Area"]
                        )
                    
                    with col2:
                        x_col = st.selectbox(
                            "X-axis",
                            all_cols if chart_type in ["Bar", "Line", "Area"] else numeric_cols
                        )
                    
                    with col3:
                        if chart_type not in ["Histogram", "Pie"]:
                            y_col = st.selectbox(
                                "Y-axis",
                                numeric_cols,
                                index=min(1, len(numeric_cols) - 1) if len(numeric_cols) > 1 else 0
                            )
                        elif chart_type == "Pie":
                            y_col = st.selectbox("Values", numeric_cols)
                    
                    with col4:
                        if chart_type in ["Scatter", "Line", "Bar"]:
                            color_col = st.selectbox(
                                "Color by (optional)",
                                ["None"] + all_cols
                            )
                            color_col = None if color_col == "None" else color_col
                    
                    st.markdown("---")
                    
                    # Generate visualization
                    try:
                        unique_key = f"viz_main_{chart_type}_{x_col}_{uuid.uuid4().hex[:8]}"
                        
                        if chart_type == "Scatter":
                            fig = px.scatter(
                                df_vis,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} vs {x_col}",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                        
                        elif chart_type == "Line":
                            fig = px.line(
                                df_vis,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} over {x_col}",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                        
                        elif chart_type == "Bar":
                            fig = px.bar(
                                df_vis,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} by {x_col}",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                        
                        elif chart_type == "Pie":
                            fig = px.pie(
                                df_vis,
                                names=x_col,
                                values=y_col,
                                title=f"{y_col} Distribution by {x_col}",
                                color_discrete_sequence=px.colors.sequential.Blues_r
                            )
                        
                        elif chart_type == "Histogram":
                            fig = px.histogram(
                                df_vis,
                                x=x_col,
                                title=f"Distribution of {x_col}",                               
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )

                        elif chart_type == "Box":
                            fig = px.box(
                                df_vis,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} Distribution across {x_col}",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )

                        elif chart_type == "Area":
                            fig = px.area(
                                df_vis,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} Trend over {x_col}",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )

                        fig.update_layout(
                            height=500,
                            template="plotly_white",
                            font=dict(size=12)
                        )

                        st.plotly_chart(fig, use_container_width=True, key=unique_key)

                    except Exception as e:
                        st.error(f"❌ Failed to generate chart: {str(e)}")
            else:
                st.info("📊 No data available for visualization")

    # ====================================
    # TAB 6: DASHBOARD
    # ====================================
    # ====================================
    # TAB 6: DASHBOARD
    # ====================================
# ====================================
    # TAB 6: DASHBOARD
    # ====================================
    with tab6:
        st.markdown("### 🗺️ Comprehensive Database Dashboard")

        if not tables:
            st.info("📊 Connect to a database to view the dashboard")
        else:
            # Ensure unique & clean table names
            tables = [t for t in tables if not t.startswith("_") and "filter" not in t.lower()]
            tables = sorted(list(set(tables)))

            # ---- Overview Metrics ----
            st.markdown("#### 📊 Database Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📋 Total Tables", len(tables))

            with col2:
                total_rows = 0
                for t in tables:
                    try:
                        if selected_db_type == "sqlite" and selected_db_path:
                            with sqlite3.connect(selected_db_path) as conn:
                                count = conn.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
                                total_rows += count
                        else:
                            count = pd.read_sql_query(
                                f"SELECT COUNT(*) as cnt FROM '{t}'",
                                selected_db_engine
                            ).iloc[0]['cnt']
                            total_rows += count
                    except:
                        pass
                st.metric("📝 Total Rows", f"{total_rows:,}")

            with col3:
                st.metric("💬 Queries Executed", len(st.session_state.chat_history))

            with col4:
                success_rate = 0
                if st.session_state.chat_history:
                    successful = sum(1 for h in st.session_state.chat_history if h.get("ok"))
                    success_rate = (successful / len(st.session_state.chat_history)) * 100
                st.metric("✅ Success Rate", f"{success_rate:.0f}%")

            # ---- Table Summaries ----
            st.markdown("---")
            st.markdown("### 📋 Detailed Table Information")

            for t in tables:
                with st.expander(f"🗂️ Table: **{t}**", expanded=False):
                    if selected_db_type == "sqlite" and selected_db_path:
                        df_dash = load_full_table_path(selected_db_path, t, n=300)
                    else:
                        df_dash = load_full_table_engine(selected_db_engine, t, n=300)

                    if df_dash.empty:
                        st.info("ℹ️ No data available in this table")
                        continue

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("**📄 Preview (Top 10 rows)**")
                        st.dataframe(df_dash.head(10), use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("**📊 Table Information**")
                        st.metric("Total Rows", f"{len(df_dash):,}")
                        st.metric("Total Columns", len(df_dash.columns))

                        # Data type distribution
                        type_counts = df_dash.dtypes.value_counts()
                        st.markdown("**🔢 Column Types:**")
                        for dtype, count in type_counts.items():
                            st.caption(f"• {dtype}: {count}")

                    # ---- SCHEMA DETAILS WITH PRIMARY KEYS ----
                    st.markdown("---")
                    st.markdown("**🔍 Schema Details**")
                    
                    try:
                        if selected_db_type == "sqlite" and selected_db_path:
                            with sqlite3.connect(selected_db_path) as conn:
                                # Get column info with primary keys
                                cols_info = conn.execute(f"PRAGMA table_info('{t}');").fetchall()
                                
                                if cols_info:
                                    schema_data = []
                                    pk_columns = []
                                    
                                    for col in cols_info:
                                        # col format: (cid, name, type, notnull, dflt_value, pk)
                                        col_id, col_name, col_type, not_null, default_val, is_pk = col
                                        
                                        if is_pk:
                                            pk_columns.append(col_name)
                                        
                                        schema_data.append({
                                            "Column": col_name,
                                            "Type": col_type,
                                            "Not Null": "Yes" if not_null else "No",
                                            "Primary Key": "🔑" if is_pk else ""
                                        })
                                    
                                    df_schema = pd.DataFrame(schema_data)
                                    st.dataframe(df_schema, use_container_width=True, hide_index=True)
                                    
                                    # Show primary keys explicitly
                                    if pk_columns:
                                        st.success(f"🔑 **Primary Key(s):** {', '.join(pk_columns)}")
                                    else:
                                        st.info("ℹ️ No primary key defined for this table")
                                    
                                    # Get foreign keys
                                    st.markdown("**🔗 Foreign Key Relationships:**")
                                    fk_results = conn.execute(f"PRAGMA foreign_key_list('{t}');").fetchall()
                                    
                                    if fk_results:
                                        for fk in fk_results:
                                            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
                                            fk_id, seq, ref_table, from_col, to_col = fk[0], fk[1], fk[2], fk[3], fk[4]
                                            st.caption(f"• **{from_col}** → **{ref_table}.{to_col}**")
                                    else:
                                        st.caption("• No foreign keys defined")
                                else:
                                    st.warning("⚠️ Could not retrieve schema information")
                        
                        elif selected_db_engine:
                            inspector = inspect(selected_db_engine)
                            
                            # Get columns
                            columns = inspector.get_columns(t)
                            
                            # Get primary key
                            pk_constraint = inspector.get_pk_constraint(t)
                            pk_columns = pk_constraint.get('constrained_columns', []) if pk_constraint else []
                            
                            # Build schema dataframe
                            schema_data = []
                            for col in columns:
                                schema_data.append({
                                    "Column": col['name'],
                                    "Type": str(col['type']),
                                    "Nullable": "Yes" if col.get('nullable', True) else "No",
                                    "Primary Key": "🔑" if col['name'] in pk_columns else ""
                                })
                            
                            df_schema = pd.DataFrame(schema_data)
                            st.dataframe(df_schema, use_container_width=True, hide_index=True)
                            
                            # Show primary keys
                            if pk_columns:
                                st.success(f"🔑 **Primary Key(s):** {', '.join(pk_columns)}")
                            else:
                                st.info("ℹ️ No primary key defined for this table")
                            
                            # Get foreign keys
                            st.markdown("**🔗 Foreign Key Relationships:**")
                            fks = inspector.get_foreign_keys(t)
                            
                            if fks:
                                for fk in fks:
                                    ref_table = fk.get('referred_table')
                                    const_cols = fk.get('constrained_columns', [])
                                    ref_cols = fk.get('referred_columns', [])
                                    
                                    for from_col, to_col in zip(const_cols, ref_cols):
                                        st.caption(f"• **{from_col}** → **{ref_table}.{to_col}**")
                            else:
                                st.caption("• No foreign keys defined")
                    
                    except Exception as e:
                        st.error(f"❌ Error retrieving schema: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

                    # Quick visualization for numeric columns
                    numeric = df_dash.select_dtypes(include="number").columns.tolist()
                    if numeric and len(numeric) > 0:
                        st.markdown("---")
                        st.markdown("**📊 Quick Visualization**")

                        try:
                            unique_key = f"dash_chart_{t}_{uuid.uuid4().hex[:6]}"
                            if len(numeric) >= 2:
                                fig = px.scatter(
                                    df_dash,
                                    x=numeric[0],
                                    y=numeric[1],
                                    title=f"{numeric[1]} vs {numeric[0]}",
                                    color_discrete_sequence=['#3b82f6']
                                )
                            else:
                                fig = px.histogram(
                                    df_dash,
                                    x=numeric[0],
                                    title=f"Distribution of {numeric[0]}",
                                    color_discrete_sequence=['#3b82f6']
                                )

                            fig.update_layout(
                                height=300,
                                template="plotly_white",
                                font=dict(size=10)
                            )
                            st.plotly_chart(fig, use_container_width=True, key=unique_key)

                        except Exception as e:
                            st.caption(f"⚠️ Could not generate chart: {e}")

            # ---- ER DIAGRAM WITH ENHANCED DEBUGGING ----
            st.markdown("---")
            st.markdown("### 🗺️ Entity Relationship Diagram")
            
            # Debug expander
            with st.expander("🔧 Debug Information", expanded=False):
                st.write(f"**Database Type:** {selected_db_type}")
                st.write(f"**Database Path:** {selected_db_path if selected_db_path else 'Using Engine'}")
                st.write(f"**Tables Found:** {', '.join(tables)}")

            try:
                if selected_db_type == "sqlite" and selected_db_path:
                    engine = create_engine(f"sqlite:///{selected_db_path}")
                else:
                    engine = selected_db_engine

                inspector = inspect(engine)
                G = nx.DiGraph()

                # Add all tables
                all_tables = inspector.get_table_names()
                
                # Filter out system tables
                all_tables = [t for t in all_tables if not t.startswith("sqlite_")]
                
                for table in all_tables:
                    G.add_node(table, node_type="table")

                # Track relationships
                relationships_found = 0
                relationship_details = []

                # Method 1: SQLAlchemy Inspector
                st.write("**🔍 Method 1: Checking via SQLAlchemy Inspector...**")
                for table in all_tables:
                    try:
                        fks = inspector.get_foreign_keys(table)
                        if fks:
                            st.caption(f"✓ Table '{table}' has {len(fks)} foreign key(s)")
                        for fk in fks:
                            ref_table = fk.get("referred_table")
                            const_cols = fk.get("constrained_columns", [])
                            ref_cols = fk.get("referred_columns", [])
                            
                            if ref_table and ref_table in all_tables:
                                if not G.has_edge(table, ref_table):
                                    G.add_edge(table, ref_table, relation="foreign_key")
                                    relationships_found += 1
                                    relationship_details.append(f"FK: {table} ({', '.join(const_cols)}) → {ref_table} ({', '.join(ref_cols)})")
                    except Exception as e:
                        st.caption(f"⚠️ Could not check FKs for '{table}': {str(e)}")

                # Method 2: SQLite PRAGMA
                if selected_db_type == "sqlite" and selected_db_path:
                    st.write("**🔍 Method 2: Checking via SQLite PRAGMA...**")
                    try:
                        with sqlite3.connect(selected_db_path) as conn:
                            for table in all_tables:
                                try:
                                    fk_results = conn.execute(f"PRAGMA foreign_key_list('{table}');").fetchall()
                                    
                                    if fk_results:
                                        st.caption(f"✓ Table '{table}' has {len(fk_results)} FK constraint(s)")
                                        
                                    for fk in fk_results:
                                        ref_table = fk[2]
                                        from_col = fk[3]
                                        to_col = fk[4]
                                        
                                        if ref_table and ref_table in all_tables:
                                            if not G.has_edge(table, ref_table):
                                                G.add_edge(table, ref_table, relation="foreign_key")
                                                relationships_found += 1
                                                relationship_details.append(f"PRAGMA FK: {table}.{from_col} → {ref_table}.{to_col}")
                                except Exception as e:
                                    st.caption(f"⚠️ PRAGMA check failed for '{table}': {str(e)}")
                    except Exception as e:
                        st.error(f"SQLite connection error: {str(e)}")

                # Method 3: Heuristic Column Matching
                st.write("**🔍 Method 3: Checking via Column Name Patterns...**")
                if relationships_found == 0:
                    for table in all_tables:
                        try:
                            columns = inspector.get_columns(table)
                            for col in columns:
                                col_name = col['name'].lower()
                                
                                for other_table in all_tables:
                                    if other_table != table:
                                        other_lower = other_table.lower()
                                        
                                        # Multiple pattern matching
                                        patterns = [
                                            f"{other_lower}_id",
                                            f"{other_lower}id",
                                            f"id_{other_lower}",
                                            f"{other_lower}_key",
                                        ]
                                        
                                        # Also try singular/plural variations
                                        if other_lower.endswith('s'):
                                            patterns.append(f"{other_lower[:-1]}_id")
                                            patterns.append(f"{other_lower[:-1]}id")
                                        else:
                                            patterns.append(f"{other_lower}s_id")
                                        
                                        for pattern in patterns:
                                            if col_name == pattern or pattern in col_name:
                                                if not G.has_edge(table, other_table):
                                                    G.add_edge(table, other_table, relation="inferred")
                                                    relationships_found += 1
                                                    relationship_details.append(f"Inferred: {table}.{col_name} → {other_table}")
                                                    st.caption(f"✓ Inferred: {table}.{col_name} likely references {other_table}")
                                                break
                        except Exception as e:
                            st.caption(f"⚠️ Column check failed for '{table}': {str(e)}")
                
                if relationships_found == 0:
                    st.warning("⚠️ No relationships detected by any method")

                # Display relationship summary
                if relationship_details:
                    st.markdown("**📋 Relationships Detected:**")
                    for rel in relationship_details:
                        st.caption(f"• {rel}")

                # Draw the diagram
                if len(G.nodes()) > 0:
                    st.markdown("---")
                    fig, ax = plt.subplots(figsize=(16, 10))

                    # Layout
                    if len(G.nodes()) <= 5:
                        pos = nx.circular_layout(G, scale=2)
                    elif len(G.nodes()) <= 10:
                        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
                    else:
                        pos = nx.kamada_kawai_layout(G)

                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_color='#3b82f6', node_size=5000, alpha=0.9, ax=ax)
                    
                    # Draw labels
                    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='white', ax=ax)
                    
                    # Draw edges
                    if len(G.edges()) > 0:
                        nx.draw_networkx_edges(
                            G, pos, 
                            edge_color='#94a3b8', 
                            width=3, 
                            alpha=0.7, 
                            ax=ax, 
                            arrows=True,
                            arrowsize=25,
                            arrowstyle='->',
                            connectionstyle='arc3,rad=0.15',
                            node_size=5000
                        )

                        # Edge labels
                        edge_labels = nx.get_edge_attributes(G, 'relation')
                        if edge_labels:
                            nx.draw_networkx_edge_labels(
                                G, pos, 
                                edge_labels, 
                                font_size=9, 
                                font_color='#1e40af',
                                font_weight='bold',
                                ax=ax
                            )

                    ax.set_title("Database Schema Relationships", fontsize=22, fontweight='bold', pad=20)
                    ax.axis('off')
                    ax.set_facecolor('#f8fafc')
                    fig.patch.set_facecolor('#f8fafc')
                    plt.tight_layout()

                    st.pyplot(fig)
                    
                    # Summary
                    if relationships_found > 0:
                        st.success(f"✅ Displaying {len(G.nodes())} tables and {len(G.edges())} relationships")
                        
                        with st.expander("🔗 View All Relationships", expanded=True):
                            for edge in G.edges(data=True):
                                from_table, to_table, data = edge
                                rel_type = data.get('relation', 'unknown')
                                icon = "🔑" if rel_type == "foreign_key" else "🔍"
                                st.caption(f"{icon} **{from_table}** → **{to_table}** ({rel_type})")
                    else:
                        st.warning("⚠️ No relationships detected. Possible reasons:")
                        st.caption("• Tables don't have foreign key constraints defined")
                        st.caption("• Database doesn't enforce referential integrity")
                        st.caption("• Column naming doesn't follow common patterns")
                        st.caption("• Data was imported without preserving relationships")
                else:
                    st.info("📊 No tables found to visualize")

            except Exception as e:
                st.error(f"❌ Error generating ER diagram: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
