"""
AI Text-to-SQL Pro Assistant - Enhanced Professional Version with Image Support & Normalization
"""

import os
import io
import sqlite3
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from PIL import Image
import pytesseract

# Optional: Gemini LLM import
try:
    import google.generativeai as genai
except Exception:
    genai = None

# -------------------------
# ENHANCED PROFESSIONAL STYLING
# -------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Professional color palette */
    :root {
        --primary: #3b82f6;
        --primary-dark: #1e40af;
        --secondary: #0891b2;
        --success: #059669;
        --warning: #d97706;
        --danger: #dc2626;
        --dark: #1e293b;
        --light: #f8fafc;
        --border: #e2e8f0;
    }

    .main {
        padding: 2rem 3rem;
        background-color: #f9fafb;
    }

    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
        text-align: center;
    }

    .hero-card h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .hero-card p {
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-size: 1.15rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.7rem 2rem;
        border: none;
        transition: all 0.3s ease-in-out;
        background-color: var(--primary) !important;
        color: white !important;
        font-size: 1rem;
        letter-spacing: 0.4px;
    }

    .stButton>button:hover {
        background-color: var(--primary-dark) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }

    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }

    .element-container .stInfo {
        background-color: #dbeafe;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 8px;
        color: #1e40af;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #334155 100%);
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton>button {
        background-color: #2563eb !important;
        color: white !important;
        width: 100%;
    }

    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #1e40af !important;
    }

    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
    }

    h1, h2, h3 {
        font-weight: 700;
        color: #1e293b;
    }

    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
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
# AUTH FUNCTIONS
# -------------------------
def init_user_table():
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
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username, password, display_name=None):
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

def generate_sql_with_gemini(question, schema, chat_history=None):
    if genai is None:
        raise RuntimeError("LLM not configured.")
    if chat_history is None:
        chat_history = []
    context = "\n".join([f"Q: {h.get('question')} → SQL: {h.get('sql')}" for h in chat_history[-3:]])
    prompt = f"""
Schema:
{schema}

Conversation context:
{context}

User question:
{question}

Return only one valid SELECT query (no explanation, no markdown).
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    sql = (resp.text or "").strip().replace("```sql", "").replace("```", "").strip()
    if "select" in sql.lower():
        sql = sql[sql.lower().find("select"):]
    sql = sql.split(";")[0].strip()
    if not sql.lower().startswith("select"):
        raise ValueError("LLM did not produce a valid SELECT query.")
    return sql

def generate_ai_explanation(question, df):
    if genai is None:
        return None
    try:
        prompt = f"""Analyze this SQL query result and provide key insights.

Question: {question}

Preview:
{df.head(5).to_string(index=False)}

Rows: {len(df)}
Columns: {', '.join(df.columns.tolist())}
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

# -------------------------
# DB UTILS
# -------------------------
def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)

def safe_read_excel(uploaded_file):
    excel = pd.ExcelFile(uploaded_file)
    result = {}
    for sheet in excel.sheet_names:
        df = excel.parse(sheet)
        result[sheet.strip().replace(" ", "_")] = df
    return result

def load_database_to_sqlite(uploaded_file, auto_normalize=False):
    orig_name = uploaded_file.name
    base = _safe_filename(orig_name.rsplit(".", 1)[0])
    target_db = f"{UPLOAD_DB_PREFIX}{base}.db"
    uploaded_file.seek(0, io.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    if size > MAX_UPLOAD_SIZE:
        raise ValueError("File too large (>50MB).")

    lower = orig_name.lower()
    conn = sqlite3.connect(target_db)
    
    try:
        if lower.endswith((".db", ".sqlite")):
            conn.close()
            with open(target_db, "wb") as f:
                f.write(uploaded_file.getbuffer())
            conn = sqlite3.connect(target_db)
            
        elif lower.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df.to_sql(base or "data", conn, if_exists="replace", index=False)
                
        elif lower.endswith(".xlsx"):
            sheets = safe_read_excel(uploaded_file)
            for sheet_name, df in sheets.items():
                df.to_sql(sheet_name, conn, if_exists="replace", index=False)
        else:
            conn.close()
            raise ValueError("Unsupported file type")
            
        conn.commit()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        conn.close()
        return target_db, tables
        
    except Exception as e:
        conn.close()
        raise e

def get_schema_text_from_engine(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    lines = []
    for t in tables:
        cols = [c["name"] for c in inspector.get_columns(t)]
        lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)

def get_schema_text_from_path(db_path):
    with sqlite3.connect(db_path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        lines = []
        for t in tables:
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t});").fetchall()]
            lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)





# -------------------------
# SAFE SQL
# -------------------------
DANGEROUS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
def is_safe_sql(sql):
    if not sql: return False
    up = sql.upper()
    for kw in DANGEROUS:
        if kw in up:
            return False
    return up.strip().startswith("SELECT")

# -------------------------
# TABLE LOADERS
# -------------------------
@st.cache_data
def load_table_sample_path(db_path, table_name, n=100):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {n}", conn)

@st.cache_data
def load_table_sample_engine(engine, table_name, n=100):
    try:
        return pd.read_sql_table(table_name, engine).head(n)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_full_table_path(db_path, table_name, n=None):
    with sqlite3.connect(db_path) as conn:
        sql = f"SELECT * FROM {table_name}" + (f" LIMIT {n}" if n else "")
        return pd.read_sql_query(sql, conn)

@st.cache_data
def load_full_table_engine(engine, table_name, n=None):
    try:
        df = pd.read_sql_table(table_name, engine)
        return df.head(n) if n else df
    except Exception:
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

# SESSION STATE INIT
for key in ["logged_in","username","display_name","chat_history","last_result","databases","selected_db"]:
    if key not in st.session_state:
        if key=="logged_in": st.session_state[key]=False
        elif key=="chat_history": st.session_state[key]=[]
        elif key=="last_result": st.session_state[key]={"columns":[],"records":[]}
        elif key=="databases": st.session_state[key]={}
        else: st.session_state[key]=None

# -------------------------
# SIDEBAR AUTH
# -------------------------
with st.sidebar:
    st.markdown("### 🔐 Account")
    
    if not st.session_state.logged_in:
        auth_mode = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        username_input = st.text_input("👤 Username", key="auth_user")
        password_input = st.text_input("🔑 Password", type="password", key="auth_pw")
        
        if auth_mode == "Register":
            display_name_input = st.text_input("✨ Display name (optional)", key="auth_display")
            if st.button("Create Account", use_container_width=True, key="register_btn"):
                ok, msg = register_user(username_input.strip(), password_input, display_name_input)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            if st.button("Sign In", use_container_width=True, key="login_btn", type="primary"):
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
        <div class="stat-card">
            <h3>👋</h3>
            <p>{st.session_state.display_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        num_dbs = len(st.session_state.databases)
        num_queries = len(st.session_state.chat_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Databases", num_dbs)
        with col2:
            st.metric("💬 Queries", num_queries)
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.session_state.logged_in = False
            st.rerun()

# -------------------------
# FRONTEND AFTER LOGIN
# -------------------------
if st.session_state.logged_in:
    st.markdown("""
    <div class="hero-card">
        <h1>🤖 AI Text-to-SQL Pro</h1>
        <p>Transform natural language into powerful SQL queries with intelligent assistance</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Database Connection Section
    with st.expander("📥 Data Source Management", expanded=True):
        st.markdown("#### 📂 Upload Files")
        
        col_upload, col_settings = st.columns([2, 1])
        
        with col_upload:
            uploaded_files = st.file_uploader(
                "Select files to upload",
                type=["db","sqlite","csv","xlsx","png","jpg","jpeg"],
                accept_multiple_files=True,
                help="Supported: SQLite, CSV, Excel, Images (PNG/JPG)",
                label_visibility="collapsed"
            )
        
        with col_settings:
            auto_normalize = st.checkbox(
                "🔄 Auto-Normalize Tables",
                value=False,
                help="Automatically normalize uploaded tables to 3NF"
            )
            st.info("**💡 Quick Tips**\n\n✓ Max: 50MB\n✓ Multi-file\n✓ Image OCR")
        
        if uploaded_files:
            for f in uploaded_files:
                if f.name in st.session_state["databases"]: 
                    continue
                try:
                    with st.spinner(f"Processing {f.name}..."):
                        db_path, tables = load_database_to_sqlite(f, auto_normalize)
                        st.session_state["databases"][f.name] = {
                            "type":"sqlite",
                            "path":db_path,
                            "tables":tables,
                            "engine":None
                        }
                        norm_msg = " (normalized)" if auto_normalize else ""
                        st.success(f"✅ {f.name} ({len(tables)} tables{norm_msg})")
                except Exception as e:
                    st.error(f"❌ {f.name}: {str(e)}")

        # Server connection
        st.markdown("---")
        st.markdown("#### 🔌 Database Server Connection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            db_type = st.selectbox("Database Type", ["Select","MSSQL","MySQL","PostgreSQL"])
        
        with col2:
            if db_type != "Select":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    host = st.text_input("Host", placeholder="localhost")
                with col_b:
                    port = st.text_input("Port", placeholder="3306")
                with col_c:
                    db_name = st.text_input("Database")
                
                col_d, col_e, col_f = st.columns([1, 1, 1])
                with col_d:
                    user = st.text_input("Username")
                with col_e:
                    password = st.text_input("Password", type="password")
                with col_f:
                    st.write("")
                    st.write("")
                    if st.button("🔗 Connect", use_container_width=True):
                        with st.spinner("Connecting..."):
                            try:
                                if db_type=="MSSQL":
                                    conn_str=f"mssql+pyodbc://{user}:{password}@{host}:{port}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server"
                                elif db_type=="MySQL":
                                    conn_str=f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
                                elif db_type=="PostgreSQL":
                                    conn_str=f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
                                engine=create_engine(conn_str)
                                tables = inspect(engine).get_table_names()
                                st.session_state["databases"][f"{db_type}_{db_name}"]={
                                    "type":db_type.lower(),
                                    "engine":engine,
                                    "tables":tables,
                                    "path":None
                                }
                                st.success(f"✅ Connected to {db_type} ({len(tables)} tables)")
                            except SQLAlchemyError as e:
                                st.error(f"❌ Connection failed: {e}")

    # --- Select active DB
    st.markdown("### 🎯 Active Database")
    if st.session_state["databases"]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            db_choice = st.selectbox(
                "Select database",
                list(st.session_state["databases"].keys()),
                help="Choose which database to query",
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
            st.metric("✅ Successful", total_queries)
    else:
        st.info("👆 Upload or connect to a database to get started")
        st.stop()

    st.markdown("---")

    # -------------------------
    # TABS (Rest of the code remains the same as in the previous version)
    # -------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Query Assistant",
        "📘 Schema Explorer",
        "📊 Data Insights",
        "🕘 Query History",
        "🎨 Visualizations",
        "🗺️ Dashboard"
    ])

    with tab1:
        st.markdown("### 💬 Ask Your Question")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., Show me top 10 customers by revenue",
                label_visibility="collapsed"
            )
        with col2:
            use_ai = st.checkbox("✨ Smart Mode", value=True, help="Generate SQL with AI")
        
        manual_sql = st.text_area(
            "Or write SQL manually",
            height=120,
            placeholder="SELECT * FROM table_name LIMIT 10"
        )

        if st.button("▶️ Execute Query", type="primary", use_container_width=True):
            chat_entry = {"question": question.strip(), "sql": None, "rows": None, "explanation": None, "ok": False}

            # Validation
            import re
            def is_invalid_prompt(q):
                q_clean = q.strip().lower()
                if len(q_clean.split()) < 3: return True
                if re.fullmatch(r"[^\w]{1,}", q_clean): return True
                if len(q_clean) > 5 and len(set(q_clean.replace(" ", ""))) <= 2: return True
                meaningless = ["hello","test","asdf","qwer","zxcv","ai","random"]
                if any(word in q_clean for word in meaningless): return True
                return False

            if not manual_sql.strip() and is_invalid_prompt(question):
                st.error("❌ Please enter a meaningful question with at least 3 words")
                chat_entry["ok"] = False
                st.session_state.chat_history.append(chat_entry)
                st.stop()

            # Determine SQL
            sql_to_run = None
            schema_context = (
                get_schema_text_from_path(selected_db_path) if selected_db_path
                else get_schema_text_from_engine(selected_db_engine) if selected_db_engine
                else "No schema"
            )

            if manual_sql.strip():
                sql_to_run = manual_sql.strip()
            elif use_ai and genai:
                with st.spinner("✨ Smart Mode is generating your query..."):
                    try:
                        sql_to_run = generate_sql_with_gemini(question.strip(), schema_context, st.session_state.chat_history)
                    except Exception as e:
                        st.warning(f"⚠️ Smart Mode unavailable: {e}")
            elif tables:
                sql_to_run = f"SELECT * FROM {tables[0]} LIMIT 5"

            if not sql_to_run:
                st.error("❌ No SQL to execute. Enable Smart Mode or write manual SQL.")
                chat_entry["ok"] = False
                st.session_state.chat_history.append(chat_entry)
                st.stop()

            chat_entry["sql"] = sql_to_run
            
            st.markdown("#### 📝 Generated Query")
            st.code(sql_to_run, language="sql")

            # Execute
            try:
                with st.spinner("⚡ Executing query..."):
                    if selected_db_type == "sqlite" and selected_db_path:
                        df = pd.read_sql_query(sql_to_run, sqlite3.connect(selected_db_path))
                    elif selected_db_engine:
                        df = pd.read_sql_query(sql_to_run, selected_db_engine)
                    else:
                        df = pd.DataFrame()

                if df.empty:
                    st.warning("⚠️ Query returned no results")
                    chat_entry["ok"] = False
                else:
                    chat_entry["rows"] = len(df)
                    chat_entry["ok"] = True
                    
                    st.markdown("#### 📊 Results")
                    st.dataframe(df, use_container_width=True, height=400)

                    # AI Explanation
                    if genai and use_ai and question.strip():
                        st.markdown("#### 💡 AI Insights")
                        with st.spinner("✨ Analyzing results..."):
                            explanation = generate_ai_explanation(question.strip(), df)
                            if explanation:
                                chat_entry["explanation"] = explanation
                                st.info(explanation)

                    # Save & download
                    st.session_state.last_result = {
                        "columns": df.columns.tolist(),
                        "records": df.to_dict(orient="records")
                    }
                    
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        file_name="query_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
                chat_entry["ok"] = False

            st.session_state.chat_history.append(chat_entry)

    # TAB 2: SCHEMA EXPLORER
    with tab2:
        st.markdown("### 📘 Database Schema")
        
        if selected_db_type == "sqlite" and selected_db_path:
            schema_text = get_schema_text_from_path(selected_db_path)
        elif selected_db_engine:
            try:
                inspector = inspect(selected_db_engine)
                schema_lines = []
                for t in inspector.get_table_names():
                    cols = [c['name'] for c in inspector.get_columns(t)]
                    schema_lines.append(f"{t}({', '.join(cols)})")
                schema_text = "\n".join(schema_lines)
            except Exception as e:
                st.error(f"❌ Schema error: {e}")
                schema_text = ""
        else:
            st.info("Select a database to view schema")
            schema_text = ""
        
        if schema_text:
            st.code(schema_text, language="sql")
            
            st.markdown("---")
            st.markdown("### 📋 Table Details")
            
            for t in tables:
                with st.expander(f"📊 {t}"):
                    if selected_db_type == "sqlite":
                        with sqlite3.connect(selected_db_path) as conn:
                            cols_info = conn.execute(f"PRAGMA table_info({t});").fetchall()
                            df_cols = pd.DataFrame(cols_info, columns=["cid","name","type","notnull","dflt_value","pk"])
                            
                            st.markdown("**Column Information**")
                            st.dataframe(
                                df_cols[["name","type","notnull","pk"]].rename(columns={
                                    "name": "Column Name",
                                    "type": "Data Type",
                                    "notnull": "Not Null",
                                    "pk": "Primary Key"
                                }), 
                                use_container_width=True
                            )
                    else:
                        try:
                            inspector = inspect(selected_db_engine)
                            cols = inspector.get_columns(t)
                            df_cols = pd.DataFrame(cols)
                            st.dataframe(df_cols, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

    # TAB 3: DATA INSIGHTS
    with tab3:
        st.markdown("### 📊 Data Quality & Insights")
        
        if tables:
            for t in tables:
                with st.expander(f"📊 {t}", expanded=len(tables)==1):
                    if selected_db_type == "sqlite":
                        df_head = load_table_sample_path(selected_db_path, t, n=100)
                    else:
                        df_head = load_table_sample_engine(selected_db_engine, t, n=100)
                    
                    if df_head.empty:
                        st.info("No data available")
                        continue
                    
                    st.markdown("#### 📄 Sample Data")
                    st.dataframe(df_head.head(10), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Summary Statistics")
                        stats = df_head.describe(include="all").transpose()
                        st.dataframe(stats, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### ⚠️ Data Quality Check")
                        missing = df_head.isna().sum()
                        missing_pct = (missing / len(df_head) * 100).round(2)
                        quality_df = pd.DataFrame({
                            "Column": missing.index,
                            "Missing Values": missing.values,
                            "Missing %": missing_pct.values
                        })
                        quality_df = quality_df[quality_df["Missing Values"] > 0]
                        
                        if quality_df.empty:
                            st.success("✅ No missing values!")
                        else:
                            st.dataframe(quality_df, use_container_width=True)
                            st.warning(f"⚠️ Found missing values in {len(quality_df)} column(s)")
        else:
            st.info("📊 No tables available")

    # TAB 4: QUERY HISTORY
    with tab4:
        st.markdown("### 🕘 Query History")
        
        history = st.session_state.get("chat_history", [])
        
        if not history:
            st.info("📝 No queries yet. Start asking questions!")
        else:
            total = len(history)
            successful = sum(1 for h in history if h.get("ok"))
            failed = total - successful
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Queries", total)
            with col2:
                st.metric("✅ Successful", successful)
            with col3:
                st.metric("❌ Failed", failed)
            
            st.markdown("---")
            
            for i, h in enumerate(reversed(history), 1):
                status_icon = "✅" if h.get("ok") else "❌"
                question_preview = h.get('question', 'No question')[:80]
                
                with st.expander(f"{status_icon} Query #{i}: {question_preview}...", expanded=(i==1)):
                    st.markdown(f"**❓ Question:** {h.get('question', 'N/A')}")
                    
                    if h.get("sql"):
                        st.markdown("**📝 Generated SQL:**")
                        st.code(h.get("sql"), language="sql")
                    
                    if h.get("rows") is not None:
                        st.metric("📊 Rows Returned", h.get("rows"))
                    
                    if h.get("explanation"):
                        st.markdown("**💡 AI Analysis:**")
                        st.info(h.get("explanation"))
                    
                    status_text = "✅ Success" if h.get("ok") else "❌ Failed"
                    st.caption(f"**Status:** {status_text}")

    # TAB 5: VISUALIZATION
    with tab5:
        st.markdown("### 🎨 Data Visualization")
        
        last = st.session_state.get("last_result", {"columns":[],"records":[]})
        last_df = pd.DataFrame(last["records"]) if last["columns"] and last["records"] else None
        
        table_options = []
        if last_df is not None and not last_df.empty:
            table_options.append("-- Last Query Result --")
        if tables:
            table_options += tables
        
        if not table_options:
            st.info("📊 Run a query or upload data to create visualizations")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                table_choice = st.selectbox("📋 Select Data Source", table_options)
            
            if table_choice == "-- Last Query Result --":
                df_vis = last_df
                st.success(f"✅ Using last query result ({len(df_vis)} rows)")
            else:
                with col2:
                    row_limit = st.number_input("Row Limit", min_value=10, max_value=10000, value=500)
                
                with st.spinner("Loading data..."):
                    if selected_db_type == "sqlite":
                        df_vis = load_full_table_path(selected_db_path, table_choice, n=row_limit)
                    else:
                        df_vis = load_full_table_engine(selected_db_engine, table_choice, n=row_limit)
            
            if df_vis is not None and not df_vis.empty:
                numeric_cols = df_vis.select_dtypes(include="number").columns.tolist()
                all_cols = df_vis.columns.tolist()
                
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns found")
                else:
                    st.markdown("---")
                    st.markdown("#### 📊 Chart Configuration")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Pie", "Histogram", "Box"])
                    
                    with col2:
                        x_col = st.selectbox("X-axis", all_cols if chart_type in ["Bar", "Line"] else numeric_cols)
                    
                    with col3:
                        if chart_type != "Histogram":
                            y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0)
                    
                    st.markdown("---")
                    
                    try:
                        if chart_type == "Scatter":
                            fig = px.scatter(df_vis, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", color_discrete_sequence=['#2563eb'])
                        elif chart_type == "Line":
                            fig = px.line(df_vis, x=x_col, y=y_col, title=f"{y_col} over {x_col}", color_discrete_sequence=['#2563eb'])
                        elif chart_type == "Bar":
                            fig = px.bar(df_vis, x=x_col, y=y_col, title=f"{y_col} by {x_col}", color_discrete_sequence=['#2563eb'])
                        elif chart_type == "Pie":
                            fig = px.pie(df_vis, names=x_col, values=y_col, title=f"{y_col} Distribution", color_discrete_sequence=px.colors.sequential.Blues_r)
                        elif chart_type == "Histogram":
                            fig = px.histogram(df_vis, x=x_col, title=f"Distribution of {x_col}", color_discrete_sequence=['#2563eb'])
                        elif chart_type == "Box":
                            fig = px.box(df_vis, y=y_col, title=f"Box Plot of {y_col}", color_discrete_sequence=['#2563eb'])
                        
                        fig.update_layout(template="plotly_white", height=500, font=dict(size=12), title_font_size=20, title_font_color='#1e293b')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("#### 📈 Quick Statistics")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        if chart_type != "Histogram":
                            with stat_col1:
                                st.metric(f"{y_col} Mean", f"{df_vis[y_col].mean():.2f}")
                            with stat_col2:
                                st.metric(f"{y_col} Median", f"{df_vis[y_col].median():.2f}")
                            with stat_col3:
                                st.metric(f"{y_col} Max", f"{df_vis[y_col].max():.2f}")
                            with stat_col4:
                                st.metric(f"{y_col} Min", f"{df_vis[y_col].min():.2f}")
                        else:
                            with stat_col1:
                                st.metric(f"{x_col} Mean", f"{df_vis[x_col].mean():.2f}")
                            with stat_col2:
                                st.metric(f"{x_col} Median", f"{df_vis[x_col].median():.2f}")
                            with stat_col3:
                                st.metric(f"{x_col} Std Dev", f"{df_vis[x_col].std():.2f}")
                            with stat_col4:
                                st.metric("Count", len(df_vis))
                        
                    except Exception as e:
                        st.error(f"❌ Visualization error: {str(e)}")

    # TAB 6: DASHBOARD
    with tab6:
        st.markdown("### 🗺️ Database Dashboard")
        
        if not tables:
            st.info("📊 Connect to a database to view dashboard")
        else:
            st.markdown("#### 📊 Database Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📋 Total Tables", len(tables))
            
            with col2:
                total_rows = 0
                for t in tables:
                    try:
                        if selected_db_type == "sqlite":
                            with sqlite3.connect(selected_db_path) as conn:
                                count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                                total_rows += count
                        else:
                            count = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {t}", selected_db_engine).iloc[0]['cnt']
                            total_rows += count
                    except:
                        pass
                st.metric("📝 Total Rows", f"{total_rows:,}")
            
            with col3:
                st.metric("💬 Queries Run", len(st.session_state.chat_history))
            
            with col4:
                success_rate = 0
                if st.session_state.chat_history:
                    success_rate = sum(1 for h in st.session_state.chat_history if h.get("ok")) / len(st.session_state.chat_history) * 100
                st.metric("✅ Success Rate", f"{success_rate:.0f}%")
            
            st.markdown("---")
            st.markdown("#### 📋 Table Previews")
            
            for t in tables:
                with st.expander(f"📊 {t}", expanded=False):
                    if selected_db_type == "sqlite":
                        df_dash = load_full_table_path(selected_db_path, t, n=200)
                    else:
                        df_dash = load_full_table_engine(selected_db_engine, t, n=200)
                    
                    if df_dash.empty:
                        st.info("No data available")
                        continue
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Preview (Top 10 rows)**")
                        st.dataframe(df_dash.head(10), use_container_width=True)
                    
                    with col2:
                        st.markdown(f"**Table Information**")
                        st.metric("Total Rows", len(df_dash))
                        st.metric("Total Columns", len(df_dash.columns))
                        
                        type_counts = df_dash.dtypes.value_counts()
                        st.markdown("**Column Types:**")
                        for dtype, count in type_counts.items():
                            st.caption(f"• {dtype}: {count}")
                    
                    numeric = df_dash.select_dtypes(include="number").columns.tolist()
                    if numeric and len(numeric) > 0:
                        st.markdown("---")
                        st.markdown("**📊 Quick Visualization**")
                        
                        try:
                            if len(numeric) >= 2:
                                fig = px.scatter(df_dash, x=numeric[0], y=numeric[1], title=f"{numeric[1]} vs {numeric[0]}", color_discrete_sequence=['#2563eb'])
                            else:
                                fig = px.histogram(df_dash, x=numeric[0], title=f"Distribution of {numeric[0]}", color_discrete_sequence=['#2563eb'])
                            
                            fig.update_layout(height=300, template="plotly_white", font=dict(size=10))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate chart: {e}")
            
            st.markdown("---")
            st.markdown("#### 🗺️ Entity Relationship Diagram")
            
            try:
                if selected_db_type == "sqlite":
                    engine = create_engine(f"sqlite:///{selected_db_path}")
                else:
                    engine = selected_db_engine
                
                inspector = inspect(engine)
                G = nx.Graph()
                
                for t in inspector.get_table_names():
                    G.add_node(t, node_type="table")
                
                for t in inspector.get_table_names():
                    try:
                        fks = inspector.get_foreign_keys(t)
                        for fk in fks:
                            ref_table = fk.get('referred_table')
                            if ref_table:
                                G.add_edge(t, ref_table, relation="foreign_key")
                    except:
                        pass
                
                fig, ax = plt.subplots(figsize=(14, 8))
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                nx.draw_networkx_nodes(G, pos, node_color='#2563eb', node_size=3500, alpha=0.9, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='white', ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color='#94a3b8', width=2.5, alpha=0.6, ax=ax, style='solid')
                
                ax.set_title("Database Schema Relationships", fontsize=18, fontweight='bold', color='#1e293b', pad=20)
                ax.axis('off')
                ax.set_facecolor('#f8fafc')
                fig.patch.set_facecolor('#f8fafc')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                st.info(f"📊 Displaying {len(G.nodes())} tables and {len(G.edges())} relationships")
                
            except Exception as e:
                st.error(f"❌ Could not generate ER diagram: {str(e)}")
