# app.py
"""
AI Text-to-SQL Pro Assistant - Full corrected version
Handles login/register/logout, DB upload (.db, .csv, .xlsx),
Query generation, Tab visualization, Insights, Chat history,
and Dashboard with fast caching and error-free execution.
"""

import os
import io
import sqlite3
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect

# Optional: Gemini LLM import
try:
    import google.generativeai as genai
except Exception:
    genai = None

# -------------------------
# CONFIG
# -------------------------
DB_FILE = "user_data.db"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# -------------------------
# AUTH FUNCTIONS
# -------------------------
def init_user_table():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
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

Return only one valid SQLite SELECT query (no explanation, no markdown).
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

# -------------------------
# DB UTILS
# -------------------------
def safe_read_excel(uploaded_file):
    excel = pd.ExcelFile(uploaded_file)
    result = {}
    for sheet in excel.sheet_names:
        df = excel.parse(sheet)
        result[sheet.strip().replace(" ", "_")] = df
    return result

def load_database(uploaded_file):
    name = uploaded_file.name.lower()
    uploaded_file.seek(0, io.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    if size > MAX_UPLOAD_SIZE:
        raise ValueError("File too large (>50MB).")
    conn = sqlite3.connect(DB_FILE)
    if name.endswith((".db", ".sqlite")):
        with open(DB_FILE, "wb") as f:
            f.write(uploaded_file.getbuffer())
        conn.close()
        conn = sqlite3.connect(DB_FILE)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        df.to_sql("data", conn, if_exists="replace", index=False)
    elif name.endswith(".xlsx"):
        sheets = safe_read_excel(uploaded_file)
        for tname, df in sheets.items():
            df.to_sql(tname, conn, if_exists="replace", index=False)
    else:
        conn.close()
        raise ValueError("Unsupported file type")
    conn.commit()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    return conn, tables

def get_schema_text(conn):
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

def validate_sql_with_explain(conn, sql):
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True
    except Exception:
        return False

# -------------------------
# CACHED TABLE LOADERS
# -------------------------
@st.cache_data
def load_table_sample(_db_path, table_name, n=100):
    if table_name is None:
        return pd.DataFrame()
    with sqlite3.connect(_db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {n}", conn)
    return df

@st.cache_data
def load_full_table(_db_path, table_name, n=None):
    if table_name is None:
        return pd.DataFrame()
    with sqlite3.connect(_db_path) as conn:
        sql = f"SELECT * FROM {table_name}"
        if n is not None:
            sql += f" LIMIT {n}"
        df = pd.read_sql_query(sql, conn)
    return df

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Text-to-SQL Pro", layout="wide")

# SESSION STATE INIT
for key in ["logged_in","username","display_name","chat_history","last_result"]:
    if key not in st.session_state:
        if key=="logged_in": st.session_state[key]=False
        elif key=="chat_history": st.session_state[key]=[]
        elif key=="last_result": st.session_state[key]={"columns":[],"records":[]}
        else: st.session_state[key]=None

# -------------------------
# SIDEBAR AUTH
# -------------------------
st.sidebar.title("Account")
if not st.session_state.logged_in:
    auth_mode = st.sidebar.selectbox("Action", ["Login", "Register"])
    username_input = st.sidebar.text_input("Username", key="auth_user")
    password_input = st.sidebar.text_input("Password", type="password", key="auth_pw")
    display_name_input = None
    if auth_mode=="Register":
        display_name_input = st.sidebar.text_input("Display name (optional)", key="auth_display")
        if st.sidebar.button("Register"):
            ok,msg=register_user(username_input.strip(), password_input, display_name_input)
            if ok: st.sidebar.success(msg)
            else: st.sidebar.error(msg)
    else:
        if st.sidebar.button("Login"):
            ok,msg=login_user(username_input.strip(), password_input)
            if ok:
                st.session_state.logged_in=True
                st.session_state.username=username_input.strip()
                st.session_state.display_name=msg
                st.sidebar.success(f"Welcome, {msg}")
                if hasattr(st, "runtime") and hasattr(st.runtime, "legacy_rerun"):
                    st.runtime.legacy_rerun()
                else:
                    st.warning("Please refresh page.")
            else:
                st.sidebar.error(msg)
else:
    st.sidebar.markdown(f"👋 Logged in as **{st.session_state.display_name}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.session_state.logged_in=False
        st.session_state.chat_history=[]
        if hasattr(st, "runtime") and hasattr(st.runtime, "legacy_rerun"):
            st.runtime.legacy_rerun()
        else:
            st.warning("Please refresh page.")

# -------------------------
# UPLOAD DATABASE
# -------------------------
if not st.session_state.logged_in:
    st.info("Please login/register to use the app.")
    st.stop()

uploaded = st.file_uploader("Upload .db / .csv / .xlsx", type=["db","sqlite","csv","xlsx"])
conn = None
tables = []
schema_text = ""
if uploaded:
    try:
        conn, tables = load_database(uploaded)
        schema_text = get_schema_text(conn)
        st.success("Database loaded.")
    except Exception as e:
        st.error(f"Upload error: {e}")
        conn, tables = None, []

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Query","📘 Schema","📊 Insights","🕘 Chat History","🎨 Visualization","🗺️ Dashboard"])

# -------------------------
# TAB 1: QUERY
# -------------------------
with tab1:
    st.subheader("Ask a question")
    question = st.text_input("Type your question", key="main_question")
    universal_mode = st.checkbox("Universal SQL mode (no DB)")
    manual_sql = st.text_area("Or paste SQL manually", height=100)

    if st.button("Generate & Run"):
        chat_entry = {"question": question.strip(), "sql": None, "rows": None, "explanation": None, "ok": False}
        schema_context = schema_text if conn else "No schema"

        # ✅ 1. Early validation: reject nonsense or too short inputs
        invalid_prompt = (
            not question.strip()
            or len(question.split()) < 2
            or any(c in question for c in "!@#$%^&*()_=+[]{}<>?/\\|")
            or question.strip().isdigit()
        )
        gibberish_patterns = [
            "asdf", "qwer", "zxcv", "emhf", "vkwef", "random",
            "test", "hello", "ai", "emotion", "amd", "who are you"
        ]
        if invalid_prompt or any(pat in question.lower() for pat in gibberish_patterns):
            st.error("❌ Invalid prompt — please enter a meaningful data-related question.")
            chat_entry["ok"] = False
            st.session_state.chat_history.append(chat_entry)
            st.stop()

        # ✅ 2. Manual SQL check
        sql_to_run = manual_sql.strip() if manual_sql.strip() else None

        # ✅ 3. Generate SQL if not manually provided
        if sql_to_run is None and genai:
            try:
                sql_to_run = generate_sql_with_gemini(question.strip(), schema_context, st.session_state.chat_history)
            except Exception as e:
                st.warning(f"SQL generation unavailable: {e}")

        # ✅ 4. Validate generated SQL
        if sql_to_run:
            chat_entry["sql"] = sql_to_run
            st.code(sql_to_run, language="sql")

            # 🚫 Reject meaningless SQL (e.g. SELECT 1, SELECT 'text', etc.)
            fake_sql_patterns = [
                "select 1", "select '", 'select "', "select * from dual",
                "ai", "response", "emotion", "hello", "world"
            ]
            if any(pat in sql_to_run.lower() for pat in fake_sql_patterns):
                st.error("❌ Invalid query — unrelated or meaningless SQL generated.")
                chat_entry["ok"] = False
                st.session_state.chat_history.append(chat_entry)
                st.stop()

            elif universal_mode or conn is None:
                chat_entry["ok"] = True
                st.success("SQL generated (universal mode).")
                st.session_state.chat_history.append(chat_entry)

            else:
                if not is_safe_sql(sql_to_run):
                    st.error("Unsafe SQL — only SELECT allowed.")
                    chat_entry["ok"] = False

                elif not validate_sql_with_explain(conn, sql_to_run):
                    st.error("SQL invalid (EXPLAIN failed).")
                    chat_entry["ok"] = False

                else:
                    try:
                        df = pd.read_sql_query(sql_to_run, conn)

                        # 🚫 Reject empty or fake outputs
                        if (
                            df.empty
                            or (df.shape[1] == 1 and df.columns[0].lower() in ["response", "result", "output"])
                        ):
                            st.error("Invalid query — not based on dataset tables.")
                            chat_entry["ok"] = False
                        else:
                            chat_entry["rows"] = len(df)
                            chat_entry["ok"] = True
                            st.dataframe(df, use_container_width=True)

                            # AI Explanation
                            if genai:
                                try:
                                    prompt = f"Explain key findings from this result for question: {question}\nPreview:\n{df.head(5).to_string(index=False)}"
                                    model = genai.GenerativeModel("gemini-2.5-flash")
                                    resp = model.generate_content(prompt)
                                    chat_entry["explanation"] = resp.text.strip()
                                    st.info(resp.text.strip())
                                except:
                                    st.info("AI explanation unavailable.")

                            st.session_state.last_result = {
                                "columns": df.columns.tolist(),
                                "records": df.to_dict(orient="records"),
                            }
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button("Download CSV", csv, file_name="query_results.csv")

                        st.session_state.chat_history.append(chat_entry)

                    except Exception as e:
                        st.error(f"Query execution error: {e}")
                        chat_entry["ok"] = False
                        st.session_state.chat_history.append(chat_entry)
        else:
            st.info("No SQL to run. Paste SQL manually or LLM unavailable.")

# -------------------------
# TAB 2: SCHEMA
# -------------------------
with tab2:
    st.subheader("Database Schema")
    if not conn: st.info("Upload DB to see schema.")
    else: st.code(schema_text, language="sql")

# -------------------------
# TAB 3: INSIGHTS
# -------------------------
with tab3:
    st.subheader("Table Insights")
    if not conn: st.info("Upload DB to see insights.")
    else:
        for t in tables:
            st.markdown(f"### Table: `{t}`")
            df_head = load_table_sample(DB_FILE, t, n=100)
            if df_head.empty:
                st.info("No data in this table.")
                continue
            st.dataframe(df_head, use_container_width=True)
            with st.expander("Summary stats"):
                st.dataframe(df_head.describe(include="all").transpose())
            missing = df_head.isna().sum().to_dict()
            if any(v>0 for v in missing.values()):
                st.warning(f"Missing values (preview): {missing}")
            else:
                st.info("No missing values in preview.")

# -------------------------
# TAB 4: CHAT HISTORY
# -------------------------
with tab4:
    st.subheader("Chat History")
    history = st.session_state.get("chat_history", [])
    if not history: st.info("No history yet.")
    else:
        for i,h in enumerate(reversed(history),1):
            st.markdown(f"**#{i} — Question:** {h.get('question')}")
            if h.get("sql"): st.code(h.get("sql"), language="sql")
            rows=h.get("rows")
            if rows is not None: st.write(f"Rows: {rows}")
            if h.get("explanation"): st.info(h.get("explanation"))
            status="Success" if h.get("ok") else "Failed / Partial"
            st.caption(f"Status: {status}")
            st.markdown("---")

# -------------------------
# TAB 5: VISUALIZATION
# -------------------------
with tab5:
    st.subheader("Visualize table or last query")
    last = st.session_state.get("last_result", {"columns":[],"records":[]})
    last_df = pd.DataFrame(last["records"]) if last["columns"] and last["records"] else None
    table_options=[]
    if last_df is not None: table_options.append("-- last query result --")
    if tables: table_options+=tables
    if not table_options:
        st.info("Upload DB or run query first.")
    else:
        table_choice = st.selectbox("Select table", table_options)
        if table_choice=="-- last query result --": df_vis=last_df
        elif table_choice in tables: df_vis=load_full_table(DB_FILE, table_choice)
        else: df_vis=None
        if df_vis is not None and not df_vis.empty:
            numeric_cols = df_vis.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1,len(numeric_cols)-1))
                chart_type = st.selectbox("Chart type", ["Scatter","Line","Bar","Pie"])
                fig = px.scatter(df_vis, x=x_col, y=y_col)
                if chart_type=="Line": fig=px.line(df_vis,x=x_col,y=y_col)
                elif chart_type=="Bar": fig=px.bar(df_vis,x=x_col,y=y_col)
                elif chart_type=="Pie": fig=px.pie(df_vis,names=x_col,values=y_col)
                st.plotly_chart(fig,use_container_width=True)
            else:
                st.info("No numeric columns to plot.")

# -------------------------
# TAB 6: DASHBOARD
# -------------------------
with tab6:
    st.subheader("Dashboard Preview & ER Diagram")
    if not tables: st.info("Upload DB to see dashboard.")
    else:
        for t in tables:
            df_dash=load_full_table(DB_FILE, t, n=200)
            st.markdown(f"**Table `{t}` top 10 rows**")
            st.dataframe(df_dash.head(10))
            numeric=df_dash.select_dtypes(include="number").columns.tolist()
            if numeric:
                st.bar_chart(df_dash[numeric])
        # ER Diagram
        try:
            engine=create_engine(f"sqlite:///{DB_FILE}")
            inspector=inspect(engine)
            G=nx.Graph()
            for t in inspector.get_table_names():
                G.add_node(t)
                for c in inspector.get_columns(t):
                    G.add_edge(t, c["name"])
            fig,ax=plt.subplots(figsize=(10,4))
            nx.draw(G,with_labels=True,node_color="lightblue",node_size=900,font_size=8)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ER Diagram error: {e}")
