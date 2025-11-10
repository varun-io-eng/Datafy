# 🤖 AI Text-to-SQL Pro Assistant  

An **AI-powered Streamlit application** that converts natural language queries into SQL commands, executes them across multiple databases, and provides instant visual analytics with interactive dashboards.

---

## 🚀 Overview  
AI Text-to-SQL Pro Assistant bridges the gap between non-technical users and data-driven insights.  
It leverages **Google Gemini LLM** to translate user questions into valid SQL queries, execute them securely, and visualize the results — all within an intuitive Streamlit interface.

---

## 🔑 Key Features  
- 🧠 **Text-to-SQL Conversion:** Converts plain English into optimized SQL queries using **Gemini LLM**.  
- 🧩 **Multi-Database Support:** Works with **SQLite, MySQL, PostgreSQL, and MSSQL**.  
- 📊 **Data Visualization:** Creates interactive charts using **Plotly** and **Matplotlib**.  
- 🔐 **User Authentication:** Secure login system with **SHA-256 password hashing**.  
- 📘 **Schema Explorer:** Automatically detects and visualizes database schema relationships.  
- 💬 **AI Insights:** Explains key findings from SQL query results.  

---

## 🧠 Why It Matters  
Data professionals and analysts spend hours crafting and debugging SQL queries.  
**Datafy** bridges the gap between **human language and data logic**, empowering users to:  
- Analyze data faster using natural questions  
- Understand database relationships visually  
- Generate dashboards and insights in real time  
- Enable smarter decision-making for both technical and non-technical teams  

This app is especially useful for:  
- 📈 Business analysts and managers  
- 🎓 Students and researchers  
- 🧮 Data teams handling ad-hoc analytics  
- 🚀 Startups and SMBs without full BI platforms  

---

## 🏗️ Tech Stack  
**Frontend:** Streamlit  
**Backend:** Python, SQLAlchemy  
**Databases:** SQLite, MySQL, PostgreSQL, MSSQL  
**AI Model:** Google Gemini LLM  
**Visualization:** Plotly, Matplotlib, NetworkX  
**Data Processing:** Pandas, NumPy  
**Authentication:** SQLite + SHA-256  
**Deployment:** Streamlit Cloud / AWS  


---

## ⚙️ How It Works  
**1️⃣ Upload Dataset**  
- Upload CSV, Excel, or SQLite database files.  

**2️⃣ Ask a Question or Enter SQL**  
- Type your question in plain English.  
- Or manually paste an SQL query (universal SQL mode).  
- Example: *“Show average revenue per region in 2024.”*  

**3️⃣ SQL Generation**  
- The app uses **Gemini AI** to generate SQL queries automatically.  
- Queries are validated for safety before execution.  

**4️⃣ Query Execution & Visualization**  
- Runs valid queries on the uploaded data.  
- Displays results as an interactive table and chart.  

**5️⃣ Insights & Dashboards**  
- Automatically creates meaningful charts and summaries.  
- Explains patterns and key findings using AI.  

**6️⃣ Schema & ER Diagram**  
- Visualizes table structures and relationships.  
- Helps users understand data organization quickly.  

**7️⃣ Chat History & Export**  
- Stores all previous questions, SQLs, and results.  
- Download query results as CSV anytime.  

---


## 💡 Key Learning Outcomes  
- Integration of **LLMs (Google Gemini)** for Text-to-SQL  
- Safe SQL validation and sandboxed query execution  
- Building **ER Diagrams** dynamically from schemas  
- Streamlit-based **interactive dashboards**  
- Combining **AI, data visualization, and databases** into one workflow  
- Deploying a full-stack AI data assistant on **Streamlit Cloud**

---

## 👨‍💻 Author  
**Varun Khera**  
📧 varunkhera.20@gmail.com
💼 https://www.linkedin.com/in/varun-khera/  
 

---

⭐ *If you found this project helpful, please star the repository!*  
Your support motivates further improvements and innovation.

