import sqlite3

def init_db():
    conn = sqlite3.connect("news_analysis.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file TEXT,
        bias TEXT,
        summary TEXT,
        context TEXT,
        wiki_context TEXT
    )
    """)
    conn.commit()
    return conn

def save_result(conn, file, bias, summary, context, wiki_context):
    c = conn.cursor()
    c.execute("""
    INSERT INTO analysis (file, bias, summary, context, wiki_context)
    VALUES (?, ?, ?, ?, ?)
    """, (file, bias, summary, context, wiki_context))
    conn.commit()
