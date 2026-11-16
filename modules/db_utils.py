import sqlite3, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "incidents.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT, timestamp TEXT, level TEXT, risk_score REAL,
            category TEXT, behavior_conf REAL, scene_conf REAL,
            raw_path TEXT, blurred_path TEXT, false_alarm INTEGER DEFAULT 0)""")


def mark_false_alarm(incident_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE incidents SET false_alarm=1 WHERE id=?", (incident_id,))


def log_incident(meta, raw_path, blurred_path):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO incidents (camera_id,timestamp,level,risk_score,category,behavior_conf,scene_conf,raw_path,blurred_path) VALUES (?,?,?,?,?,?,?,?,?)",
            (meta["camera_id"], meta["timestamp"], meta["level"], meta["risk_score"],
             meta["category"], meta["behavior_conf"], meta["scene_conf"], raw_path, blurred_path))


def get_incidents(level=None, camera_id=None):
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []
    if level:
        query += " AND level=?"; params.append(level)
    if camera_id:
        query += " AND camera_id=?"; params.append(camera_id)
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(query + " ORDER BY id DESC", params).fetchall()


def get_recent_incidents(limit=10):
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def count_today_incidents():
    today = datetime.now().strftime("%Y%m%d")
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM incidents WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()[0]


def get_false_alarms():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT * FROM incidents WHERE false_alarm=1").fetchall()