# import sqlite3

# def connect_db():
#     conn = sqlite3.connect("database/users.db")
#     c = conn.cursor()
#     return conn, c

# def create_usertable():
#     conn, c = connect_db()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS usertable (
#             username TEXT PRIMARY KEY,
#             password TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# def add_userdata(username, password):
#     conn, c = connect_db()
#     c.execute('INSERT INTO usertable (username, password) VALUES (?, ?)', (username, password))
#     conn.commit()
#     conn.close()

# def login_user(username, password):
#     conn, c = connect_db()
#     c.execute('SELECT * FROM usertable WHERE username = ? AND password = ?', (username, password))
#     data = c.fetchone()
#     conn.close()
#     return data










# db.py
import sqlite3
import hashlib
from pathlib import Path
import os

# Database configuration
DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "users.db")

def get_db():
    """Ensure database directory exists and return connection"""
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize the database with tables"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        
        # Add default admin if not exists
        cursor.execute("SELECT username FROM users WHERE username='admin'")
        if not cursor.fetchone():
            hashed_pw = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute("INSERT INTO users VALUES (?, ?)", ("admin", hashed_pw))
        
        conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        conn.close()

# Initialize database when module loads
init_db()

def add_user(username, password):
    """Add new user to database"""
    try:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        result = cursor.fetchone()
        
        if result:
            return hashlib.sha256(password.encode()).hexdigest() == result[0]
        return False
    finally:
        conn.close()