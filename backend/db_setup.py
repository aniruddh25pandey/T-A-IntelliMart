import sqlite3
from datetime import datetime

def create_database():
    conn = sqlite3.connect('reviews.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            username TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review_text TEXT NOT NULL,
            review_date TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database created successfully!")

if __name__ == '__main__':
    create_database()