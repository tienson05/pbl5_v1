import psycopg2

def connect_database():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="palm_lockers",
            user="postgres",
            password="tienson",
            port="5432"
        )
        print("Kết nối database thành công!")
        return conn
    except Exception as e:
        print("Lỗi kết nối:", e)
        return None

# lấy tất cả locker
def get_all_locker(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM locker")
        return cursor.fetchall()

def get_available_locker(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM locker where status = 'available'")
        return cursor.fetchall()

# lấy tất cả session (embedding)
def get_all_sessions(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM session")
        return cursor.fetchall()

# lấy session đang active
def get_active_session(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM session WHERE status = %s", ("active",))
        return cursor.fetchall()

# thêm locker
def add_locker(conn, locker_id, location, status="available"):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO locker (id, location, status)
            VALUES (%s, %s, %s)
            """,
            (locker_id, location, status)
        )
    conn.commit()

# thêm session (lưu palm_hash)
def add_session(conn, session_id, locker_id, palm_hash="empty", status="active"):
    try:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE locker
                SET status = 'occupied'
                WHERE id = %s AND status = 'available'
                """,
                (locker_id,)
            )

            if cursor.rowcount == 0:
                raise Exception("Locker already occupied")

            cursor.execute(
                """
                INSERT INTO session (id, locker_id, palm_hash, start_time, status)
                VALUES (%s, %s, %s, NOW(), %s)
                """,
                (session_id, locker_id, palm_hash, status)
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(e)


def deactivate_active_sessions(conn, locker_id):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE session
            SET status = 'inactive', end_time = NOW()
            WHERE locker_id = %s AND status = 'active'
            """,
            (locker_id,)
        )

        cursor.execute(
            """
            UPDATE locker
            SET status = 'available'
            WHERE id = %s
            """,
            (locker_id,)
        )
    conn.commit()