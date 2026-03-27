import psycopg2

class Dao:
    def __init__(self):
        self.conn = None

    def connect_database(self):
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="palm_lockers",
                user="postgres",
                password="tienson",
                port="5432"
            )
            print("Kết nối database thành công!")
            self.conn = conn
        except Exception as e:
            print("Lỗi kết nối:", e)
            self.conn = None

    # lấy tất cả locker
    def get_all_locker(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM locker")
            return cursor.fetchall()

    def get_available_locker(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM locker where status = 'available'")
            return cursor.fetchall()

    # lấy tất cả session (embedding)
    def get_all_sessions(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM session")
            return cursor.fetchall()

    # lấy session đang active
    def get_active_session(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM session WHERE status = %s", ("active",))
            return cursor.fetchall()

    # thêm locker
    def add_locker(self, locker_id, location, status="available"):
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO locker (id, location, status)
                VALUES (%s, %s, %s)
                """,
                (locker_id, location, status)
            )
        self.conn.commit()

    # thêm session (lưu palm_hash)
    def add_session(self, session_id, locker_id, palm_hash="empty", status="active"):
        try:
            with self.conn.cursor() as cursor:

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

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            print(e)

    def deactivate_active_sessions(self, locker_id):
        with self.conn.cursor() as cursor:
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
        self.conn.commit()

dao = Dao() # instance singleton