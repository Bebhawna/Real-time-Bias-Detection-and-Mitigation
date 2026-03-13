# db_config.py

import os
import json
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, Json, execute_values
from psycopg2 import pool

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "db.drbvdbpfgqhbnsryrmjp.supabase.co"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DBNAME", "postgres"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "bias_monitoring_mp"),
}
 

connection_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    dbname=DB_CONFIG["dbname"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    sslmode="require"               #new commit here
)


DB_SCHEMA = "fairness_system"

def get_connection():
    """
    Get a connection from the pool.
    """
    return connection_pool.getconn()


def release_connection(conn):
    """
    Return connection to the pool.
    """
    connection_pool.putconn(conn)

def insert_record(
    record: Dict[str, Any],
    conn: Optional[psycopg2.extensions.connection] = None,

) -> int:
    gender = record["gender"]
    race = record["race"]
    features = record["features"]
    prediction = record["prediction"]
    """
    Insert a single record into the fairness_system.predictions_log table.

    Args:
        gender (str): Gender of person ('Male', 'Female', 'Other').
        race (str): Race group (free text).
        features (dict): JSON-serializable dict containing input features.
        prediction (int): Model's output (0 or 1).
        conn (connection, optional): Existing DB connection. If None,
                                     a new connection is created and closed.

    Returns:
        int: The ID of the inserted record.

    Raises:
        psycopg2.Error: If any database error occurs.
    """
    owns_connection = False
    if conn is None:
        conn = get_connection()
        owns_connection = True

    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO fairness_system.predictions_log (gender, race, features, prediction)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """
            cur.execute(
                query,
                (
                    gender,
                    race,
                    Json(features), 
                    prediction,
                ),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
        return new_id
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            release_connection(conn)



def insert_multiple_records(
    records: List[Dict[str, Any]],
    conn: Optional[psycopg2.extensions.connection] = None,
) -> List[int]:
    """
    Insert multiple records into predictions_log in a single query.

    Args:
        records (List[Dict]): Each dict must have keys:
            - "gender": str
            - "race": str
            - "features": dict
            - "prediction": int
        conn (optional): Existing DB connection. If None, a new one is created.

    Returns:
        List[int]: List of IDs for the inserted records, in the same order.
    """
    if not records:
        return []

    owns_connection = False
    if conn is None:
        conn = get_connection()
        owns_connection = True

    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO fairness_system.predictions_log (gender, race, features, prediction)
                VALUES %s
                RETURNING id;
            """

            values = [
                (
                    rec["gender"],
                    rec["race"],
                    Json(rec["features"]),
                    rec["prediction"],
                )
                for rec in records
            ]

            execute_values(cur, query, values)
            ids = [row[0] for row in cur.fetchall()]
            conn.commit()
        return ids
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            release_connection(conn)



def fetch_latest_records(
    n: int,
    conn: Optional[psycopg2.extensions.connection] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the latest N records from predictions_log based on timestamp.
    Useful for implementing a sliding window.

    Args:
        n (int): Number of latest records to fetch.
        conn (connection, optional): Existing DB connection. If None,
                                     a new connection is created and closed.

    Returns:
        List[Dict[str, Any]]: A list of records as dictionaries.

    Raises:
        psycopg2.Error: If any database error occurs.
    """
    owns_connection = False
    if conn is None:
        conn = get_connection()
        owns_connection = True

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT id, gender, race, features, prediction, timestamp
                FROM fairness_system.predictions_log
                ORDER BY timestamp DESC
                LIMIT %s;
            """
            cur.execute(query, (n,))
            rows = cur.fetchall()
        return list(rows)
    finally:
        if owns_connection:
            release_connection(conn)


def insert_final_record(
    raw_id: int,
    gender: str,
    race: str,
    features: Dict[str, Any],
    prediction: int,
    mitigation_applied: bool = False,
    conn: Optional[psycopg2.extensions.connection] = None,
) -> None:
    """
    Insert a cleaned or corrected record into final_predictions_log.

    Args:
        raw_id (int): ID of the RAW record from predictions_log.
        gender (str): Gender.
        race (str): Race group.
        features (dict): JSON feature set.
        prediction (int): Model prediction.
        mitigation_applied (bool): Whether bias mitigation was applied.
    """

    owns_connection = False
    if conn is None:
        conn = get_connection()
        owns_connection = True

    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO fairness_system.final_predictions_log
                (raw_id, gender, race, features, prediction, mitigation_applied)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (raw_id) DO NOTHING;
            """

            cur.execute(
                query,
                (
                    raw_id,
                    gender,
                    race,
                    Json(features),
                    prediction,
                    mitigation_applied,
                ),
            )

            conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        if owns_connection:
            release_connection(conn)


def fetch_final_records(
    n: int,
    conn: Optional[psycopg2.extensions.connection] = None,
) -> List[Dict[str, Any]]:
    
    owns_connection = False
    if conn is None:
        conn = get_connection()
        owns_connection = True

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            query = """
                SELECT *
                FROM fairness_system.final_predictions_log
                ORDER BY timestamp DESC
                LIMIT %s;
            """

            cur.execute(query, (n,))
            rows = cur.fetchall()

        return list(rows)

    finally:
        if owns_connection:
            release_connection(conn)


if __name__ == "__main__":
    # Example usage (uncomment to test, after updating DB_CONFIG):
    
    # Sample input/retrival test
    
    # conn = get_connection()
    # new_id = insert_record(
    #     gender="Male",
    #     race="Group A",
    #     features={"age": 30, "income": 50000},
    #     prediction=1,
    #     conn=conn,
    # )
    # print("Inserted ID:", new_id)
    
    # latest = fetch_latest_records(5, conn=conn)
    # print("Latest records:", latest)
    
    # release_connection(conn)
    
    # Example usage for multiple insert:
    # test_records = [
    #     {
    #         "gender": "Male",
    #         "race": "Group A",
    #         "features": {"age": 30, "income": 50000},
    #         "prediction": 1,
    #     },
    #     {
    #         "gender": "Other",
    #         "race": "Group B",
    #         "features": {"age": 25, "income": 40000},
    #         "prediction": 0,
    #     }
    # ]
    # print(insert_multiple_records(test_records))
    pass
