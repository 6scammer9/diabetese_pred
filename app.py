import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import joblib
import pandas as pd
import streamlit as st

from simple_model import FEATURES


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts" / "simple_model" / "logistic_model.joblib"
DATABASE_PATH = Path(
    os.environ.get("APP_DB_PATH", BASE_DIR / "data" / "diabetes.sqlite3")
)
PASSWORD_SCRYPT_N = 2**14
PASSWORD_SCRYPT_R = 8
PASSWORD_SCRYPT_P = 1
PASSWORD_HASH_LENGTH = 32


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with database_connection() as connection:
        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    measurements_json TEXT NOT NULL,
                    positive_score REAL NOT NULL CHECK (positive_score >= 0 AND positive_score <= 1),
                    predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1))
                );
                """
            )


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    password_salt = salt or secrets.token_bytes(16)
    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=password_salt,
        n=PASSWORD_SCRYPT_N,
        r=PASSWORD_SCRYPT_R,
        p=PASSWORD_SCRYPT_P,
        dklen=PASSWORD_HASH_LENGTH,
    )
    return password_salt, password_hash


def create_user(username: str, password: str) -> int:
    normalized_username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", normalized_username):
        raise ValueError("Username must be 3-32 letters, numbers, dots, underscores, or hyphens.")
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")

    salt, password_hash = hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with database_connection() as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_salt, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_username, salt, password_hash, created_at),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError:
        raise ValueError("That username is already registered.") from None


def authenticate_user(username: str, password: str) -> tuple[int, str] | None:
    normalized_username = username.strip().lower()
    with database_connection() as connection:
        row = connection.execute(
            "SELECT user_id, username, password_salt, password_hash FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
    if row is None:
        return None

    _, supplied_hash = hash_password(password, row["password_salt"])
    if not hmac.compare_digest(supplied_hash, row["password_hash"]):
        return None
    return int(row["user_id"]), str(row["username"])


def save_prediction(
    user_id: int,
    measurements: dict[str, float],
    positive_score: float,
    predicted_class: int,
) -> None:
    if set(measurements) != set(FEATURES):
        raise ValueError("Prediction measurements do not match the model features.")
    if not 0 <= positive_score <= 1 or predicted_class not in (0, 1):
        raise ValueError("Prediction score or class is invalid.")

    stored_measurements = {name: float(measurements[name]) for name in FEATURES}
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database_connection() as connection, connection:
        connection.execute(
            """
            INSERT INTO predictions (
                user_id, created_at, measurements_json, positive_score, predicted_class
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                created_at,
                json.dumps(stored_measurements),
                positive_score,
                predicted_class,
            ),
        )


def get_prediction_history(user_id: int) -> list[dict[str, object]]:
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT created_at, measurements_json, positive_score, predicted_class
            FROM predictions
            WHERE user_id = ?
            ORDER BY prediction_id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_prediction_history(user_id: int) -> int:
    with database_connection() as connection, connection:
        cursor = connection.execute(
            "DELETE FROM predictions WHERE user_id = ?", (user_id,)
        )
        return cursor.rowcount


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first with `python train_model.py`."
        )
    return joblib.load(MODEL_PATH)


def show_authentication() -> None:
    mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
    if mode == "Sign in":
        with st.form("sign_in_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            user = authenticate_user(username, password)
            if user is None:
                st.error("Username or password is incorrect.")
            else:
                st.session_state["user_id"], st.session_state["username"] = user
                st.rerun()
        return

    with st.form("create_account_form"):
        username = st.text_input("Username")
        password = st.text_input("Password (at least 12 characters)", type="password")
        confirmation = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")
    if not submitted:
        return
    if password != confirmation:
        st.error("Passwords do not match.")
        return
    try:
        user_id = create_user(username, password)
    except ValueError as error:
        st.error(str(error))
        return
    st.session_state["user_id"] = user_id
    st.session_state["username"] = username.strip().lower()
    st.rerun()


def show_prediction_form(model, user_id: int) -> None:
    st.subheader("Enter measurements")
    st.caption(
        "Use the same units as the training data. Enter 0 for an unavailable "
        "glucose, blood pressure, skin thickness, insulin, or BMI value."
    )
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20, value=None, step=1
            )
            glucose = st.number_input(
                "Glucose (mg/dL)", min_value=0, max_value=300, value=None, step=1
            )
            blood_pressure = st.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=0,
                max_value=200,
                value=None,
                step=1,
            )
            skin_thickness = st.number_input(
                "Skin thickness (mm)", min_value=0, max_value=100, value=None, step=1
            )
        with right:
            insulin = st.number_input(
                "Insulin (mu U/mL)", min_value=0, max_value=1000, value=None, step=1
            )
            bmi = st.number_input(
                "BMI (kg/m2)", min_value=0.0, max_value=80.0, value=None, step=0.1
            )
            pedigree = st.number_input(
                "Diabetes pedigree function",
                min_value=0.0,
                max_value=5.0,
                value=None,
                step=0.01,
            )
            age = st.number_input(
                "Age (years)", min_value=18, max_value=120, value=None, step=1
            )
        submitted = st.form_submit_button("Run experimental prediction")

    if not submitted:
        return

    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }
    if any(value is None for value in values.values()):
        st.error("Enter every field before running a prediction.")
        return

    features = pd.DataFrame([values], columns=FEATURES)
    positive_score = float(model.predict_proba(features)[0, 1])
    predicted_class = int(positive_score >= 0.5)
    save_prediction(user_id, values, positive_score, predicted_class)

    st.metric("Model score for Outcome = 1", f"{positive_score:.1%}")
    if predicted_class:
        st.warning("The model selected class 1. This is not a diagnosis.")
    else:
        st.info("The model selected class 0. This does not rule out diabetes.")


def show_prediction_history(user_id: int) -> None:
    st.subheader("Your local prediction history")
    history = get_prediction_history(user_id)
    if not history:
        st.caption("No predictions saved yet.")
        return

    rows = []
    for prediction in history:
        measurements = json.loads(str(prediction["measurements_json"]))
        rows.append(
            {
                "Recorded (UTC)": prediction["created_at"],
                **measurements,
                "Model score (Outcome = 1)": prediction["positive_score"],
                "Selected class": prediction["predicted_class"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if st.button("Delete my prediction history"):
        st.session_state["confirm_history_delete"] = True
    if st.session_state.get("confirm_history_delete"):
        st.warning("This permanently deletes your saved predictions for this account.")
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm delete"):
            delete_prediction_history(user_id)
            st.session_state["confirm_history_delete"] = False
            st.rerun()
        if cancel.button("Cancel"):
            st.session_state["confirm_history_delete"] = False
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Diabetes prediction demo", page_icon=None)
    initialize_database()
    st.title("Diabetes prediction demo")
    st.warning(
        "Educational prototype only. This model is not clinically validated. "
        "Do not use it to diagnose, treat, or make care decisions, and do not "
        "enter identifiable or real patient information."
    )

    if "user_id" not in st.session_state:
        show_authentication()
        st.stop()

    st.sidebar.write(f"Signed in as **{st.session_state['username']}**")
    if st.sidebar.button("Sign out"):
        del st.session_state["user_id"]
        del st.session_state["username"]
        st.rerun()

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    show_prediction_form(model, int(st.session_state["user_id"]))
    st.divider()
    show_prediction_history(int(st.session_state["user_id"]))


if __name__ == "__main__":
    main()
