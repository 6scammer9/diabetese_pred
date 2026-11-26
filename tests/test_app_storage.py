import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from simple_model import FEATURES


class AppStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.database_patch = patch.object(app, "DATABASE_PATH", self.database_path)
        self.database_patch.start()
        app.initialize_database()

    def tearDown(self):
        self.database_patch.stop()
        self.temp_dir.cleanup()

    def test_password_is_hashed_and_login_is_case_insensitive(self):
        user_id = app.create_user("Demo_User", "a long test password")

        self.assertEqual(
            app.authenticate_user("demo_user", "a long test password"),
            (user_id, "demo_user"),
        )
        self.assertIsNone(app.authenticate_user("demo_user", "wrong password"))
        with app.database_connection() as connection:
            stored_hash = connection.execute(
                "SELECT password_hash FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()["password_hash"]
        self.assertNotEqual(stored_hash, b"a long test password")

    def test_history_is_private_to_each_account_and_can_be_cleared(self):
        first_user = app.create_user("first_user", "first user password")
        second_user = app.create_user("second_user", "second user password")
        measurements = {name: 1.0 for name in FEATURES}
        app.save_prediction(first_user, measurements, 0.7, 1)
        app.save_prediction(second_user, measurements, 0.2, 0)

        self.assertEqual(len(app.get_prediction_history(first_user)), 1)
        self.assertEqual(len(app.get_prediction_history(second_user)), 1)
        self.assertEqual(app.delete_prediction_history(first_user), 1)
        self.assertEqual(app.get_prediction_history(first_user), [])
        self.assertEqual(len(app.get_prediction_history(second_user)), 1)

    def test_invalid_user_and_duplicate_username_are_rejected(self):
        with self.assertRaises(ValueError):
            app.create_user("x", "a long test password")
        app.create_user("demo_user", "a long test password")
        with self.assertRaises(ValueError):
            app.create_user("DEMO_USER", "another long password")


if __name__ == "__main__":
    unittest.main()
