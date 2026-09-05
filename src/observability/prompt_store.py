from typing import Optional, Dict
from src.observability.db import get_db_connection

class PromptStore:
    @staticmethod
    def get_active_prompt(prompt_type: str) -> Optional[Dict]:
        """Fetch the active prompt template by type."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT prompt_id, prompt_type, version, template_text, description FROM prompts WHERE prompt_type = ? AND is_active = 1 ORDER BY created_at DESC LIMIT 1",
                (prompt_type,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    @staticmethod
    def register_new_version(prompt_id: str, prompt_type: str, version: str, template_text: str, description: str = ""):
        """Register a new prompt version and set it as active, deactivating previous ones."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Deactivate older prompts of this type
            cursor.execute("UPDATE prompts SET is_active = 0 WHERE prompt_type = ?", (prompt_type,))
            # Insert new prompt
            cursor.execute(
                "INSERT INTO prompts (prompt_id, prompt_type, version, template_text, description, is_active) VALUES (?, ?, ?, ?, ?, 1)",
                (prompt_id, prompt_type, version, template_text, description)
            )
            conn.commit()
