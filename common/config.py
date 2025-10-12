# common/config.py (النسخة المصححة)
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import timezone # --- بداية الإضافة ---

load_dotenv()

class Config:
    VERSION = "v2.1.0"
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"

    # API
    API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    BASE_URL = "https://api.football-data.org/v4"
    MAX_RETRIES = 5
    TIMEOUT = 25
    MIN_INTERVAL_SEC = 6.2

    # Modeling
    PRIOR_GAMES = 12
    HALF_LIFE_DAYS = 270
    DC_RHO_MAX = 0.3

    # ELO
    ELO_K = 20.0
    ELO_HFA = 60.0
    ELO_LAMBDA_SCALE = 1200.0

    # Competitions
    TARGET_COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1", "CL", "DED", "PPL", "BSA"]
    COMPETITION_PRIORITY = ["CL", "PD", "PL", "SA", "BL1", "FL1", "DED", "PPL", "BSA"]

    # Seasons & Time
    CURRENT_SEASON_START_MONTH = 7  # يوليو
    TZ = timezone.utc # --- نهاية الإضافة: تحديد المنطقة الزمنية القياسية ---

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.API_KEY:
            raise ValueError("API Key not found. Please set FOOTBALL_DATA_API_KEY in your .env file.")

config = Config()
