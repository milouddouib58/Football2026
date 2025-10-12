# common/config.py (النسخة النهائية المدمجة والمصححة)

import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timezone # --- ✅ السطر الضروري الذي كان ناقصًا

# --- ⚙️ الإعدادات العامة والتكوين ---

# تحميل متغيرات البيئة من ملف .env
load_dotenv()
# --- ✅ تم تصحيح اسم المتغير ليتوافق مع بقية المشروع ---
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY") 

# إصدار المشروع
VERSION = "2.0.0"

# الشهر الذي تبدأ فيه المواسم الأوروبية
CURRENT_SEASON_START_MONTH = 7

# --- ✅ تم إضافة المنطقة الزمنية الناقصة ---
TZ = timezone.utc

# المسارات الأساسية للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# --- 🌐 إعدادات واجهة برمجة التطبيقات (API) ---
BASE_URL = "https://api.football-data.org/v4"
TIMEOUT = 30
MAX_RETRIES = 5
# --- ✅ تمت إضافة فاصل زمني بين الطلبات لتجنب الأخطاء ---
MIN_INTERVAL_SEC = 6.2

# --- ⚽ الدوريات المستهدفة (من ملفك الأصلي) ---
TARGET_COMPETITIONS = [
    'PL',   # الدوري الإنجليزي الممتاز
    'BL1',  # الدوري الألماني (بوندسليجا)
    'SA',   # الدوري الإيطالي (سيريا أي)
    'PD',   # الدوري الإسباني (لا ليجا)
    'FL1',  # الدوري الفرنسي (ليج 1)
    'PPL'   # الدوري البرتغالي
]
# --- ✅ تمت إضافة قائمة الأولوية للمساعدة في البحث عن الفرق ---
COMPETITION_PRIORITY = ["CL", "PL", "PD", "SA", "BL1", "FL1", "PPL"]

# --- 🧠 إعدادات النمذجة الإحصائية (من ملفك الأصلي) ---
HALF_LIFE_DAYS = 120
PRIOR_GAMES = 10
DC_RHO_MAX = 0.25

# --- ✨ إعدادات نموذج ELO (من ملفك الأصلي) ---
ELO_K = 20.0
ELO_HFA = 65.0
ELO_LAMBDA_SCALE = 800.0

# --- ✅ تمت إعادة هيكلة الكود كـ Class لضمان التوافق ---
class Config:
    def __init__(self):
        self.VERSION = VERSION
        self.BASE_DIR = BASE_DIR
        self.DATA_DIR = DATA_DIR
        self.MODELS_DIR = MODELS_DIR
        self.API_KEY = API_KEY
        self.BASE_URL = BASE_URL
        self.MAX_RETRIES = MAX_RETRIES
        self.TIMEOUT = TIMEOUT
        self.MIN_INTERVAL_SEC = MIN_INTERVAL_SEC
        self.PRIOR_GAMES = PRIOR_GAMES
        self.HALF_LIFE_DAYS = HALF_LIFE_DAYS
        self.DC_RHO_MAX = DC_RHO_MAX
        self.ELO_K = ELO_K
        self.ELO_HFA = ELO_HFA
        self.ELO_LAMBDA_SCALE = ELO_LAMBDA_SCALE
        self.TARGET_COMPETITIONS = TARGET_COMPETITIONS
        self.COMPETITION_PRIORITY = COMPETITION_PRIORITY
        self.CURRENT_SEASON_START_MONTH = CURRENT_SEASON_START_MONTH
        self.TZ = TZ

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not self.API_KEY:
            raise ValueError("API Key not found. Please set FOOTBALL_DATA_API_KEY in your .env file.")

config = Config()
