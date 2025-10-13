import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timezone

# --- ⚙️ الإعدادات العامة والتكوين ---

# تحميل متغيرات البيئة من ملف .env
load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

# إصدار المشروع
VERSION = "2.0.0"

# الشهر الذي تبدأ فيه المواسم الأوروبية (مهم لحل مشكلة تجميع المواسم)
CURRENT_SEASON_START_MONTH = 7

# المنطقة الزمنية الموحدة
TZ = timezone.utc

# المسارات الأساسية للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# --- 🌐 إعدادات واجهة برمجة التطبيقات (API) ---
BASE_URL = "https://api.football-data.org/v4"
TIMEOUT = 30
MAX_RETRIES = 5
MIN_INTERVAL_SEC = 6.2

# --- ⚽ الدوريات المستهدفة ---
TARGET_COMPETITIONS = [
    'PL', 'BL1', 'SA', 'PD', 'FL1', 'PPL'
]
COMPETITION_PRIORITY = ["CL", "PL", "PD", "SA", "BL1", "FL1", "PPL"]

# --- 🧠 إعدادات النمذجة الإحصائية ---

# ✅ --- الحل الرئيسي: شبكة البحث عن أفضل المعاملات (Hyperparameter Grid Search) ---
# بدلاً من استخدام قيم ثابتة، نعرّف هنا قائمة من الخيارات ليتم اختبارها.
# سيقوم سكريبت التدريب باختيار أفضل تركيبة لكل موسم.
HYPERPARAM_GRID = {
    "TEAM_FACTORS_HALFLIFE_DAYS": [90, 180, 365],
    "TEAM_FACTORS_PRIOR_GLOBAL": [2.0, 3.0, 5.0],
    "TEAM_FACTORS_TEAM_PRIOR_WEIGHT": [0.0, 5.0],
    "TEAM_FACTORS_DAMPING": [0.3, 0.5, 0.7], # ✅ تمت إضافة معامل التخميد هنا
    "DC_RHO_MAX": [0.15, 0.20],
}
# --- نهاية شبكة البحث ---


# --- المعاملات الأخرى التي لا تتغير في شبكة البحث ---
DC_RHO_MIN = -0.2
DC_RHO_STEP = 0.001
POISSON_TAIL_EPS = 1e-7
POISSON_MAX_GOALS_CAP = 16

# --- إعدادات نموذج ELO (يمكن إضافتها لشبكة البحث مستقبلاً إذا لزم الأمر) ---
ELO_START = 1500.0
ELO_K_BASE = 24.0
ELO_HFA = 60.0
ELO_SCALE = 400.0
ELO_HALFLIFE_DAYS = 365

# --- 🔴 تم حذف الإعدادات القديمة الثابتة من هنا لتجنب التكرار والارتباك ---
# TEAM_FACTORS_HALFLIFE_DAYS = 180 (تم نقله إلى HYPERPARAM_GRID)
# TEAM_FACTORS_PRIOR_GLOBAL = 3.0 (تم نقله إلى HYPERPARAM_GRID)
# TEAM_FACTORS_TEAM_PRIOR_WEIGHT = 5.0 (تم نقله إلى HYPERPARAM_GRID)


# --- فئة التكوين الرئيسية (Config Class) ---
class Config:
    def __init__(self):
        # --- الإعدادات العامة ---
        self.VERSION = VERSION
        self.BASE_DIR = BASE_DIR
        self.DATA_DIR = DATA_DIR
        self.MODELS_DIR = MODELS_DIR
        self.CURRENT_SEASON_START_MONTH = CURRENT_SEASON_START_MONTH
        self.TZ = TZ

        # --- إعدادات API ---
        self.API_KEY = API_KEY
        self.BASE_URL = BASE_URL
        self.MAX_RETRIES = MAX_RETRIES
        self.TIMEOUT = TIMEOUT
        self.MIN_INTERVAL_SEC = MIN_INTERVAL_SEC

        # --- إعدادات الدوريات ---
        self.TARGET_COMPETITIONS = TARGET_COMPETITIONS
        self.COMPETITION_PRIORITY = COMPETITION_PRIORITY

        # --- إعدادات النمذجة ---
        # ✅ تم استبدال القيم المفردة بمرجع إلى شبكة البحث
        self.HYPERPARAM_GRID = HYPERPARAM_GRID 
        
        self.DC_RHO_MIN = DC_RHO_MIN
        self.DC_RHO_STEP = DC_RHO_STEP
        self.POISSON_TAIL_EPS = POISSON_TAIL_EPS
        self.POISSON_MAX_GOALS_CAP = POISSON_MAX_GOALS_CAP

        # --- إعدادات ELO ---
        self.ELO_START = ELO_START
        self.ELO_K_BASE = ELO_K_BASE
        self.ELO_HFA = ELO_HFA
        self.ELO_SCALE = ELO_SCALE
        self.ELO_HALFLIFE_DAYS = ELO_HALFLIFE_DAYS

        # --- التأكد من وجود المجلدات والمفتاح ---
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not self.API_KEY:
            raise ValueError("API Key not found. Please set FOOTBALL_DATA_API_KEY in your .env file.")

config = Config()
