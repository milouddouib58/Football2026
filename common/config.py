import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timezone # --- ✅ تم إضافة هذا السطر الضروري

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

# --- 🧠 إعدادات النمذجة الإحصائية (من ملفك الأصلي + الإضافات) ---
HALF_LIFE_DAYS = 120
PRIOR_GAMES = 10
DC_RHO_MAX = 0.2 # <-- تم التعديل من 0.25 حسب الطلب

# --- الإضافات المطلوبة ---
TEAM_FACTORS_HALFLIFE_DAYS = 180        # نصف عمر الانحلال الزمني لعوامل الفرق
TEAM_FACTORS_PRIOR_GLOBAL = 3.0         # قوة انكماش جاما نحو 1.0
TEAM_FACTORS_TEAM_PRIOR_WEIGHT = 5.0    # وزن الانكماش نحو عوامل الموسم السابق (إن وجدت)

DC_RHO_MIN = -0.2
DC_RHO_STEP = 0.001

POISSON_TAIL_EPS = 1e-7                 # حساسية قصّ ذيل بواسون
POISSON_MAX_GOALS_CAP = 16              # حد أقصى صارم للقصّ (أمان)
# --- نهاية الإضافات ---

# --- ✨ إعدادات نموذج ELO (من ملفك الأصلي + الإضافات) ---
ELO_K = 20.0
ELO_HFA = 60.0 # <-- تم التعديل من 65.0 حسب الطلب
ELO_LAMBDA_SCALE = 800.0

# --- الإضافات المطلوبة ---
ELO_START = 1500.0
ELO_K_BASE = 24.0
ELO_SCALE = 400.0
ELO_HALFLIFE_DAYS = 365
# --- نهاية الإضافات ---


# --- ✅ تم إعادة هيكلة الكود كـ Class لضمان التوافق ---
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
        self.DC_RHO_MAX = DC_RHO_MAX # القيمة المحدثة
        self.ELO_K = ELO_K
        self.ELO_HFA = ELO_HFA # القيمة المحدثة
        self.ELO_LAMBDA_SCALE = ELO_LAMBDA_SCALE
        self.TARGET_COMPETITIONS = TARGET_COMPETITIONS
        self.COMPETITION_PRIORITY = COMPETITION_PRIORITY
        self.CURRENT_SEASON_START_MONTH = CURRENT_SEASON_START_MONTH
        self.TZ = TZ
        
        # إضافة المتغيرات الجديدة إلى الكلاس
        self.TEAM_FACTORS_HALFLIFE_DAYS = TEAM_FACTORS_HALFLIFE_DAYS
        self.TEAM_FACTORS_PRIOR_GLOBAL = TEAM_FACTORS_PRIOR_GLOBAL
        self.TEAM_FACTORS_TEAM_PRIOR_WEIGHT = TEAM_FACTORS_TEAM_PRIOR_WEIGHT
        self.DC_RHO_MIN = DC_RHO_MIN
        self.DC_RHO_STEP = DC_RHO_STEP
        self.POISSON_TAIL_EPS = POISSON_TAIL_EPS
        self.POISSON_MAX_GOALS_CAP = POISSON_MAX_GOALS_CAP
        self.ELO_START = ELO_START
        self.ELO_K_BASE = ELO_K_BASE
        self.ELO_SCALE = ELO_SCALE
        self.ELO_HALFLIFE_DAYS = ELO_HALFLIFE_DAYS

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not self.API_KEY:
            raise ValueError("API Key not found. Please set FOOTBALL_DATA_API_KEY in your .env file.")

config = Config()

