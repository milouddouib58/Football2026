# common/utils.py (النسخة المحدثة)

import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

# --- [إضافة جديدة] قاموس للأسماء العربية الشائعة ---
ARABIC_TEAM_NAMES_MAP = {
    "ريال مدريد": "real madrid",
    "برشلونة": "fc barcelona",
    "أتلتيكو مدريد": "atlético de madrid",
    "ليفربول": "liverpool fc",
    "مانشستر سيتي": "manchester city fc",
    "مانشستر يونايتد": "manchester united fc",
    "ارسنال": "arsenal fc",
    "أرسنال": "arsenal fc",
    "تشيلسي": "chelsea fc",
    "بايرن ميونخ": "fc bayern münchen",
    "بوروسيا دورتموند": "borussia dortmund",
    "يوفنتوس": "juventus",
    "ميلان": "ac milan",
    "انتر ميلان": "inter",
    "باريس سان جيرمان": "paris saint-germain",
}

# --- دوال موجودة سابقًا (تبقى كما هي) ---
def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [{level.upper()}] - {message}")

def parse_date_safe(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None

def parse_score(match: Dict) -> Tuple[Optional[int], Optional[int]]:
    score = match.get('score', {}).get('fullTime', {})
    hg = score.get('home')
    ag = score.get('away')
    if hg is not None and ag is not None:
        return int(hg), int(ag)
    return None, None

def poisson_pmf(k: int, lam: float) -> float:
    import math
    if lam < 0 or k < 0:
        return 0.0
    try:
        return (lam ** k * math.exp(-lam)) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0

# --- [إضافة جديدة] دالة البحث الذكي عن الفرق ---
def enhanced_team_search(
    query: str, teams_map: Dict[str, Dict], comp_code: str = None
) -> Optional[int]:
    """
    يبحث عن ID الفريق بناءً على اسم البحث.
    يدعم الأسماء العربية والبحث الجزئي والحساس لحالة الأحرف.
    """
    if not query or not teams_map:
        return None

    query_lower = query.strip().lower()

    # 1. التحقق من القاموس العربي
    if query_lower in ARABIC_TEAM_NAMES_MAP:
        query_lower = ARABIC_TEAM_NAMES_MAP[query_lower]

    # قائمة المرشحين المحتملين
    candidates = []
    for team_id, team_data in teams_map.items():
        # التأكد من أن الفريق يلعب في الدوري المطلوب (إذا تم تحديده)
        if comp_code and comp_code not in team_data.get("competitions", []):
            continue
        
        all_names = [n.lower() for n in team_data.get("names", []) if n]
        
        # 2. البحث عن تطابق كامل (case-insensitive)
        if query_lower in all_names:
            candidates.append((team_id, 100)) # أعلى أولوية
        
        # 3. البحث عن تطابق جزئي
        for name in all_names:
            if query_lower in name:
                candidates.append((team_id, 50)) # أولوية متوسطة

    if not candidates:
        return None

    # فرز المرشحين حسب الأولوية وإرجاع الأفضل
    candidates.sort(key=lambda x: x[1], reverse=True)
    return int(candidates[0][0])
