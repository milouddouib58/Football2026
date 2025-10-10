# common/utils.py
import re
import math
import difflib
from datetime import datetime
from typing import Optional, Tuple, Dict, List

from .config import config

ARABIC_SYNONYMS = {
    "ريال مدريد": "Real Madrid", "برشلونة": "Barcelona", "برشلونه": "Barcelona",
    "اتلتيكو مدريد": "Atletico Madrid", "أتلتيكو مدريد": "Atletico Madrid",
    "إشبيلية": "Sevilla", "اشبيلية": "Sevilla", "مانشستر سيتي": "Manchester City",
    "مان سيتي": "Manchester City", "مانشستر يونايتد": "Manchester United",
    "ليفربول": "Liverpool", "تشيلسي": "Chelsea", "توتنهام": "Tottenham Hotspur",
    "أرسنال": "Arsenal", "بايرن ميونخ": "Bayern Munich", "بايرن ميونيخ": "Bayern Munich",
    "بوروسيا دورتموند": "Borussia Dortmund", "باريس سان جيرمان": "Paris Saint-Germain",
}

def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def parse_date_safe(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

def parse_score(match: Dict) -> Tuple[Optional[int], Optional[int]]:
    score = match.get("score", {}).get("fullTime", {})
    return score.get("home"), score.get("away")

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
    except (ValueError, OverflowError):
        return 0.0

def _norm_ascii(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _remove_ar_diacritics(s: str) -> str:
    # إزالة التشكيل إن وجد
    return re.sub(r'[\u064B-\u0652]', '', s)

def transliterate_ar_to_en(s: str) -> str:
    ar_to_latin = {
        "ا":"a","أ":"a","إ":"i","آ":"a","ؤ":"u","ئ":"i","ب":"b","ت":"t","ث":"th",
        "ج":"j","ح":"h","خ":"kh","د":"d","ذ":"dh","ر":"r","ز":"z","س":"s","ش":"sh",
        "ص":"s","ض":"d","ط":"t","ظ":"z","ع":"a","غ":"gh","ف":"f","ق":"q","ك":"k",
        "ل":"l","م":"m","ن":"n","ه":"h","و":"w","ي":"y","ى":"y","ة":"a",
    }
    s = _remove_ar_diacritics(s or "")
    s_norm = re.sub(r'[^\u0600-\u06FF\s]', '', s)
    out = "".join([ar_to_latin.get(ch, ch) for ch in s_norm])
    return _norm_ascii(out)

def enhanced_team_search(team_name: str, teams_map: Dict, prefer_comp: Optional[str] = None) -> Optional[int]:
    clean_name = (team_name or "").strip()
    if len(clean_name) < 2:
        return None

    # مرادفات عربية مباشرة
    ar_key = clean_name.replace("ى", "ي").replace("ة", "ه")
    if ar_key in ARABIC_SYNONYMS:
        clean_name = ARABIC_SYNONYMS[ar_key]

    best_score, best_id = 0.0, None
    tname_norm = clean_name.lower()
    tname_trans = transliterate_ar_to_en(clean_name)

    teams_to_search: List[Dict] = list(teams_map.values())
    if prefer_comp and prefer_comp in config.COMPETITION_PRIORITY:
        preferred = [t for t in teams_to_search if prefer_comp in t.get('competitions', [])]
        others = [t for t in teams_to_search if prefer_comp not in t.get('competitions', [])]
        teams_to_search = preferred + others

    for team_data in teams_to_search:
        tid = team_data.get('id')
        all_names = [n for n in team_data.get('names', []) if n]

        for name in all_names:
            score = difflib.SequenceMatcher(None, tname_norm, (name or "").lower()).ratio()
            if score > best_score:
                best_score, best_id = score, tid

        if tname_trans:
            for name in all_names:
                score = difflib.SequenceMatcher(None, tname_trans, _norm_ascii(name or "")).ratio()
                if score > best_score:
                    best_score, best_id = score, tid

    return best_id if best_score > 0.65 else None
