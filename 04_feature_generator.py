# 04_feature_generator.py
# -----------------------------------------------------------------------------
# الوصف:
# يقوم هذا السكريبت بإنشاء مجموعة بيانات التدريب (Features) لنموذج تعلم الآلة.
#
# لكل مباراة منتهية، يتم استخراج الميزات التالية:
# - عوامل الهجوم والدفاع (Team Factors) من النموذج الإحصائي
# - تقييمات ELO وفرق ELO
# - فورمة الفريق (متوسط النقاط في آخر N مباراة قبل المباراة الحالية)
# - النتيجة الفعلية كهدف (1 = فوز المضيف، 0 = تعادل، -1 = فوز الضيف)
#
# ملاحظة مهمة حول تسرّب المعلومات (Data Leakage):
# الاعتماد على Team Factors / Elo المحسوبة على مستوى الموسم ككل
# يعني أن الميزات تحتوي ضمنياً على معلومات من مباريات مستقبلية.
# هذا مقبول كتقريب أولي، لكن يُستحسن مستقبلاً حساب هذه الميزات
# "زمنياً" (incrementally) بحيث تعكس القيم المتاحة فقط قبل كل مباراة.
#
# التحسينات:
# - التحقق من صحة البيانات قبل إنشاء الميزات
# - إضافة ميزات إضافية (متوسطات الدوري، عدد المباريات في الفورمة)
# - حماية من القيم المفقودة واللانهائية
# - إنشاء نسخة احتياطية قبل الكتابة فوق الملفات
# - حفظ بيانات وصفية (metadata) مع مجموعة البيانات
# - تقرير إحصائي مفصّل بعد الإنشاء
# - دعم معاملات سطر الأوامر
# - شريط تقدم وتسجيل أوضح
# - حفظ آمن عبر ملف مؤقت
# - دعم وضع التشغيل الجاف (--dry-run)
# - إصلاح مشكلة مقارنة التواريخ (naive vs aware)
#
# الاستخدام:
#   python 04_feature_generator.py
#   python 04_feature_generator.py --form-matches 10
#   python 04_feature_generator.py --dry-run
#   python 04_feature_generator.py --no-backup
# -----------------------------------------------------------------------------

import sys
import os
import json
import shutil
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# مكتبات بيانات
try:
    import pandas as pd
except ImportError:
    print("خطأ: مكتبة pandas غير مثبتة. يرجى تثبيتها: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None  # اختيارية — تُستخدم فقط للتحقق من القيم

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe, parse_score

# استيراد دالة حساب الفورمة
try:
    from common.modeling import calculate_team_form as _calculate_team_form_base
except ImportError:
    _calculate_team_form_base = None


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# عدد المباريات الافتراضي لحساب الفورمة
DEFAULT_FORM_MATCHES = 5

# الحد الأدنى لعدد المباريات المطلوب لإنتاج مجموعة بيانات مفيدة
MIN_FEATURES_REQUIRED = 10

# اسم ملف النسخة الاحتياطية
BACKUP_SUFFIX = ".backup"

# عدد النسخ الاحتياطية المحتفظ بها
MAX_BACKUPS = 3

# قائمة الميزات المُنتجة (بدون الأعمدة الوصفية والهدف)
FEATURE_COLUMNS = [
    "home_attack",
    "away_attack",
    "home_defense",
    "away_defense",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_avg_points",
    "away_avg_points",
]

# أعمدة وصفية (metadata) تُحفظ في CSV لكن لا تُستخدم كميزات تدريب
METADATA_COLUMNS = [
    "match_id",
    "match_date",
    "season_key",
    "competition_code",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
]

# أعمدة الهدف
TARGET_COLUMNS = [
    "actual_home_goals",
    "actual_away_goals",
    "result",
]


# -----------------------------------------------------------------------------
# القسم الأول: توحيد التواريخ
# -----------------------------------------------------------------------------

def to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    تحويل أي كائن datetime إلى naive-UTC لمنع أخطاء المقارنة.

    المعاملات:
        dt: كائن datetime (قد يكون aware أو naive أو None)

    العائد:
        datetime بدون معلومات منطقة زمنية (naive-UTC)، أو None
    """
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return dt


def calculate_team_form(
    all_matches: List[Dict],
    team_id: int,
    ref_date: datetime,
    num_matches: int = DEFAULT_FORM_MATCHES,
) -> Dict[str, Any]:
    """
    حساب فورمة الفريق (آخر N مباراة قبل تاريخ معيّن).

    يتم توحيد التواريخ إلى naive-UTC قبل المقارنة لمنع أخطاء
    مقارنة aware vs naive.

    إذا كانت الدالة الأصلية من common.modeling متوفرة، يتم استخدامها.
    وإلا يتم استخدام بديل بسيط.

    المعاملات:
        all_matches: جميع المباريات
        team_id: معرّف الفريق
        ref_date: التاريخ المرجعي (لا تُحسب المباريات بعده)
        num_matches: عدد المباريات الأخيرة لحساب المتوسط

    العائد:
        قاموس يحتوي على:
        - avg_points: متوسط النقاط
        - matches_found: عدد المباريات التي تم إيجادها
    """
    # توحيد التاريخ المرجعي
    ref_date = to_naive_utc(ref_date)

    if ref_date is None:
        return {"avg_points": 1.0, "matches_found": 0}

    # محاولة استخدام الدالة الأصلية
    if _calculate_team_form_base is not None:
        try:
            result = _calculate_team_form_base(
                all_matches, team_id, ref_date, num_matches=num_matches
            )
            # التأكد من أن النتيجة قاموس
            if isinstance(result, dict):
                if "matches_found" not in result:
                    result["matches_found"] = num_matches
                return result
        except Exception:
            pass

    # بديل بسيط في حالة عدم توفر الدالة الأصلية
    rows = []

    for m in all_matches:
        # قراءة تاريخ المباراة وتوحيده
        dt = parse_date_safe(m.get("utcDate"))
        dt = to_naive_utc(dt)

        if dt is None or dt >= ref_date:
            continue

        # قراءة معرّفات الفرق
        h_id = m.get("homeTeam", {}).get("id")
        a_id = m.get("awayTeam", {}).get("id")

        if h_id is None or a_id is None:
            continue

        # التحقق من مشاركة الفريق
        if int(h_id) != team_id and int(a_id) != team_id:
            continue

        # قراءة النتيجة
        hg, ag = parse_score(m)
        if hg is None or ag is None:
            continue

        # حساب النقاط
        if int(h_id) == team_id:
            pts = 3 if hg > ag else (1 if hg == ag else 0)
        else:
            pts = 3 if ag > hg else (1 if hg == ag else 0)

        rows.append((dt, pts))

    # ترتيب حسب التاريخ (الأحدث أولاً)
    rows.sort(key=lambda x: x[0], reverse=True)
    last = rows[:num_matches]

    if not last:
        return {"avg_points": 1.0, "matches_found": 0}

    avg_pts = sum(p for _, p in last) / len(last)
    return {"avg_points": avg_pts, "matches_found": len(last)}


# -----------------------------------------------------------------------------
# القسم الثاني: تحميل البيانات
# -----------------------------------------------------------------------------

def load_json_file(path: Path, description: str) -> Optional[Any]:
    """
    تحميل ملف JSON بأمان مع رسائل خطأ واضحة.

    المعاملات:
        path: مسار الملف
        description: وصف الملف (للرسائل)

    العائد:
        محتوى الملف (قاموس أو قائمة)، أو None في حالة الفشل
    """
    if not path.exists():
        log(f"ملف {description} غير موجود: {path}", "ERROR")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_size = path.stat().st_size
        log(f"تم تحميل {description}: {path} ({file_size:,} بايت)", "INFO")
        return data

    except json.JSONDecodeError as e:
        log(f"خطأ في تحليل ملف {description}: {e}", "ERROR")
        return None
    except Exception as e:
        log(f"خطأ أثناء تحميل {description}: {e}", "ERROR")
        return None


def load_all_required_data() -> Optional[Dict[str, Any]]:
    """
    تحميل جميع الملفات المطلوبة لإنشاء الميزات.

    الملفات المطلوبة:
    - matches.json: بيانات المباريات
    - team_factors.json: عوامل الهجوم والدفاع
    - elo_ratings.json: تقييمات ELO
    - league_averages.json: متوسطات الدوري

    العائد:
        قاموس يحتوي على جميع البيانات المحمّلة، أو None في حالة الفشل
    """
    log("جارٍ تحميل جميع الملفات المطلوبة...", "INFO")

    # تحميل المباريات
    all_matches = load_json_file(
        config.DATA_DIR / "matches.json",
        "بيانات المباريات"
    )

    if all_matches is None:
        log(
            "فشل تحميل بيانات المباريات. "
            "يرجى تشغيل 01_pipeline.py أولاً.",
            "CRITICAL"
        )
        return None

    if not isinstance(all_matches, list):
        log("تنسيق ملف المباريات غير متوقع (ليس قائمة).", "CRITICAL")
        return None

    # تحميل عوامل الفرق
    team_factors = load_json_file(
        config.MODELS_DIR / "team_factors.json",
        "عوامل الفرق"
    )

    if team_factors is None:
        log(
            "فشل تحميل عوامل الفرق. "
            "يرجى تشغيل 02_trainer.py أولاً.",
            "CRITICAL"
        )
        return None

    # تحميل تقييمات ELO
    elo_ratings = load_json_file(
        config.MODELS_DIR / "elo_ratings.json",
        "تقييمات ELO"
    )

    if elo_ratings is None:
        log(
            "فشل تحميل تقييمات ELO. "
            "يرجى تشغيل 02_trainer.py أولاً.",
            "CRITICAL"
        )
        return None

    # تحميل متوسطات الدوري
    league_averages = load_json_file(
        config.MODELS_DIR / "league_averages.json",
        "متوسطات الدوري"
    )

    if league_averages is None:
        log(
            "فشل تحميل متوسطات الدوري. "
            "يرجى تشغيل 02_trainer.py أولاً.",
            "CRITICAL"
        )
        return None

    # ملخص التحميل
    log("", "INFO")
    log("ملخص البيانات المحمّلة:", "INFO")
    log(f"  المباريات: {len(all_matches)}", "INFO")
    log(f"  مواسم عوامل الفرق: {len(team_factors)}", "INFO")
    log(f"  مواسم ELO: {len(elo_ratings)}", "INFO")
    log(f"  مواسم متوسطات الدوري: {len(league_averages)}", "INFO")

    return {
        "all_matches": all_matches,
        "team_factors": team_factors,
        "elo_ratings": elo_ratings,
        "league_averages": league_averages,
    }


# -----------------------------------------------------------------------------
# القسم الثالث: التحقق من صحة المباراة
# -----------------------------------------------------------------------------

def validate_match_for_features(match: Dict) -> Tuple[bool, Optional[str]]:
    """
    التحقق من صلاحية مباراة لاستخراج الميزات منها.

    المعاملات:
        match: بيانات المباراة

    العائد:
        tuple يحتوي على (صالحة?, سبب الرفض إن وُجد)
    """
    # يجب أن تكون قاموساً
    if not isinstance(match, dict):
        return False, "ليست قاموساً"

    # يجب أن تحتوي على معرّف
    if "id" not in match:
        return False, "بدون معرّف"

    # يجب أن تحتوي على تاريخ صالح
    utc_date = match.get("utcDate")
    if not utc_date:
        return False, "بدون تاريخ"

    dt = parse_date_safe(utc_date)
    if dt is None:
        return False, "تاريخ غير صالح"

    # يجب أن تحتوي على نتيجة
    hg, ag = parse_score(match)
    if hg is None or ag is None:
        return False, "بدون نتيجة"

    # يجب أن تحتوي على فريقين بمعرّفات
    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})

    if not isinstance(home_team, dict) or not isinstance(away_team, dict):
        return False, "بيانات فرق غير صالحة"

    h_id = home_team.get("id")
    a_id = away_team.get("id")

    if h_id is None or a_id is None:
        return False, "معرّفات فرق مفقودة"

    # يجب أن تحتوي على بيانات موسم
    season = match.get("season", {})
    if not isinstance(season, dict):
        return False, "بدون بيانات موسم"

    start_date = season.get("startDate", "")
    if not start_date or len(start_date) < 4:
        return False, "سنة بداية الموسم مفقودة"

    # يجب أن تحتوي على بيانات مسابقة
    competition = match.get("competition", {})
    if not isinstance(competition, dict):
        return False, "بدون بيانات مسابقة"

    comp_code = competition.get("code")
    if not comp_code:
        return False, "رمز المسابقة مفقود"

    return True, None


def determine_season_key(match: Dict) -> Optional[str]:
    """
    تحديد مفتاح الموسم لمباراة.

    المعاملات:
        match: بيانات المباراة

    العائد:
        مفتاح الموسم (مثلاً "PL_2024") أو None
    """
    season = match.get("season", {}) or {}
    competition = match.get("competition", {}) or {}

    start_date = season.get("startDate", "")
    comp_code = competition.get("code", "UNK")

    if not start_date or len(start_date) < 4:
        return None

    season_year = start_date[:4]
    return f"{comp_code}_{season_year}"


# -----------------------------------------------------------------------------
# القسم الرابع: استخراج الميزات لمباراة واحدة
# -----------------------------------------------------------------------------

def extract_features_for_match(
    match: Dict,
    all_matches: List[Dict],
    team_factors: Dict,
    elo_ratings: Dict,
    league_averages: Dict,
    form_matches: int = DEFAULT_FORM_MATCHES,
) -> Optional[Dict[str, Any]]:
    """
    استخراج الميزات لمباراة واحدة.

    المعاملات:
        match: بيانات المباراة
        all_matches: جميع المباريات (لحساب الفورمة)
        team_factors: عوامل الفرق لكل موسم
        elo_ratings: تقييمات ELO لكل موسم
        league_averages: متوسطات الدوري لكل موسم
        form_matches: عدد المباريات لحساب الفورمة

    العائد:
        قاموس الميزات، أو None إذا تعذّر الاستخراج
    """
    # --- تحديد المباراة ---
    match_id = match.get("id")
    dt = parse_date_safe(match.get("utcDate"))
    hg, ag = parse_score(match)

    if dt is None or hg is None or ag is None:
        return None

    # --- تحديد الموسم ---
    season_key = determine_season_key(match)
    if season_key is None:
        return None

    # --- استخراج معرّفات الفرق ---
    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})

    h_id_val = home_team.get("id")
    a_id_val = away_team.get("id")

    if h_id_val is None or a_id_val is None:
        return None

    h_id = str(h_id_val)
    a_id = str(a_id_val)

    # أسماء الفرق (للتوثيق)
    h_name = home_team.get("name", "Unknown")
    a_name = away_team.get("name", "Unknown")

    # رمز المسابقة
    comp_code = match.get("competition", {}).get("code", "UNK")

    # --- استرجاع بيانات الموسم ---
    season_factors = team_factors.get(season_key, {})
    season_elo = elo_ratings.get(season_key, {})
    season_avg = league_averages.get(season_key, {})

    # التحقق من وجود بيانات الموسم
    if not season_factors or not season_elo or not season_avg:
        return None

    # --- استخراج عوامل الهجوم والدفاع ---
    attack_factors = season_factors.get("attack", {})
    defense_factors = season_factors.get("defense", {})

    home_attack = float(attack_factors.get(h_id, 1.0))
    away_attack = float(attack_factors.get(a_id, 1.0))
    home_defense = float(defense_factors.get(h_id, 1.0))
    away_defense = float(defense_factors.get(a_id, 1.0))

    # --- استخراج تقييمات ELO ---
    home_elo = float(season_elo.get(h_id, 1500.0))
    away_elo = float(season_elo.get(a_id, 1500.0))
    elo_diff = home_elo - away_elo

    # --- حساب الفورمة ---
    home_form = calculate_team_form(
        all_matches, int(h_id_val), dt, num_matches=form_matches
    )
    away_form = calculate_team_form(
        all_matches, int(a_id_val), dt, num_matches=form_matches
    )

    home_avg_points = float(home_form.get("avg_points", 1.0))
    away_avg_points = float(away_form.get("avg_points", 1.0))

    home_form_count = int(home_form.get("matches_found", 0))
    away_form_count = int(away_form.get("matches_found", 0))

    # --- حساب النتيجة (الهدف) ---
    if hg > ag:
        result = 1       # فوز المضيف
    elif hg == ag:
        result = 0       # تعادل
    else:
        result = -1      # فوز الضيف

    # --- التحقق من صحة القيم العددية ---
    numeric_values = [
        home_attack, away_attack, home_defense, away_defense,
        home_elo, away_elo, elo_diff,
        home_avg_points, away_avg_points,
    ]

    for val in numeric_values:
        if np is not None:
            if np.isnan(val) or np.isinf(val):
                return None
        else:
            if val != val:  # NaN check بدون numpy
                return None

    # --- بناء قاموس الميزات ---
    features = {
        # أعمدة وصفية
        "match_id": match_id,
        "match_date": dt.isoformat(),
        "season_key": season_key,
        "competition_code": comp_code,
        "home_team_id": h_id,
        "away_team_id": a_id,
        "home_team_name": h_name,
        "away_team_name": a_name,

        # ميزات التدريب
        "home_attack": home_attack,
        "away_attack": away_attack,
        "home_defense": home_defense,
        "away_defense": away_defense,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_avg_points": home_avg_points,
        "away_avg_points": away_avg_points,

        # أعمدة معلوماتية إضافية (لا تُستخدم في التدريب لكنها مفيدة للتحليل)
        "home_form_matches": home_form_count,
        "away_form_matches": away_form_count,
        "avg_home_goals_league": float(season_avg.get("avg_home_goals", 0.0)),
        "avg_away_goals_league": float(season_avg.get("avg_away_goals", 0.0)),

        # أعمدة الهدف
        "actual_home_goals": hg,
        "actual_away_goals": ag,
        "result": result,
    }

    return features


# -----------------------------------------------------------------------------
# القسم الخامس: إنشاء مجموعة البيانات الكاملة
# -----------------------------------------------------------------------------

def generate_features(
    data: Dict[str, Any],
    form_matches: int = DEFAULT_FORM_MATCHES,
    progress_interval: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    إنشاء الميزات لجميع المباريات الصالحة.

    المعاملات:
        data: البيانات المحمّلة (من load_all_required_data)
        form_matches: عدد المباريات لحساب الفورمة
        progress_interval: عدد المباريات بين كل رسالة تقدم

    العائد:
        tuple يحتوي على:
        - قائمة قواميس الميزات
        - قاموس إحصائيات (عدد المعالجة، المقبولة، المرفوضة، أسباب الرفض)
    """
    all_matches = data["all_matches"]
    team_factors = data["team_factors"]
    elo_ratings = data["elo_ratings"]
    league_averages = data["league_averages"]

    total_matches = len(all_matches)
    log(f"جارٍ معالجة {total_matches} مباراة...", "INFO")

    feature_list: List[Dict[str, Any]] = []

    # إحصائيات
    stats = {
        "total": total_matches,
        "processed": 0,
        "accepted": 0,
        "rejected_validation": 0,
        "rejected_no_season": 0,
        "rejected_extraction": 0,
    }

    # أسباب الرفض المفصّلة
    rejection_reasons: Dict[str, int] = {}

    for i, match in enumerate(all_matches):
        stats["processed"] += 1

        # عرض التقدم
        if (i + 1) % progress_interval == 0 or (i + 1) == total_matches:
            pct = ((i + 1) / total_matches) * 100
            log(
                f"  التقدم: {i + 1}/{total_matches} ({pct:.1f}%) — "
                f"مقبولة: {stats['accepted']}",
                "INFO"
            )

        # التحقق من صلاحية المباراة
        is_valid, rejection_reason = validate_match_for_features(match)

        if not is_valid:
            stats["rejected_validation"] += 1
            if rejection_reason:
                rejection_reasons[rejection_reason] = (
                    rejection_reasons.get(rejection_reason, 0) + 1
                )
            continue

        # استخراج الميزات
        try:
            features = extract_features_for_match(
                match=match,
                all_matches=all_matches,
                team_factors=team_factors,
                elo_ratings=elo_ratings,
                league_averages=league_averages,
                form_matches=form_matches,
            )

            if features is not None:
                feature_list.append(features)
                stats["accepted"] += 1
            else:
                stats["rejected_extraction"] += 1
                rejection_reasons["فشل استخراج الميزات"] = (
                    rejection_reasons.get("فشل استخراج الميزات", 0) + 1
                )

        except Exception as e:
            stats["rejected_extraction"] += 1
            error_key = f"خطأ: {type(e).__name__}"
            rejection_reasons[error_key] = rejection_reasons.get(error_key, 0) + 1

    # عرض الإحصائيات
    print("")
    log("إحصائيات الاستخراج:", "INFO")
    log(f"  إجمالي المباريات: {stats['total']}", "INFO")
    log(f"  المباريات المُعالجة: {stats['processed']}", "INFO")
    log(f"  المباريات المقبولة: {stats['accepted']}", "INFO")
    log(
        f"  المباريات المرفوضة (تحقق): {stats['rejected_validation']}",
        "INFO"
    )
    log(
        f"  المباريات المرفوضة (استخراج): {stats['rejected_extraction']}",
        "INFO"
    )

    if rejection_reasons:
        log("", "INFO")
        log("أسباب الرفض:", "DEBUG")
        for reason, count in sorted(
            rejection_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            log(f"    {reason}: {count}", "DEBUG")

    return feature_list, stats


# -----------------------------------------------------------------------------
# القسم السادس: تحليل مجموعة البيانات
# -----------------------------------------------------------------------------

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    تحليل مجموعة البيانات المُنتجة وعرض إحصائيات مفصّلة.

    المعاملات:
        df: مجموعة البيانات

    العائد:
        قاموس يحتوي على نتائج التحليل
    """
    analysis = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    log("", "INFO")
    log("=" * 60, "INFO")
    log("تحليل مجموعة البيانات المُنتجة", "INFO")
    log("=" * 60, "INFO")
    log(f"  عدد الصفوف: {len(df)}", "INFO")
    log(f"  عدد الأعمدة: {len(df.columns)}", "INFO")

    # --- توزيع النتائج ---
    if "result" in df.columns:
        result_counts = df["result"].value_counts().sort_index()
        result_labels = {1: "فوز المضيف", 0: "تعادل", -1: "فوز الضيف"}

        log("", "INFO")
        log("  توزيع النتائج:", "INFO")

        result_distribution = {}
        for result_value, count in result_counts.items():
            pct = (count / len(df)) * 100
            label = result_labels.get(result_value, str(result_value))
            log(f"    {label} ({result_value}): {count} ({pct:.1f}%)", "INFO")
            result_distribution[str(result_value)] = {
                "count": int(count),
                "percentage": round(pct, 2),
            }

        analysis["result_distribution"] = result_distribution

    # --- توزيع المواسم ---
    if "season_key" in df.columns:
        season_counts = df["season_key"].value_counts().sort_index()

        log("", "INFO")
        log("  توزيع المواسم:", "INFO")

        season_distribution = {}
        for season, count in season_counts.items():
            log(f"    {season}: {count} مباراة", "INFO")
            season_distribution[season] = int(count)

        analysis["season_distribution"] = season_distribution

    # --- توزيع المسابقات ---
    if "competition_code" in df.columns:
        comp_counts = df["competition_code"].value_counts().sort_index()

        log("", "INFO")
        log("  توزيع المسابقات:", "INFO")

        comp_distribution = {}
        for comp, count in comp_counts.items():
            log(f"    {comp}: {count} مباراة", "INFO")
            comp_distribution[comp] = int(count)

        analysis["competition_distribution"] = comp_distribution

    # --- إحصائيات الميزات العددية ---
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    if available_features:
        log("", "INFO")
        log("  إحصائيات الميزات:", "INFO")

        feature_stats = {}
        for feat in available_features:
            col = df[feat]
            stats = {
                "mean": round(float(col.mean()), 4),
                "std": round(float(col.std()), 4),
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
                "nulls": int(col.isnull().sum()),
            }
            feature_stats[feat] = stats

            log(
                f"    {feat:20s}: "
                f"μ={stats['mean']:8.4f}, "
                f"σ={stats['std']:8.4f}, "
                f"min={stats['min']:8.4f}, "
                f"max={stats['max']:8.4f}"
                + (f", nulls={stats['nulls']}" if stats['nulls'] > 0 else ""),
                "INFO"
            )

        analysis["feature_stats"] = feature_stats

    # --- القيم المفقودة ---
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if not cols_with_nulls.empty:
        log("", "INFO")
        log("  أعمدة تحتوي على قيم مفقودة:", "WARNING")
        for col_name, null_count in cols_with_nulls.items():
            pct = (null_count / len(df)) * 100
            log(f"    {col_name}: {null_count} ({pct:.1f}%)", "WARNING")

        analysis["columns_with_nulls"] = {
            str(k): int(v) for k, v in cols_with_nulls.items()
        }
    else:
        log("", "INFO")
        log("  ✅ لا توجد قيم مفقودة.", "INFO")
        analysis["columns_with_nulls"] = {}

    # --- فورمة الفرق ---
    if "home_form_matches" in df.columns and "away_form_matches" in df.columns:
        zero_home_form = int((df["home_form_matches"] == 0).sum())
        zero_away_form = int((df["away_form_matches"] == 0).sum())

        if zero_home_form > 0 or zero_away_form > 0:
            log("", "INFO")
            log(
                f"  ⚠ مباريات بدون فورمة سابقة: "
                f"مضيف={zero_home_form}, ضيف={zero_away_form}",
                "WARNING"
            )

        analysis["zero_form"] = {
            "home": zero_home_form,
            "away": zero_away_form,
        }

    log("=" * 60, "INFO")

    return analysis


# -----------------------------------------------------------------------------
# القسم السابع: حفظ الملفات
# -----------------------------------------------------------------------------

def create_backup(file_path: Path) -> Optional[Path]:
    """
    إنشاء نسخة احتياطية من ملف موجود.

    المعاملات:
        file_path: مسار الملف الأصلي

    العائد:
        مسار النسخة الاحتياطية، أو None
    """
    if not file_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{BACKUP_SUFFIX}{file_path.suffix}"
    backup_path = file_path.parent / backup_name

    try:
        shutil.copy2(file_path, backup_path)
        log(f"نسخة احتياطية: {backup_path.name}", "DEBUG")
        return backup_path
    except OSError as e:
        log(f"فشل إنشاء النسخة الاحتياطية: {e}", "WARNING")
        return None


def cleanup_old_backups(
    directory: Path,
    base_name: str,
    keep_last: int = MAX_BACKUPS,
):
    """
    حذف النسخ الاحتياطية القديمة.

    المعاملات:
        directory: مجلد البحث
        base_name: الاسم الأساسي للملف
        keep_last: عدد النسخ المراد الاحتفاظ بها
    """
    try:
        pattern = f"{base_name}_*{BACKUP_SUFFIX}.*"
        backups = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)

        if len(backups) > keep_last:
            for old_backup in backups[:len(backups) - keep_last]:
                try:
                    old_backup.unlink()
                except OSError:
                    pass
    except Exception:
        pass


def save_dataset(
    df: pd.DataFrame,
    output_path: Path,
    create_backups_flag: bool = True,
) -> bool:
    """
    حفظ مجموعة البيانات في ملف CSV بأمان.

    يتم الحفظ عبر ملف مؤقت أولاً ثم إعادة التسمية لمنع تلف البيانات.

    المعاملات:
        df: مجموعة البيانات
        output_path: مسار ملف CSV النهائي
        create_backups_flag: إنشاء نسخة احتياطية قبل الكتابة

    العائد:
        True إذا تم الحفظ بنجاح
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # إنشاء نسخة احتياطية
        if create_backups_flag:
            create_backup(output_path)
            cleanup_old_backups(output_path.parent, output_path.stem)

        # حفظ في ملف مؤقت
        temp_path = output_path.with_suffix(".tmp")
        df.to_csv(temp_path, index=False, encoding="utf-8")

        # التحقق من صحة الملف المؤقت
        verification_df = pd.read_csv(temp_path)
        if len(verification_df) != len(df):
            log(
                f"⚠ تحذير: حجم الملف المحفوظ ({len(verification_df)}) "
                f"لا يطابق الأصل ({len(df)})",
                "WARNING"
            )

        # نقل الملف المؤقت ليحلّ محل النهائي
        temp_path.replace(output_path)

        file_size = output_path.stat().st_size
        log(
            f"✅ تم حفظ مجموعة البيانات: {output_path} "
            f"({len(df)} صف، {file_size:,} بايت)",
            "INFO"
        )
        return True

    except Exception as e:
        log(f"❌ فشل حفظ مجموعة البيانات: {e}", "ERROR")
        # تنظيف الملف المؤقت
        temp_path = output_path.with_suffix(".tmp")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def save_metadata(
    metadata_path: Path,
    stats: Dict[str, int],
    analysis: Dict[str, Any],
    form_matches: int,
    duration_seconds: float,
) -> bool:
    """
    حفظ البيانات الوصفية لعملية إنشاء الميزات.

    المعاملات:
        metadata_path: مسار ملف البيانات الوصفية
        stats: إحصائيات الاستخراج
        analysis: نتائج التحليل
        form_matches: عدد مباريات الفورمة
        duration_seconds: مدة العملية

    العائد:
        True إذا تم الحفظ بنجاح
    """
    try:
        metadata = {
            "version": getattr(config, "VERSION", "N/A"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "parameters": {
                "form_matches": form_matches,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": "result",
                "target_classes": {
                    "1": "فوز المضيف (Home Win)",
                    "0": "تعادل (Draw)",
                    "-1": "فوز الضيف (Away Win)",
                },
            },
            "extraction_stats": stats,
            "dataset_analysis": analysis,
        }

        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        log(f"تم حفظ البيانات الوصفية: {metadata_path}", "INFO")
        return True

    except Exception as e:
        log(f"فشل حفظ البيانات الوصفية: {e}", "WARNING")
        return False


# -----------------------------------------------------------------------------
# القسم الثامن: الدالة الرئيسية
# -----------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """
    تحويل المدة بالثواني إلى نص قابل للقراءة.

    المعاملات:
        seconds: المدة بالثواني

    العائد:
        نص يمثّل المدة
    """
    if seconds < 60:
        return f"{seconds:.1f} ثانية"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes} دقيقة و {secs} ثانية"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} ساعة و {minutes} دقيقة"


def run_feature_generator(
    form_matches: int = DEFAULT_FORM_MATCHES,
    create_backups_flag: bool = True,
    dry_run: bool = False,
    progress_interval: int = 500,
):
    """
    الدالة الرئيسية لإنشاء ميزات تعلم الآلة.

    المعاملات:
        form_matches: عدد المباريات الأخيرة لحساب الفورمة
        create_backups_flag: إنشاء نسخ احتياطية قبل الكتابة
        dry_run: تشغيل بدون حفظ
        progress_interval: عدد المباريات بين كل رسالة تقدم
    """
    start_time = datetime.now(timezone.utc)

    log("=" * 70, "INFO")
    log("بدء عملية إنشاء الميزات لتعلم الآلة", "INFO")
    log(f"الوقت: {start_time.isoformat()}", "INFO")
    log(f"الإصدار: {getattr(config, 'VERSION', 'N/A')}", "INFO")
    log("=" * 70, "INFO")

    if dry_run:
        log("⚠ وضع التشغيل الجاف (Dry Run): لن يتم حفظ أي ملفات.", "WARNING")

    log(f"عدد مباريات الفورمة: {form_matches}", "INFO")

    # =========================================================================
    # المرحلة 1: تحميل البيانات
    # =========================================================================
    log("--- المرحلة 1: تحميل البيانات ---", "INFO")

    data = load_all_required_data()
    if data is None:
        log("فشل تحميل البيانات المطلوبة. لا يمكن المتابعة.", "CRITICAL")
        return

    # =========================================================================
    # المرحلة 2: إنشاء الميزات
    # =========================================================================
    log("--- المرحلة 2: إنشاء الميزات ---", "INFO")

    feature_list, stats = generate_features(
        data=data,
        form_matches=form_matches,
        progress_interval=progress_interval,
    )

    # التحقق من وجود ميزات كافية
    if not feature_list:
        log(
            "لم يتم إنشاء أي ميزات. "
            "يرجى التحقق من الملفات المدخلة وتوافقها.",
            "CRITICAL"
        )
        return

    if len(feature_list) < MIN_FEATURES_REQUIRED:
        log(
            f"عدد الميزات المُنتجة ({len(feature_list)}) "
            f"أقل من الحد الأدنى ({MIN_FEATURES_REQUIRED}). "
            f"قد لا تكون مجموعة البيانات كافية للتدريب.",
            "WARNING"
        )

    # =========================================================================
    # المرحلة 3: بناء DataFrame
    # =========================================================================
    log("--- المرحلة 3: بناء مجموعة البيانات ---", "INFO")

    df = pd.DataFrame(feature_list)

    # ترتيب الأعمدة
    desired_order = METADATA_COLUMNS + FEATURE_COLUMNS + [
        "home_form_matches",
        "away_form_matches",
        "avg_home_goals_league",
        "avg_away_goals_league",
    ] + TARGET_COLUMNS

    # إعادة ترتيب الأعمدة الموجودة فقط
    existing_columns = [c for c in desired_order if c in df.columns]
    # إضافة أي أعمدة إضافية لم تكن في الترتيب المرغوب
    remaining_columns = [c for c in df.columns if c not in existing_columns]
    df = df[existing_columns + remaining_columns]

    # ترتيب حسب التاريخ
    if "match_date" in df.columns:
        df = df.sort_values("match_date").reset_index(drop=True)
        log("تم ترتيب مجموعة البيانات حسب التاريخ.", "INFO")

    log(f"حجم مجموعة البيانات: {len(df)} صف × {len(df.columns)} عمود", "INFO")

    # =========================================================================
    # المرحلة 4: تحليل مجموعة البيانات
    # =========================================================================
    log("--- المرحلة 4: تحليل مجموعة البيانات ---", "INFO")

    analysis = analyze_dataset(df)

    # =========================================================================
    # المرحلة 5: حفظ الملفات
    # =========================================================================
    end_time = datetime.now(timezone.utc)
    duration_seconds = (end_time - start_time).total_seconds()

    if dry_run:
        log("--- المرحلة 5: تخطي الحفظ (وضع تجريبي) ---", "INFO")
        log(f"[DRY RUN] كان سيتم حفظ {len(df)} صف.", "INFO")
    else:
        log("--- المرحلة 5: حفظ الملفات ---", "INFO")

        # التأكد من وجود مجلد البيانات
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # حفظ مجموعة البيانات
        output_path = config.DATA_DIR / "ml_dataset.csv"
        dataset_saved = save_dataset(
            df, output_path,
            create_backups_flag=create_backups_flag,
        )

        if not dataset_saved:
            log("❌ فشل حفظ مجموعة البيانات.", "ERROR")

        # حفظ البيانات الوصفية
        metadata_path = config.DATA_DIR / "ml_dataset_metadata.json"
        save_metadata(
            metadata_path=metadata_path,
            stats=stats,
            analysis=analysis,
            form_matches=form_matches,
            duration_seconds=duration_seconds,
        )

    # =========================================================================
    # ملخص نهائي
    # =========================================================================
    print("")
    log("=" * 70, "INFO")
    log("ملخص عملية إنشاء الميزات", "INFO")
    log("=" * 70, "INFO")
    log(f"  إجمالي المباريات المُعالجة: {stats['processed']}", "INFO")
    log(f"  المباريات المقبولة: {stats['accepted']}", "INFO")
    log(
        f"  المباريات المرفوضة: "
        f"{stats['rejected_validation'] + stats['rejected_extraction']}",
        "INFO"
    )
    log(f"  حجم مجموعة البيانات: {len(df)} صف × {len(df.columns)} عمود", "INFO")
    log(f"  عدد ميزات التدريب: {len(FEATURE_COLUMNS)}", "INFO")
    log(f"  عدد مباريات الفورمة: {form_matches}", "INFO")
    log(f"  المدة: {format_duration(duration_seconds)}", "INFO")

    if not dry_run:
        log(f"  ملف البيانات: {config.DATA_DIR / 'ml_dataset.csv'}", "INFO")
        log(
            f"  ملف البيانات الوصفية: {config.DATA_DIR / 'ml_dataset_metadata.json'}",
            "INFO"
        )

    log("=" * 70, "INFO")
    log("انتهت عملية إنشاء الميزات بنجاح ✅", "INFO")
    log("=" * 70, "INFO")


# -----------------------------------------------------------------------------
# نقطة الدخول
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="إنشاء مجموعة بيانات الميزات لتدريب نموذج تعلم الآلة.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  python 04_feature_generator.py                       # تشغيل عادي
  python 04_feature_generator.py --form-matches 10     # فورمة من آخر 10 مباريات
  python 04_feature_generator.py --dry-run             # تشغيل تجريبي
  python 04_feature_generator.py --no-backup           # بدون نسخ احتياطية
  python 04_feature_generator.py --progress 100        # تقرير كل 100 مباراة
        """
    )

    parser.add_argument(
        "--form-matches",
        type=int,
        default=DEFAULT_FORM_MATCHES,
        help=f"عدد المباريات الأخيرة لحساب فورمة الفريق. (افتراضي: {DEFAULT_FORM_MATCHES})"
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="تعطيل إنشاء النسخ الاحتياطية قبل الكتابة فوق الملفات."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="تشغيل تجريبي: إنشاء الميزات وتحليلها بدون حفظ أي ملفات."
    )

    parser.add_argument(
        "--progress",
        type=int,
        default=500,
        help="عدد المباريات بين كل رسالة تقدم. (افتراضي: 500)"
    )

    args = parser.parse_args()

    # التحقق من صحة المدخلات
    if args.form_matches < 1:
        print("خطأ: عدد مباريات الفورمة يجب أن يكون >= 1")
        sys.exit(1)

    if args.progress < 1:
        print("خطأ: فترة التقدم يجب أن تكون >= 1")
        sys.exit(1)

    try:
        run_feature_generator(
            form_matches=args.form_matches,
            create_backups_flag=not args.no_backup,
            dry_run=args.dry_run,
            progress_interval=args.progress,
        )
    except KeyboardInterrupt:
        print("")
        log("تم إيقاف العملية بواسطة المستخدم (Ctrl+C).", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"خطأ غير متوقع: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)
