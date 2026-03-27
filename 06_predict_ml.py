# 06_predict_ml.py
# -----------------------------------------------------------------------------
# الوصف:
# السكريبت النهائي للتنبؤ بنتيجة مباراة قادمة باستخدام نموذج XGBoost المدرب.
#
# يقوم بـ:
# 1. تحميل نموذج XGBoost المدرب والبيانات الإحصائية المطلوبة
# 2. بناء ميزات المباراة المستهدفة
# 3. إجراء التنبؤ واستخراج الاحتمالات
# 4. عرض النتائج وحفظها (اختياري)
#
# التحسينات:
# - استبدال LabelEncoder بـ model.classes_ لضمان التوافق مع النموذج المدرّب
# - حماية استيراد المكتبات (xgboost, sklearn)
# - دعم معاملات سطر الأوامر (--home, --away, --comp, --season)
# - تحميل أسماء الفرق من teams.json لعرض أوضح
# - تحميل البيانات الوصفية للنموذج (xgboost_metadata.json) للتحقق من التوافق
# - إصلاح مشكلة مقارنة التواريخ (naive vs aware)
# - حماية من القيم الفارغة والأخطاء غير المتوقعة
# - دعم حفظ نتيجة التنبؤ في ملف JSON
# - دعم التنبؤ باسم الفريق بدلاً من المعرّف فقط
# - إضافة تقرير مفصّل يشمل الميزات المستخدمة والثقة
# - دعم وضع التشغيل الجاف (--dry-run)
#
# الاستخدام:
#   python 06_predict_ml.py
#   python 06_predict_ml.py --home 65 --away 64 --comp PL --season 2025
#   python 06_predict_ml.py --home-name "Manchester City" --away-name "Liverpool" --comp PL
#   python 06_predict_ml.py --home 65 --away 64 --comp PL --save
# -----------------------------------------------------------------------------

import sys
import os
import json
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

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
    np = None

# قد لا يتوفر XGBoost في كل بيئة تشغيل
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe, parse_score


# -----------------------------------------------------------------------------
# استيراد calculate_team_form مع حماية التواريخ
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


# محاولة استيراد الدالة الأصلية
try:
    from common.modeling import calculate_team_form as _calculate_team_form_base

    def calculate_team_form(
        all_matches: List[Dict],
        team_id: int,
        ref_date: datetime,
        num_matches: int = 5,
    ) -> Dict[str, Any]:
        """
        غلاف حول calculate_team_form الأصلية مع توحيد التواريخ إلى naive-UTC.

        المعاملات:
            all_matches: جميع المباريات
            team_id: معرّف الفريق
            ref_date: التاريخ المرجعي
            num_matches: عدد المباريات الأخيرة

        العائد:
            قاموس يحتوي على avg_points وغيرها
        """
        ref_date = to_naive_utc(ref_date)
        return _calculate_team_form_base(
            all_matches, team_id, ref_date, num_matches=num_matches
        )

except ImportError:

    def calculate_team_form(
        all_matches: List[Dict],
        team_id: int,
        ref_date: datetime,
        num_matches: int = 5,
    ) -> Dict[str, Any]:
        """
        بديل بسيط لحساب فورمة الفريق.
        يبحث عن آخر num_matches مباراة قبل ref_date ويحسب متوسط النقاط.

        المعاملات:
            all_matches: جميع المباريات
            team_id: معرّف الفريق
            ref_date: التاريخ المرجعي
            num_matches: عدد المباريات الأخيرة

        العائد:
            قاموس يحتوي على avg_points و matches_found
        """
        ref_date = to_naive_utc(ref_date)

        if ref_date is None:
            return {"avg_points": 1.0, "matches_found": 0}

        rows = []
        for m in all_matches:
            dt = parse_date_safe(m.get("utcDate"))
            dt = to_naive_utc(dt)

            if dt is None or dt >= ref_date:
                continue

            h = m.get("homeTeam", {}).get("id")
            a = m.get("awayTeam", {}).get("id")

            if h is None or a is None:
                continue

            if int(h) != team_id and int(a) != team_id:
                continue

            hg, ag = parse_score(m)
            if hg is None or ag is None:
                continue

            if int(h) == team_id:
                pts = 3 if hg > ag else (1 if hg == ag else 0)
            else:
                pts = 3 if ag > hg else (1 if hg == ag else 0)

            rows.append((dt, pts))

        rows.sort(key=lambda x: x[0], reverse=True)
        last = rows[:num_matches]

        if not last:
            return {"avg_points": 1.0, "matches_found": 0}

        avg_pts = sum(p for _, p in last) / len(last)
        return {"avg_points": avg_pts, "matches_found": len(last)}


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# قائمة الميزات المتوقعة (يجب أن تتطابق مع 04_feature_generator و 05_train_ml_model)
EXPECTED_FEATURES = [
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

# الفئات المتوقعة
EXPECTED_CLASSES = [-1, 0, 1]

# أسماء الفئات بالعربية
CLASS_LABELS = {
    -1: "فوز الضيف (Away Win)",
    0: "تعادل (Draw)",
    1: "فوز المضيف (Home Win)",
}

# عدد المباريات الافتراضي لحساب الفورمة
DEFAULT_FORM_MATCHES = 5

# القيم الافتراضية للمباراة التجريبية
DEFAULT_HOME_TEAM_ID = 65   # Manchester City
DEFAULT_AWAY_TEAM_ID = 64   # Liverpool
DEFAULT_COMP_CODE = "PL"


# -----------------------------------------------------------------------------
# القسم الأول: تحميل البيانات والنماذج
# -----------------------------------------------------------------------------

def check_dependencies() -> bool:
    """
    التحقق من توفر جميع المكتبات المطلوبة.

    العائد:
        True إذا كانت جميع المكتبات متوفرة
    """
    missing = []

    if xgb is None:
        missing.append("xgboost")

    if missing:
        log(
            f"المكتبات التالية غير متوفرة: {', '.join(missing)}. "
            f"يرجى تثبيتها عبر: pip install {' '.join(missing)}",
            "CRITICAL"
        )
        return False

    return True


def load_json_file(path: Path, description: str) -> Optional[Any]:
    """
    تحميل ملف JSON بأمان.

    المعاملات:
        path: مسار الملف
        description: وصف الملف (للرسائل)

    العائد:
        محتوى الملف، أو None في حالة الفشل
    """
    if not path.exists():
        log(f"ملف {description} غير موجود: {path}", "ERROR")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        log(f"تم تحميل {description}: {path.name}", "INFO")
        return data

    except json.JSONDecodeError as e:
        log(f"خطأ في تحليل ملف {description}: {e}", "ERROR")
        return None
    except Exception as e:
        log(f"خطأ أثناء تحميل {description}: {e}", "ERROR")
        return None


def load_xgb_model(model_path: Path) -> Optional[Any]:
    """
    تحميل نموذج XGBoost المدرّب.

    المعاملات:
        model_path: مسار ملف النموذج

    العائد:
        نموذج XGBoost مُحمّل، أو None في حالة الفشل
    """
    if xgb is None:
        log("مكتبة xgboost غير مثبتة.", "CRITICAL")
        return None

    if not model_path.exists():
        log(
            f"ملف نموذج XGBoost غير موجود: {model_path}. "
            f"يرجى تشغيل 05_train_ml_model.py أولاً.",
            "CRITICAL"
        )
        return None

    try:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        log(f"تم تحميل نموذج XGBoost: {model_path.name}", "INFO")

        # عرض معلومات النموذج
        model_classes = list(model.classes_)
        log(f"  فئات النموذج (model.classes_): {model_classes}", "DEBUG")

        return model

    except Exception as e:
        log(f"فشل تحميل نموذج XGBoost: {e}", "CRITICAL")
        return None


def load_model_metadata(metadata_path: Path) -> Optional[Dict]:
    """
    تحميل البيانات الوصفية للنموذج المدرّب (إن وُجدت).
    هذه البيانات تُستخدم للتحقق من توافق الميزات والفئات.

    المعاملات:
        metadata_path: مسار ملف البيانات الوصفية

    العائد:
        قاموس البيانات الوصفية، أو None
    """
    if not metadata_path.exists():
        log(
            "ملف البيانات الوصفية للنموذج غير موجود (اختياري). "
            "سيتم استخدام الإعدادات الافتراضية.",
            "DEBUG"
        )
        return None

    return load_json_file(metadata_path, "بيانات وصفية للنموذج")


def load_teams_map(teams_path: Path) -> Optional[Dict]:
    """
    تحميل خريطة الفرق لعرض أسماء الفرق بدلاً من المعرّفات فقط.

    المعاملات:
        teams_path: مسار ملف teams.json

    العائد:
        قاموس الفرق، أو None
    """
    return load_json_file(teams_path, "خريطة الفرق")


def load_all_prediction_data() -> Optional[Dict[str, Any]]:
    """
    تحميل جميع الملفات المطلوبة للتنبؤ.

    العائد:
        قاموس يحتوي على جميع البيانات المحمّلة، أو None في حالة الفشل
    """
    log("جارٍ تحميل البيانات والنماذج المطلوبة...", "INFO")

    # تحميل المباريات
    all_matches = load_json_file(
        config.DATA_DIR / "matches.json",
        "بيانات المباريات"
    )
    if all_matches is None:
        log(
            "فشل تحميل بيانات المباريات. يرجى تشغيل 01_pipeline.py أولاً.",
            "CRITICAL"
        )
        return None

    # تحميل عوامل الفرق
    team_factors = load_json_file(
        config.MODELS_DIR / "team_factors.json",
        "عوامل الفرق"
    )
    if team_factors is None:
        log(
            "فشل تحميل عوامل الفرق. يرجى تشغيل 02_trainer.py أولاً.",
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
            "فشل تحميل تقييمات ELO. يرجى تشغيل 02_trainer.py أولاً.",
            "CRITICAL"
        )
        return None

    # تحميل نموذج XGBoost
    model = load_xgb_model(config.MODELS_DIR / "xgboost_model.json")
    if model is None:
        return None

    # تحميل البيانات الوصفية (اختياري)
    model_metadata = load_model_metadata(
        config.MODELS_DIR / "xgboost_metadata.json"
    )

    # تحميل خريطة الفرق (اختياري — لعرض الأسماء)
    teams_map = load_teams_map(config.DATA_DIR / "teams.json")

    log("تم تحميل جميع البيانات بنجاح.", "INFO")

    return {
        "all_matches": all_matches,
        "team_factors": team_factors,
        "elo_ratings": elo_ratings,
        "model": model,
        "model_metadata": model_metadata,
        "teams_map": teams_map,
    }


# -----------------------------------------------------------------------------
# القسم الثاني: البحث عن الفرق
# -----------------------------------------------------------------------------

def get_current_season_year() -> int:
    """
    تحديد سنة بداية الموسم الحالي.

    العائد:
        سنة بداية الموسم الحالي
    """
    now = datetime.now()
    season_start_month = getattr(config, "CURRENT_SEASON_START_MONTH", 7)

    if now.month >= season_start_month:
        return now.year
    else:
        return now.year - 1


def find_team_id_by_name(
    teams_map: Optional[Dict],
    team_name: str,
    comp_code: Optional[str] = None,
) -> Optional[int]:
    """
    البحث عن معرّف فريق بناءً على اسمه.

    يبحث في خريطة الفرق عن تطابق جزئي (case-insensitive).

    المعاملات:
        teams_map: خريطة الفرق
        team_name: اسم الفريق المطلوب
        comp_code: رمز المسابقة (اختياري — لتضييق البحث)

    العائد:
        معرّف الفريق، أو None إذا لم يُوجد
    """
    if teams_map is None or not team_name:
        return None

    team_name_lower = team_name.lower().strip()
    candidates = []

    for team_key, team_data in teams_map.items():
        if not isinstance(team_data, dict):
            continue

        team_id = team_data.get("id")
        if team_id is None:
            continue

        # التحقق من المسابقة إذا حُددت
        if comp_code:
            competitions = team_data.get("competitions", [])
            if comp_code not in competitions:
                continue

        # البحث في جميع الأسماء
        names = team_data.get("names", [])
        if not isinstance(names, list):
            names = [names] if names else []

        for name in names:
            if not name:
                continue

            name_lower = name.lower().strip()

            # تطابق كامل
            if name_lower == team_name_lower:
                return int(team_id)

            # تطابق جزئي
            if team_name_lower in name_lower or name_lower in team_name_lower:
                candidates.append((int(team_id), name, len(name)))

    # إرجاع أفضل تطابق جزئي (الأقصر اسماً — أكثر دقة)
    if candidates:
        candidates.sort(key=lambda x: x[2])
        best_id, best_name, _ = candidates[0]

        if len(candidates) > 1:
            log(
                f"وُجد أكثر من تطابق لـ '{team_name}': "
                f"{[(c[1], c[0]) for c in candidates[:5]]}. "
                f"تم اختيار: {best_name} (ID: {best_id})",
                "WARNING"
            )
        else:
            log(f"تم العثور على الفريق: {best_name} (ID: {best_id})", "INFO")

        return best_id

    log(f"لم يتم العثور على فريق يطابق: '{team_name}'", "WARNING")
    return None


def get_team_name(
    teams_map: Optional[Dict],
    team_id: int
) -> str:
    """
    الحصول على اسم الفريق من معرّفه.

    المعاملات:
        teams_map: خريطة الفرق
        team_id: معرّف الفريق

    العائد:
        اسم الفريق، أو "Team {id}" إذا لم يُوجد
    """
    if teams_map is None:
        return f"Team {team_id}"

    team_id_str = str(team_id)

    for team_key, team_data in teams_map.items():
        if not isinstance(team_data, dict):
            continue

        if str(team_data.get("id")) == team_id_str:
            names = team_data.get("names", [])
            if isinstance(names, list) and names:
                # اختيار الاسم الأطول (عادة الأكثر وصفاً)
                valid_names = [n for n in names if n]
                if valid_names:
                    return max(valid_names, key=len)
            return f"Team {team_id}"

    return f"Team {team_id}"


# -----------------------------------------------------------------------------
# القسم الثالث: بناء الميزات والتنبؤ
# -----------------------------------------------------------------------------

def validate_season_data(
    season_key: str,
    team_factors: Dict,
    elo_ratings: Dict,
    h_id_str: str,
    a_id_str: str,
) -> Tuple[bool, Optional[str]]:
    """
    التحقق من توفر بيانات الموسم للفريقين.

    المعاملات:
        season_key: مفتاح الموسم
        team_factors: جميع عوامل الفرق
        elo_ratings: جميع تقييمات ELO
        h_id_str: معرّف المضيف كنص
        a_id_str: معرّف الضيف كنص

    العائد:
        tuple يحتوي على (متوفر?, سبب عدم التوفر إن وُجد)
    """
    # التحقق من وجود بيانات الموسم
    season_factors = team_factors.get(season_key)
    season_elo = elo_ratings.get(season_key)

    if not season_factors:
        available_seasons = sorted(team_factors.keys())
        return False, (
            f"لم يتم العثور على عوامل الفرق للموسم '{season_key}'. "
            f"المواسم المتاحة: {available_seasons}"
        )

    if not season_elo:
        available_seasons = sorted(elo_ratings.keys())
        return False, (
            f"لم يتم العثور على تقييمات ELO للموسم '{season_key}'. "
            f"المواسم المتاحة: {available_seasons}"
        )

    # التحقق من وجود الفريقين في بيانات الموسم (تحذير فقط)
    attack_factors = season_factors.get("attack", {})
    defense_factors = season_factors.get("defense", {})

    warnings = []

    if h_id_str not in attack_factors:
        warnings.append(
            f"الفريق المضيف (ID: {h_id_str}) غير موجود في عوامل الهجوم. "
            f"سيتم استخدام القيمة الافتراضية (1.0)."
        )

    if a_id_str not in attack_factors:
        warnings.append(
            f"الفريق الضيف (ID: {a_id_str}) غير موجود في عوامل الهجوم. "
            f"سيتم استخدام القيمة الافتراضية (1.0)."
        )

    if h_id_str not in season_elo:
        warnings.append(
            f"الفريق المضيف (ID: {h_id_str}) غير موجود في تقييمات ELO. "
            f"سيتم استخدام القيمة الافتراضية (1500.0)."
        )

    if a_id_str not in season_elo:
        warnings.append(
            f"الفريق الضيف (ID: {a_id_str}) غير موجود في تقييمات ELO. "
            f"سيتم استخدام القيمة الافتراضية (1500.0)."
        )

    for warning in warnings:
        log(f"  ⚠ {warning}", "WARNING")

    return True, None


def build_features(
    home_team_id: int,
    away_team_id: int,
    competition_code: str,
    season_start_year: int,
    all_matches: List[Dict],
    team_factors: Dict,
    elo_ratings: Dict,
    form_matches: int = DEFAULT_FORM_MATCHES,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    بناء ميزات التنبؤ لمباراة واحدة.

    المعاملات:
        home_team_id: معرّف الفريق المضيف
        away_team_id: معرّف الفريق الضيف
        competition_code: رمز المسابقة
        season_start_year: سنة بداية الموسم
        all_matches: جميع المباريات (لحساب الفورمة)
        team_factors: عوامل الفرق لكل موسم
        elo_ratings: تقييمات ELO لكل موسم
        form_matches: عدد المباريات لحساب الفورمة

    العائد:
        tuple يحتوي على:
        - DataFrame الميزات (صف واحد)، أو None في حالة الفشل
        - قاموس قيم الميزات (للعرض)، أو None
    """
    log("جارٍ بناء ميزات التنبؤ...", "INFO")

    # بناء مفتاح الموسم
    season_key = f"{competition_code}_{season_start_year}"
    h_id_str = str(home_team_id)
    a_id_str = str(away_team_id)

    log(f"  مفتاح الموسم: {season_key}", "INFO")
    log(f"  المضيف: ID={h_id_str}", "INFO")
    log(f"  الضيف: ID={a_id_str}", "INFO")

    # التحقق من توفر بيانات الموسم
    is_available, error_msg = validate_season_data(
        season_key, team_factors, elo_ratings, h_id_str, a_id_str
    )

    if not is_available:
        log(error_msg, "ERROR")
        return None, None

    # استخراج بيانات الموسم
    season_factors = team_factors[season_key]
    season_elo = elo_ratings[season_key]

    attack_factors = season_factors.get("attack", {})
    defense_factors = season_factors.get("defense", {})

    # --- استخراج عوامل الهجوم والدفاع ---
    home_attack = float(attack_factors.get(h_id_str, 1.0))
    away_attack = float(attack_factors.get(a_id_str, 1.0))
    home_defense = float(defense_factors.get(h_id_str, 1.0))
    away_defense = float(defense_factors.get(a_id_str, 1.0))

    # --- استخراج تقييمات ELO ---
    home_elo = float(season_elo.get(h_id_str, 1500.0))
    away_elo = float(season_elo.get(a_id_str, 1500.0))
    elo_diff = home_elo - away_elo

    # --- حساب الفورمة ---
    # نستخدم التاريخ الحالي كمرجع (أي نأخذ آخر N مباراة حتى الآن)
    prediction_date = datetime.now(timezone.utc)

    home_form = calculate_team_form(
        all_matches, home_team_id, prediction_date, num_matches=form_matches
    )
    away_form = calculate_team_form(
        all_matches, away_team_id, prediction_date, num_matches=form_matches
    )

    home_avg_points = float(home_form.get("avg_points", 1.0))
    away_avg_points = float(away_form.get("avg_points", 1.0))

    home_form_count = int(home_form.get("matches_found", 0))
    away_form_count = int(away_form.get("matches_found", 0))

    # --- بناء قاموس الميزات ---
    features_values = {
        "home_attack": home_attack,
        "away_attack": away_attack,
        "home_defense": home_defense,
        "away_defense": away_defense,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_avg_points": home_avg_points,
        "away_avg_points": away_avg_points,
    }

    # عرض الميزات
    log("", "INFO")
    log("  الميزات المُستخرجة:", "INFO")
    for feat_name, feat_value in features_values.items():
        log(f"    {feat_name:20s}: {feat_value:.4f}", "INFO")

    log(
        f"    {'home_form_count':20s}: {home_form_count} مباراة "
        f"(من أصل {form_matches} مطلوبة)",
        "INFO"
    )
    log(
        f"    {'away_form_count':20s}: {away_form_count} مباراة "
        f"(من أصل {form_matches} مطلوبة)",
        "INFO"
    )

    if home_form_count == 0:
        log("  ⚠ لم تُوجد مباريات سابقة للمضيف لحساب الفورمة.", "WARNING")
    if away_form_count == 0:
        log("  ⚠ لم تُوجد مباريات سابقة للضيف لحساب الفورمة.", "WARNING")

    # --- تحويل إلى DataFrame ---
    features_dict_for_df = {k: [v] for k, v in features_values.items()}
    features_df = pd.DataFrame(features_dict_for_df)

    # التحقق من ترتيب الأعمدة (يجب أن يطابق ترتيب التدريب)
    expected_cols = EXPECTED_FEATURES
    actual_cols = list(features_df.columns)

    if actual_cols != expected_cols:
        log(
            f"  ⚠ ترتيب الأعمدة لا يطابق المتوقع. إعادة ترتيب...",
            "WARNING"
        )
        # إعادة ترتيب الأعمدة
        missing_cols = [c for c in expected_cols if c not in actual_cols]
        if missing_cols:
            log(f"  ❌ أعمدة مفقودة: {missing_cols}", "ERROR")
            return None, None

        features_df = features_df[expected_cols]

    return features_df, features_values


def extract_probabilities(
    model: Any,
    features_df: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """
    استخراج احتمالات التنبؤ من نموذج XGBoost.

    يستخدم model.classes_ بدلاً من LabelEncoder لضمان
    التوافق التام مع ترتيب الفئات الذي تدرّب عليه النموذج.

    المعاملات:
        model: نموذج XGBoost المدرّب
        features_df: DataFrame الميزات (صف واحد)

    العائد:
        قاموس يحتوي على:
        - home_win: احتمال فوز المضيف
        - draw: احتمال التعادل
        - away_win: احتمال فوز الضيف
        - model_classes: فئات النموذج
        أو None في حالة الفشل
    """
    log("جارٍ إجراء التنبؤ...", "INFO")

    try:
        # التنبؤ بالاحتمالات
        predicted_probabilities = model.predict_proba(features_df)

        # التحقق من صحة النتائج
        if predicted_probabilities is None or len(predicted_probabilities) == 0:
            log("فشل النموذج في إنتاج احتمالات.", "ERROR")
            return None

        proba = predicted_probabilities[0]

        # استخراج فئات النموذج
        model_classes = list(model.classes_)
        log(f"  فئات النموذج (model.classes_): {model_classes}", "DEBUG")
        log(f"  الاحتمالات الخام: {[round(float(p), 4) for p in proba]}", "DEBUG")

        # التحقق من وجود جميع الفئات المتوقعة
        for expected_class in EXPECTED_CLASSES:
            if expected_class not in model_classes:
                log(
                    f"  ⚠ الفئة {expected_class} غير موجودة في model.classes_. "
                    f"فئات النموذج: {model_classes}",
                    "WARNING"
                )

        # استخراج الاحتمالات بناءً على model.classes_
        prob_dict = {}

        for class_value, class_label_key in [
            (-1, "away_win"),
            (0, "draw"),
            (1, "home_win"),
        ]:
            if class_value in model_classes:
                idx = model_classes.index(class_value)
                prob_dict[class_label_key] = float(proba[idx])
            else:
                # الفئة غير موجودة — نعطيها احتمال 0
                log(
                    f"  ⚠ الفئة {class_value} ({class_label_key}) "
                    f"غير موجودة في النموذج. سيتم تعيين الاحتمال = 0.",
                    "WARNING"
                )
                prob_dict[class_label_key] = 0.0

        prob_dict["model_classes"] = model_classes

        # التحقق من أن المجموع قريب من 1.0
        total = prob_dict["home_win"] + prob_dict["draw"] + prob_dict["away_win"]
        if abs(total - 1.0) > 0.01:
            log(
                f"  ⚠ مجموع الاحتمالات ({total:.4f}) بعيد عن 1.0",
                "WARNING"
            )

        return prob_dict

    except Exception as e:
        log(f"فشل التنبؤ: {e}", "ERROR")
        traceback.print_exc()
        return None


# -----------------------------------------------------------------------------
# القسم الرابع: عرض النتائج
# -----------------------------------------------------------------------------

def determine_prediction(prob_dict: Dict[str, float]) -> Tuple[str, float]:
    """
    تحديد التنبؤ الأرجح ومستوى الثقة.

    المعاملات:
        prob_dict: قاموس الاحتمالات

    العائد:
        tuple يحتوي على (وصف التنبؤ, مستوى الثقة)
    """
    home_win = prob_dict.get("home_win", 0.0)
    draw = prob_dict.get("draw", 0.0)
    away_win = prob_dict.get("away_win", 0.0)

    max_prob = max(home_win, draw, away_win)

    if max_prob == home_win:
        return "فوز المضيف", home_win
    elif max_prob == draw:
        return "تعادل", draw
    else:
        return "فوز الضيف", away_win


def display_results(
    home_name: str,
    away_name: str,
    home_team_id: int,
    away_team_id: int,
    competition_code: str,
    season_key: str,
    prob_dict: Dict[str, float],
    features_values: Dict[str, float],
):
    """
    عرض نتائج التنبؤ بشكل منسّق.

    المعاملات:
        home_name: اسم الفريق المضيف
        away_name: اسم الفريق الضيف
        home_team_id: معرّف المضيف
        away_team_id: معرّف الضيف
        competition_code: رمز المسابقة
        season_key: مفتاح الموسم
        prob_dict: قاموس الاحتمالات
        features_values: قاموس قيم الميزات
    """
    home_win = prob_dict.get("home_win", 0.0)
    draw = prob_dict.get("draw", 0.0)
    away_win = prob_dict.get("away_win", 0.0)

    prediction_text, confidence = determine_prediction(prob_dict)

    # شريط مرئي بسيط
    def bar(prob: float, width: int = 30) -> str:
        filled = int(prob * width)
        return "█" * filled + "░" * (width - filled)

    print("")
    print("=" * 60)
    print(f"  📊 نتائج التنبؤ (XGBoost)")
    print("=" * 60)
    print(f"  المباراة : {home_name} vs {away_name}")
    print(f"  المعرّفات: {home_team_id} vs {away_team_id}")
    print(f"  المسابقة: {competition_code} | الموسم: {season_key}")
    print("-" * 60)
    print(f"  فوز المضيف : {home_win:6.2%}  {bar(home_win)}")
    print(f"  تعادل       : {draw:6.2%}  {bar(draw)}")
    print(f"  فوز الضيف  : {away_win:6.2%}  {bar(away_win)}")
    print("-" * 60)
    print(f"  🏆 التنبؤ: {prediction_text} (ثقة: {confidence:.1%})")
    print("=" * 60)

    # عرض الميزات
    print("")
    print("  الميزات المُستخدمة:")
    print("  " + "-" * 40)
    for feat_name, feat_value in features_values.items():
        print(f"    {feat_name:20s}: {feat_value:.4f}")
    print("  " + "-" * 40)
    print("")


# -----------------------------------------------------------------------------
# القسم الخامس: حفظ النتائج
# -----------------------------------------------------------------------------

def build_result_dict(
    home_name: str,
    away_name: str,
    home_team_id: int,
    away_team_id: int,
    competition_code: str,
    season_start_year: int,
    prob_dict: Dict[str, float],
    features_values: Dict[str, float],
    form_matches: int,
) -> Dict[str, Any]:
    """
    بناء قاموس النتيجة الكاملة للحفظ أو العرض.

    المعاملات:
        home_name: اسم المضيف
        away_name: اسم الضيف
        home_team_id: معرّف المضيف
        away_team_id: معرّف الضيف
        competition_code: رمز المسابقة
        season_start_year: سنة بداية الموسم
        prob_dict: قاموس الاحتمالات
        features_values: قاموس قيم الميزات
        form_matches: عدد مباريات الفورمة

    العائد:
        قاموس النتيجة الكاملة
    """
    prediction_text, confidence = determine_prediction(prob_dict)

    result = {
        "meta": {
            "version": getattr(config, "VERSION", "N/A"),
            "model": "XGBClassifier",
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "season_key": f"{competition_code}_{season_start_year}",
            "model_classes": prob_dict.get("model_classes", []),
            "form_matches": form_matches,
        },
        "match": f"{home_name} (Home) vs {away_name} (Away)",
        "competition": competition_code,
        "teams": {
            "home": {"name": home_name, "id": home_team_id},
            "away": {"name": away_name, "id": away_team_id},
        },
        "probabilities": {
            "home_win": round(prob_dict.get("home_win", 0.0), 6),
            "draw": round(prob_dict.get("draw", 0.0), 6),
            "away_win": round(prob_dict.get("away_win", 0.0), 6),
        },
        "prediction": {
            "result": prediction_text,
            "confidence": round(confidence, 6),
        },
        "features_used": {
            k: round(v, 6) for k, v in features_values.items()
        },
    }

    return result


def save_prediction_result(
    result: Dict[str, Any],
    output_path: Path,
) -> bool:
    """
    حفظ نتيجة التنبؤ في ملف JSON.

    المعاملات:
        result: قاموس النتيجة
        output_path: مسار ملف الحفظ

    العائد:
        True إذا تم الحفظ بنجاح
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        log(f"تم حفظ نتيجة التنبؤ في: {output_path}", "INFO")
        return True

    except Exception as e:
        log(f"فشل حفظ نتيجة التنبؤ: {e}", "ERROR")
        return False


# -----------------------------------------------------------------------------
# القسم السادس: الدالة الرئيسية
# -----------------------------------------------------------------------------

def predict_match(
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    home_team_name: Optional[str] = None,
    away_team_name: Optional[str] = None,
    competition_code: str = DEFAULT_COMP_CODE,
    season_start_year: Optional[int] = None,
    form_matches: int = DEFAULT_FORM_MATCHES,
    save: bool = False,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
):
    """
    الدالة الرئيسية للتنبؤ بنتيجة مباراة واحدة.

    يمكن تحديد الفرق بالمعرّف (--home, --away) أو بالاسم
    (--home-name, --away-name). إذا استُخدم الاسم، يتم البحث
    في teams.json عن المعرّف المقابل.

    المعاملات:
        home_team_id: معرّف الفريق المضيف (اختياري إذا حُدد الاسم)
        away_team_id: معرّف الفريق الضيف (اختياري إذا حُدد الاسم)
        home_team_name: اسم الفريق المضيف (اختياري)
        away_team_name: اسم الفريق الضيف (اختياري)
        competition_code: رمز المسابقة
        season_start_year: سنة بداية الموسم (None = تلقائي)
        form_matches: عدد مباريات الفورمة
        save: حفظ النتيجة في ملف JSON
        output_path: مسار ملف الحفظ (اختياري)
        dry_run: تشغيل بدون تنبؤ فعلي
    """
    start_time = datetime.now(timezone.utc)

    log("=" * 60, "INFO")
    log("بدء التنبؤ بنتيجة المباراة (XGBoost)", "INFO")
    log(f"الوقت: {start_time.isoformat()}", "INFO")
    log("=" * 60, "INFO")

    # =========================================================================
    # 1. التحقق من المتطلبات
    # =========================================================================
    if not check_dependencies():
        return

    # =========================================================================
    # 2. تحميل البيانات والنماذج
    # =========================================================================
    log("--- تحميل البيانات والنماذج ---", "INFO")

    data = load_all_prediction_data()
    if data is None:
        return

    model = data["model"]
    all_matches = data["all_matches"]
    team_factors = data["team_factors"]
    elo_ratings = data["elo_ratings"]
    teams_map = data.get("teams_map")
    model_metadata = data.get("model_metadata")

    # التحقق من توافق الميزات (إذا كانت البيانات الوصفية متوفرة)
    if model_metadata:
        expected_features = model_metadata.get("features", EXPECTED_FEATURES)
        if expected_features != EXPECTED_FEATURES:
            log(
                f"⚠ الميزات في البيانات الوصفية ({expected_features}) "
                f"تختلف عن المتوقعة ({EXPECTED_FEATURES}). "
                f"تأكد من التوافق.",
                "WARNING"
            )

    # =========================================================================
    # 3. تحديد الفرق
    # =========================================================================
    log("--- تحديد الفرق ---", "INFO")

    # إذا حُدد الاسم بدلاً من المعرّف، نبحث عن المعرّف
    if home_team_id is None and home_team_name:
        home_team_id = find_team_id_by_name(
            teams_map, home_team_name, comp_code=competition_code
        )
        if home_team_id is None:
            log(
                f"لم يتم العثور على الفريق المضيف: '{home_team_name}'",
                "ERROR"
            )
            return

    if away_team_id is None and away_team_name:
        away_team_id = find_team_id_by_name(
            teams_map, away_team_name, comp_code=competition_code
        )
        if away_team_id is None:
            log(
                f"لم يتم العثور على الفريق الضيف: '{away_team_name}'",
                "ERROR"
            )
            return

    # التحقق من وجود معرّفات
    if home_team_id is None:
        log("لم يتم تحديد الفريق المضيف. استخدم --home أو --home-name.", "ERROR")
        return

    if away_team_id is None:
        log("لم يتم تحديد الفريق الضيف. استخدم --away أو --away-name.", "ERROR")
        return

    # التحقق من اختلاف الفريقين
    if home_team_id == away_team_id:
        log("لا يمكن التنبؤ بمباراة بين نفس الفريق.", "ERROR")
        return

    # الحصول على أسماء الفرق
    home_name = get_team_name(teams_map, home_team_id)
    away_name = get_team_name(teams_map, away_team_id)

    log(f"  المضيف: {home_name} (ID: {home_team_id})", "INFO")
    log(f"  الضيف: {away_name} (ID: {away_team_id})", "INFO")

    # =========================================================================
    # 4. تحديد الموسم
    # =========================================================================
    if season_start_year is None:
        season_start_year = get_current_season_year()
        log(f"  الموسم (تلقائي): {season_start_year}", "INFO")
    else:
        log(f"  الموسم (محدد): {season_start_year}", "INFO")

    season_key = f"{competition_code}_{season_start_year}"
    log(f"  مفتاح الموسم: {season_key}", "INFO")
    log(f"  المسابقة: {competition_code}", "INFO")

    # =========================================================================
    # 5. التشغيل الجاف
    # =========================================================================
    if dry_run:
        log("", "INFO")
        log("[DRY RUN] سيتم التنبؤ بالمباراة التالية:", "INFO")
        log(f"  {home_name} (ID: {home_team_id}) vs {away_name} (ID: {away_team_id})", "INFO")
        log(f"  المسابقة: {competition_code} | الموسم: {season_key}", "INFO")
        log(f"  مباريات الفورمة: {form_matches}", "INFO")
        log("[DRY RUN] لم يتم تنفيذ التنبؤ.", "INFO")
        return

    # =========================================================================
    # 6. بناء الميزات
    # =========================================================================
    log("--- بناء الميزات ---", "INFO")

    features_df, features_values = build_features(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        competition_code=competition_code,
        season_start_year=season_start_year,
        all_matches=all_matches,
        team_factors=team_factors,
        elo_ratings=elo_ratings,
        form_matches=form_matches,
    )

    if features_df is None or features_values is None:
        log("فشل بناء الميزات. لا يمكن إجراء التنبؤ.", "ERROR")
        return

    # =========================================================================
    # 7. إجراء التنبؤ
    # =========================================================================
    log("--- إجراء التنبؤ ---", "INFO")

    prob_dict = extract_probabilities(model, features_df)

    if prob_dict is None:
        log("فشل استخراج الاحتمالات. لا يمكن عرض النتائج.", "ERROR")
        return

    # =========================================================================
    # 8. عرض النتائج
    # =========================================================================
    display_results(
        home_name=home_name,
        away_name=away_name,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        competition_code=competition_code,
        season_key=season_key,
        prob_dict=prob_dict,
        features_values=features_values,
    )

    # =========================================================================
    # 9. حفظ النتائج (اختياري)
    # =========================================================================
    if save:
        log("--- حفظ النتائج ---", "INFO")

        result = build_result_dict(
            home_name=home_name,
            away_name=away_name,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            competition_code=competition_code,
            season_start_year=season_start_year,
            prob_dict=prob_dict,
            features_values=features_values,
            form_matches=form_matches,
        )

        if output_path is None:
            output_path = (
                config.DATA_DIR
                / f"prediction_{home_team_id}_vs_{away_team_id}_{competition_code}.json"
            )

        save_prediction_result(result, output_path)

        # طباعة JSON الكاملة
        print("")
        print("النتيجة الكاملة (JSON):")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("")

    # =========================================================================
    # ملخص
    # =========================================================================
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    log(f"اكتمل التنبؤ في {duration:.1f} ثانية.", "INFO")
    log("=" * 60, "INFO")


# -----------------------------------------------------------------------------
# نقطة الدخول
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="التنبؤ بنتيجة مباراة كرة قدم باستخدام نموذج XGBoost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  # تنبؤ باستخدام المعرّفات الافتراضية
  python 06_predict_ml.py

  # تحديد الفرق بالمعرّف
  python 06_predict_ml.py --home 65 --away 64 --comp PL --season 2025

  # تحديد الفرق بالاسم
  python 06_predict_ml.py --home-name "Manchester City" --away-name "Liverpool" --comp PL

  # حفظ النتيجة
  python 06_predict_ml.py --home 65 --away 64 --comp PL --save

  # حفظ في ملف محدد
  python 06_predict_ml.py --home 65 --away 64 --comp PL --save --output prediction.json

  # تشغيل تجريبي
  python 06_predict_ml.py --home 65 --away 64 --comp PL --dry-run

  # تغيير عدد مباريات الفورمة
  python 06_predict_ml.py --home 65 --away 64 --comp PL --form-matches 10
        """
    )

    # --- تحديد الفرق ---
    team_group = parser.add_argument_group("تحديد الفرق")

    team_group.add_argument(
        "--home",
        type=int,
        default=None,
        help=f"معرّف الفريق المضيف. (افتراضي: {DEFAULT_HOME_TEAM_ID})"
    )

    team_group.add_argument(
        "--away",
        type=int,
        default=None,
        help=f"معرّف الفريق الضيف. (افتراضي: {DEFAULT_AWAY_TEAM_ID})"
    )

    team_group.add_argument(
        "--home-name",
        type=str,
        default=None,
        help="اسم الفريق المضيف (بدلاً من المعرّف). يبحث في teams.json."
    )

    team_group.add_argument(
        "--away-name",
        type=str,
        default=None,
        help="اسم الفريق الضيف (بدلاً من المعرّف). يبحث في teams.json."
    )

    # --- إعدادات المسابقة ---
    comp_group = parser.add_argument_group("إعدادات المسابقة")

    comp_group.add_argument(
        "--comp",
        type=str,
        default=DEFAULT_COMP_CODE,
        help=f"رمز المسابقة (مثل: PL, PD, SA, BL1, FL1). (افتراضي: {DEFAULT_COMP_CODE})"
    )

    comp_group.add_argument(
        "--season",
        type=int,
        default=None,
        help="سنة بداية الموسم. (افتراضي: تلقائي بناءً على التاريخ الحالي)"
    )

    # --- إعدادات التنبؤ ---
    pred_group = parser.add_argument_group("إعدادات التنبؤ")

    pred_group.add_argument(
        "--form-matches",
        type=int,
        default=DEFAULT_FORM_MATCHES,
        help=f"عدد المباريات الأخيرة لحساب الفورمة. (افتراضي: {DEFAULT_FORM_MATCHES})"
    )

    # --- إعدادات الإخراج ---
    output_group = parser.add_argument_group("إعدادات الإخراج")

    output_group.add_argument(
        "--save",
        action="store_true",
        help="حفظ نتيجة التنبؤ في ملف JSON."
    )

    output_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="مسار ملف الحفظ (يُستخدم مع --save). (افتراضي: تلقائي)"
    )

    output_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="تشغيل تجريبي: عرض الإعدادات بدون تنفيذ التنبؤ."
    )

    args = parser.parse_args()

    # --- تحديد معرّفات الفرق ---
    # إذا لم يُحدد المعرّف ولا الاسم، نستخدم القيم الافتراضية
    home_id = args.home
    away_id = args.away
    home_name_arg = args.home_name
    away_name_arg = args.away_name

    # إذا لم يُحدد أي شيء، نستخدم الافتراضي
    if home_id is None and home_name_arg is None:
        home_id = DEFAULT_HOME_TEAM_ID
        log(
            f"لم يُحدد الفريق المضيف. استخدام الافتراضي: ID={DEFAULT_HOME_TEAM_ID}",
            "INFO"
        )

    if away_id is None and away_name_arg is None:
        away_id = DEFAULT_AWAY_TEAM_ID
        log(
            f"لم يُحدد الفريق الضيف. استخدام الافتراضي: ID={DEFAULT_AWAY_TEAM_ID}",
            "INFO"
        )

    # تحديد مسار الإخراج
    output_path = Path(args.output) if args.output else None

    # --- تشغيل التنبؤ ---
    try:
        predict_match(
            home_team_id=home_id,
            away_team_id=away_id,
            home_team_name=home_name_arg,
            away_team_name=away_name_arg,
            competition_code=args.comp,
            season_start_year=args.season,
            form_matches=args.form_matches,
            save=args.save,
            output_path=output_path,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("")
        log("تم إيقاف العملية بواسطة المستخدم (Ctrl+C).", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"خطأ غير متوقع: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)
