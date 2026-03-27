# 02_trainer.py
# -----------------------------------------------------------------------------
# الوصف:
# هذا السكريبت هو "العقل" الخاص بالمشروع. يقوم بتنفيذ الخطوات التالية:
# 1. تحميل بيانات المباريات الخام من matches.json.
# 2. تجميع المباريات حسب الموسم الحقيقي لكل دوري.
# 3. لكل موسم على حدة:
#    أ. البحث عن أفضل مجموعة من المعاملات (Hyperparameters) باستخدام تقييم
#       Log-Loss حقيقي على مجموعة تحقق منفصلة.
#    ب. تدريب النماذج الإحصائية النهائية على كامل بيانات الموسم باستخدام أفضل
#       المعاملات التي تم العثور عليها.
# 4. حفظ النماذج المدربة النهائية في مجلد `models/`.
#
# التحسينات:
# - إصلاح الاستيراد (from common import config بدلاً من from common.config)
# - إضافة دالة load_matches المفقودة
# - إكمال دالة save_models
# - إضافة حماية من القيم السالبة في مصفوفة الاحتمالات (Dixon-Coles tau)
# - إضافة نسخ احتياطية قبل الحفظ
# - حفظ بيانات وصفية (metadata) مع النماذج
# - إضافة تتبع التقدم في البحث عن المعاملات
# - معالجة أخطاء أكثر تفصيلاً
# - إضافة التحقق من صحة البيانات
# - دعم معاملات سطر الأوامر
# - إضافة تقرير تدريب مفصّل
#
# الاستخدام:
# python 02_trainer.py
# python 02_trainer.py --skip-tuning
# python 02_trainer.py --min-matches 30
# -----------------------------------------------------------------------------

import sys
import os
import json
import shutil
import argparse
import itertools
import traceback
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Set

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# مكتبات علمية
try:
    import numpy as np
except ImportError:
    print("خطأ: مكتبة numpy غير مثبتة. يرجى تثبيتها: pip install numpy")
    sys.exit(1)

try:
    from scipy.stats import poisson
except ImportError:
    print("خطأ: مكتبة scipy غير مثبتة. يرجى تثبيتها: pip install scipy")
    sys.exit(1)

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe

# استيراد دوال النمذجة
try:
    from common.modeling import (
        calculate_league_averages,
        build_team_factors,
        build_elo_ratings,
        fit_dc_rho_mle,
    )
except ImportError as e:
    log(f"فشل استيراد وحدة النمذجة (common.modeling): {e}", "CRITICAL")
    sys.exit(1)


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# الحد الأدنى لعدد المباريات في الموسم لقبول التدريب
DEFAULT_MIN_MATCHES_PER_SEASON = 50

# الحد الأدنى لعدد مباريات مجموعة التحقق لإجراء البحث
MIN_VALIDATION_SET_SIZE = 10

# نسبة تقسيم التدريب/التحقق للبحث عن المعاملات
TRAIN_VALIDATION_SPLIT = 0.8

# الحد الأقصى لعدد الأهداف في نموذج بواسون
MAX_GOALS = 10

# قيمة صغيرة لمنع log(0)
LOG_EPSILON = 1e-9

# قيمة Log-Loss الافتراضية عند الفشل
DEFAULT_BAD_LOGLOSS = 999.0

# اسم ملف النسخة الاحتياطية
BACKUP_SUFFIX = ".backup"

# عدد النسخ الاحتياطية التي يُحتفظ بها
MAX_BACKUPS = 3


# -----------------------------------------------------------------------------
# القسم الأول: دوال تحميل البيانات
# -----------------------------------------------------------------------------

def load_matches(matches_path: Path) -> List[Dict[str, Any]]:
    """
    تحميل بيانات المباريات من ملف JSON.

    المعاملات:
        matches_path: مسار ملف matches.json

    العائد:
        قائمة المباريات

    الاستثناءات:
        FileNotFoundError: إذا لم يكن الملف موجوداً
        json.JSONDecodeError: إذا كان الملف تالفاً
    """
    if not matches_path.exists():
        raise FileNotFoundError(
            f"ملف المباريات غير موجود: {matches_path}. "
            f"يرجى تشغيل 01_pipeline.py أولاً."
        )

    log(f"جارٍ تحميل المباريات من: {matches_path}", "INFO")

    with open(matches_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"تنسيق ملف المباريات غير متوقع: المتوقع قائمة، الموجود {type(data).__name__}")

    log(f"تم تحميل {len(data)} مباراة بنجاح.", "INFO")
    return data


def validate_match_for_training(match: Dict) -> bool:
    """
    التحقق من صلاحية مباراة واحدة للتدريب.
    يجب أن تحتوي على تاريخ ونتيجة وفريقين.

    المعاملات:
        match: بيانات المباراة

    العائد:
        True إذا كانت المباراة صالحة للتدريب
    """
    # يجب أن تكون قاموساً
    if not isinstance(match, dict):
        return False

    # يجب أن تحتوي على تاريخ
    utc_date = match.get("utcDate")
    if not utc_date:
        return False

    parsed_date = parse_date_safe(utc_date)
    if parsed_date is None:
        return False

    # يجب أن تحتوي على فريقين بمعرّفات
    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})

    if not isinstance(home_team, dict) or not isinstance(away_team, dict):
        return False

    if "id" not in home_team or "id" not in away_team:
        return False

    # يجب أن تحتوي على نتيجة (score)
    score = match.get("score", {})
    if not isinstance(score, dict):
        return False

    # يجب أن يكون لها فائز محدد (HOME_TEAM, DRAW, AWAY_TEAM)
    winner = score.get("winner")
    if winner not in ("HOME_TEAM", "DRAW", "AWAY_TEAM"):
        return False

    # يجب أن تكون المباراة منتهية
    status = match.get("status", "")
    if status not in ("FINISHED", ""):
        # بعض المباريات قد لا تحتوي على status ولكن لديها winner
        if not winner:
            return False

    return True


def filter_valid_matches(matches: List[Dict]) -> List[Dict]:
    """
    تصفية المباريات الصالحة للتدريب من قائمة المباريات.

    المعاملات:
        matches: قائمة جميع المباريات

    العائد:
        قائمة المباريات الصالحة للتدريب
    """
    valid = [m for m in matches if validate_match_for_training(m)]

    rejected_count = len(matches) - len(valid)
    if rejected_count > 0:
        log(
            f"تم تصفية {rejected_count} مباراة غير صالحة "
            f"(من {len(matches)} إلى {len(valid)})",
            "WARNING"
        )

    return valid


# -----------------------------------------------------------------------------
# القسم الثاني: دوال التجميع والتنظيم
# -----------------------------------------------------------------------------

def get_season_start_month() -> int:
    """
    الحصول على شهر بداية الموسم من الإعدادات.

    العائد:
        رقم الشهر (مثلاً 7 لشهر يوليو)
    """
    return getattr(config, "CURRENT_SEASON_START_MONTH", 7)


def determine_season_key(match: Dict) -> Optional[str]:
    """
    تحديد مفتاح الموسم لمباراة معيّنة بناءً على تاريخها ورمز مسابقتها.

    المعاملات:
        match: بيانات المباراة

    العائد:
        مفتاح الموسم (مثلاً "PL_2024") أو None إذا تعذّر التحديد
    """
    # قراءة التاريخ
    match_date = parse_date_safe(match.get("utcDate"))
    if match_date is None:
        return None

    # تحديد سنة بداية الموسم
    start_month = get_season_start_month()
    if match_date.month >= start_month:
        season_start_year = match_date.year
    else:
        season_start_year = match_date.year - 1

    # قراءة رمز المسابقة
    comp_code = match.get("competition", {}).get("code", "UNK")

    return f"{comp_code}_{season_start_year}"


def group_matches_by_season(matches: List[Dict]) -> Dict[str, List[Dict]]:
    """
    تجميع المباريات في قاموس حسب الموسم.
    كل مفتاح يمثّل "رمز_المسابقة_سنة_البداية" (مثلاً "PL_2024").

    المعاملات:
        matches: قائمة جميع المباريات

    العائد:
        قاموس: مفتاح الموسم -> قائمة المباريات
    """
    matches_by_season: Dict[str, List[Dict]] = defaultdict(list)

    skipped = 0
    for match in matches:
        season_key = determine_season_key(match)
        if season_key is None:
            skipped += 1
            continue
        matches_by_season[season_key].append(match)

    if skipped > 0:
        log(f"تم تخطي {skipped} مباراة لعدم إمكانية تحديد موسمها.", "WARNING")

    log(f"تم تجميع المباريات في {len(matches_by_season)} موسم فريد.", "INFO")

    # عرض تفاصيل كل موسم
    for key in sorted(matches_by_season.keys()):
        count = len(matches_by_season[key])
        log(f"  {key}: {count} مباراة", "INFO")

    return dict(matches_by_season)


def load_and_group_matches() -> Dict[str, List[Dict[str, Any]]]:
    """
    تحميل وتجميع المباريات من ملف matches.json.
    يقوم بتحميل البيانات، تصفية المباريات الصالحة، ثم تجميعها حسب الموسم.

    العائد:
        قاموس: مفتاح الموسم -> قائمة المباريات الصالحة

    الاستثناءات:
        FileNotFoundError: إذا لم يكن ملف المباريات موجوداً
    """
    log("جارٍ تحميل وتجميع بيانات المباريات...", "INFO")

    # تحميل المباريات
    matches_path = config.DATA_DIR / "matches.json"
    all_matches = load_matches(matches_path)

    # تصفية المباريات الصالحة
    valid_matches = filter_valid_matches(all_matches)

    if not valid_matches:
        raise ValueError("لا توجد مباريات صالحة للتدريب بعد التصفية.")

    # تجميع حسب الموسم
    matches_by_season = group_matches_by_season(valid_matches)

    return matches_by_season


# -----------------------------------------------------------------------------
# القسم الثالث: دوال التنبؤ باستخدام Dixon-Coles
# -----------------------------------------------------------------------------

def calculate_dc_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int = MAX_GOALS
) -> np.ndarray:
    """
    حساب مصفوفة احتمالات النتائج باستخدام نموذج Dixon-Coles.

    المعاملات:
        lambda_home: معدّل أهداف المضيف المتوقع
        lambda_away: معدّل أهداف الضيف المتوقع
        rho: معامل الارتباط (Dixon-Coles rho)
        max_goals: الحد الأقصى لعدد الأهداف

    العائد:
        مصفوفة numpy ثنائية الأبعاد تمثّل احتمالات النتائج.
        البعد الأول = أهداف المضيف، البعد الثاني = أهداف الضيف.
    """
    # التأكد من أن lambda قيم موجبة
    lambda_home = max(lambda_home, 0.001)
    lambda_away = max(lambda_away, 0.001)

    # حساب توزيع بواسون لكل فريق
    goals_range = np.arange(0, max_goals + 1)
    home_goals_pmf = poisson.pmf(goals_range, lambda_home)
    away_goals_pmf = poisson.pmf(goals_range, lambda_away)

    # بناء مصفوفة الاحتمالات المستقلة
    score_matrix = np.outer(home_goals_pmf, away_goals_pmf)

    # تطبيق معامل الارتباط rho (Dixon-Coles adjustment)
    # هذا يعدّل الاحتمالات للنتائج المنخفضة (0-0, 0-1, 1-0, 1-1)
    if rho != 0.0:
        tau = np.ones((max_goals + 1, max_goals + 1))

        # تعديل Dixon-Coles للنتائج المنخفضة
        tau[0, 0] = 1.0 - (lambda_home * lambda_away * rho)
        tau[0, 1] = 1.0 + (lambda_home * rho)
        tau[1, 0] = 1.0 + (lambda_away * rho)
        tau[1, 1] = 1.0 - rho

        # التأكد من عدم وجود قيم سالبة (حماية عددية)
        # يمكن أن تحدث قيم سالبة إذا كان rho كبيراً جداً
        tau = np.maximum(tau, 0.0)

        score_matrix = score_matrix * tau

    # التأكد من عدم وجود قيم سالبة في المصفوفة النهائية
    score_matrix = np.maximum(score_matrix, 0.0)

    return score_matrix


def extract_probabilities_from_matrix(
    score_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    استخراج احتمالات (فوز المضيف، تعادل، فوز الضيف) من مصفوفة النتائج.

    المعاملات:
        score_matrix: مصفوفة احتمالات النتائج

    العائد:
        tuple يحتوي على (prob_home_win, prob_draw, prob_away_win)
        مجموعها يساوي 1.0 بالضبط بعد التطبيع.
    """
    # فوز المضيف: أهداف المضيف > أهداف الضيف (المثلث السفلي)
    prob_home_win = float(np.sum(np.tril(score_matrix, -1)))

    # تعادل: أهداف المضيف = أهداف الضيف (القطر)
    prob_draw = float(np.sum(np.diag(score_matrix)))

    # فوز الضيف: أهداف الضيف > أهداف المضيف (المثلث العلوي)
    prob_away_win = float(np.sum(np.triu(score_matrix, 1)))

    # تطبيع الاحتمالات لضمان مجموعها = 1.0
    total_prob = prob_home_win + prob_draw + prob_away_win

    if total_prob <= 0.0:
        # حالة نادرة جداً: جميع الاحتمالات صفرية
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    prob_home_win /= total_prob
    prob_draw /= total_prob
    prob_away_win /= total_prob

    return (prob_home_win, prob_draw, prob_away_win)


def predict_match_probabilities(
    match: Dict,
    factors_attack: Dict,
    factors_defense: Dict,
    league_avgs: Dict,
    rho: float
) -> Tuple[float, float, float]:
    """
    التنبؤ باحتمالات (فوز المضيف، تعادل، فوز الضيف) لمباراة واحدة
    باستخدام نموذج Dixon-Coles.

    المعاملات:
        match: بيانات المباراة
        factors_attack: عوامل الهجوم لكل فريق
        factors_defense: عوامل الدفاع لكل فريق
        league_avgs: متوسطات الدوري
        rho: معامل الارتباط Dixon-Coles

    العائد:
        tuple يحتوي على (prob_home_win, prob_draw, prob_away_win)
    """
    # استخراج معرّفات الفرق — نحاول ID أولاً ثم الاسم
    home_team_data = match.get("homeTeam", {})
    away_team_data = match.get("awayTeam", {})

    # نستخدم المعرّف (id) كمفتاح أساسي، وإن لم يوجد نستخدم الاسم
    home_key = str(home_team_data.get("id", home_team_data.get("name", "unknown_home")))
    away_key = str(away_team_data.get("id", away_team_data.get("name", "unknown_away")))

    # استخراج متوسطات الدوري
    avg_home_goals = league_avgs.get("avg_home_goals", 1.5)
    avg_away_goals = league_avgs.get("avg_away_goals", 1.2)

    # استخراج عوامل الفرق — القيمة الافتراضية 1.0 للفرق غير المعروفة
    home_attack = factors_attack.get(home_key, 1.0)
    away_defense = factors_defense.get(away_key, 1.0)
    away_attack = factors_attack.get(away_key, 1.0)
    home_defense = factors_defense.get(home_key, 1.0)

    # حساب معدّلات الأهداف المتوقعة (λ)
    lambda_home = home_attack * away_defense * avg_home_goals
    lambda_away = away_attack * home_defense * avg_away_goals

    # حساب مصفوفة الاحتمالات
    score_matrix = calculate_dc_score_matrix(lambda_home, lambda_away, rho)

    # استخراج الاحتمالات النهائية
    probabilities = extract_probabilities_from_matrix(score_matrix)

    return probabilities


# -----------------------------------------------------------------------------
# القسم الرابع: دوال التقييم (Log-Loss)
# -----------------------------------------------------------------------------

def get_actual_result_index(match: Dict) -> Optional[int]:
    """
    تحديد مؤشر النتيجة الفعلية للمباراة.

    المعاملات:
        match: بيانات المباراة

    العائد:
        0 لفوز المضيف، 1 للتعادل، 2 لفوز الضيف، أو None إذا غير محدد
    """
    winner = match.get("score", {}).get("winner")

    if winner == "HOME_TEAM":
        return 0
    elif winner == "DRAW":
        return 1
    elif winner == "AWAY_TEAM":
        return 2
    else:
        return None


def calculate_logloss(
    predictions: List[Tuple[float, float, float]],
    validation_set: List[Dict]
) -> float:
    """
    حساب Log-Loss لمجموعة من التنبؤات والنتائج الفعلية.

    Log-Loss = -1/N * Σ log(p_actual)

    حيث p_actual هو الاحتمال الذي أعطاه النموذج للنتيجة الفعلية.
    القيمة الأقل تعني أداءً أفضل.

    المعاملات:
        predictions: قائمة التنبؤات (كل عنصر هو tuple من 3 احتمالات)
        validation_set: قائمة المباريات الفعلية

    العائد:
        قيمة Log-Loss (الأقل أفضل)
    """
    if len(predictions) != len(validation_set):
        log(
            f"تحذير: عدد التنبؤات ({len(predictions)}) "
            f"لا يطابق عدد المباريات ({len(validation_set)})",
            "WARNING"
        )
        return DEFAULT_BAD_LOGLOSS

    log_losses = []

    for i, match in enumerate(validation_set):
        # تحديد النتيجة الفعلية
        result_index = get_actual_result_index(match)

        if result_index is None:
            # تجاهل المباريات بدون نتيجة محددة
            continue

        # استخراج الاحتمال الذي أعطاه النموذج للنتيجة الفعلية
        probs = predictions[i]
        prob_actual = probs[result_index]

        # حساب log-loss مع حماية من log(0)
        log_loss_value = np.log(max(prob_actual, LOG_EPSILON))
        log_losses.append(log_loss_value)

    if not log_losses:
        # لا توجد مباريات صالحة للتقييم
        return DEFAULT_BAD_LOGLOSS

    # Log-Loss = -mean(log(p_actual))
    return float(-np.mean(log_losses))


# -----------------------------------------------------------------------------
# القسم الخامس: البحث عن أفضل المعاملات (Hyperparameter Tuning)
# -----------------------------------------------------------------------------

def get_hyperparam_grid() -> Dict[str, List]:
    """
    الحصول على شبكة البحث عن المعاملات من الإعدادات.

    العائد:
        قاموس: اسم المعامل -> قائمة القيم الممكنة
    """
    grid = getattr(config, "HYPERPARAM_GRID", None)

    if grid is None or not isinstance(grid, dict) or not grid:
        # شبكة افتراضية في حالة عدم وجودها في الإعدادات
        log("لم يتم العثور على HYPERPARAM_GRID في الإعدادات. استخدام القيم الافتراضية.", "WARNING")
        grid = {
            "TEAM_FACTORS_HALFLIFE_DAYS": [180],
            "TEAM_FACTORS_PRIOR_GLOBAL": [0.5],
            "TEAM_FACTORS_DAMPING": [0.05],
            "TEAM_FACTORS_TEAM_PRIOR_WEIGHT": [0.3],
            "DC_RHO_MAX": [0.15],
        }

    return grid


def get_default_params() -> Dict:
    """
    الحصول على المعاملات الافتراضية (أول قيمة من كل معامل في الشبكة).

    العائد:
        قاموس المعاملات الافتراضية
    """
    grid = get_hyperparam_grid()
    default_params = {}

    for param_name, param_values in grid.items():
        if param_values and len(param_values) > 0:
            default_params[param_name] = param_values[0]
        else:
            log(f"تحذير: المعامل {param_name} لا يحتوي على قيم.", "WARNING")

    return default_params


def generate_param_combinations(grid: Dict[str, List]) -> List[Dict]:
    """
    توليد جميع التركيبات الممكنة من شبكة المعاملات.

    المعاملات:
        grid: شبكة المعاملات

    العائد:
        قائمة من القواميس، كل قاموس يمثّل تركيبة واحدة
    """
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())

    combinations = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    return combinations


def split_train_validation(
    matches: List[Dict],
    split_ratio: float = TRAIN_VALIDATION_SPLIT
) -> Tuple[List[Dict], List[Dict]]:
    """
    تقسيم المباريات إلى مجموعتي تدريب وتحقق بناءً على الترتيب الزمني.
    المباريات الأقدم للتدريب والأحدث للتحقق.

    المعاملات:
        matches: قائمة المباريات (يجب أن تكون مرتبة زمنياً)
        split_ratio: نسبة التدريب (افتراضي 0.8)

    العائد:
        tuple يحتوي على (train_set, validation_set)
    """
    # ترتيب المباريات زمنياً
    sorted_matches = sorted(
        matches,
        key=lambda m: m.get("utcDate", "")
    )

    # نقطة التقسيم
    split_point = int(len(sorted_matches) * split_ratio)

    train_set = sorted_matches[:split_point]
    validation_set = sorted_matches[split_point:]

    return train_set, validation_set


def evaluate_params(
    params: Dict,
    train_set: List[Dict],
    validation_set: List[Dict],
    end_date_train: Optional[datetime],
    prev_factors: Dict
) -> float:
    """
    تقييم مجموعة معاملات واحدة عبر التدريب على مجموعة التدريب
    والتقييم على مجموعة التحقق.

    المعاملات:
        params: المعاملات المراد تقييمها
        train_set: مجموعة التدريب
        validation_set: مجموعة التحقق
        end_date_train: تاريخ نهاية مجموعة التدريب
        prev_factors: عوامل الموسم السابق (للبراير الهرمي)

    العائد:
        قيمة Log-Loss (الأقل أفضل)
    """
    try:
        # 1. تدريب نموذج مؤقت على مجموعة التدريب فقط
        league_avgs_train = calculate_league_averages(train_set)

        factors_attack_train, factors_defense_train = build_team_factors(
            train_set,
            league_avgs_train,
            end_date_train,
            decay_halflife_days=params.get("TEAM_FACTORS_HALFLIFE_DAYS", 180),
            prior_strength=params.get("TEAM_FACTORS_PRIOR_GLOBAL", 0.5),
            damping=params.get("TEAM_FACTORS_DAMPING", 0.05),
            prior_attack=prev_factors.get("attack"),
            prior_defense=prev_factors.get("defense"),
            team_prior_weight=params.get("TEAM_FACTORS_TEAM_PRIOR_WEIGHT", 0.3),
        )

        temp_rho = fit_dc_rho_mle(
            train_set,
            factors_attack_train,
            factors_defense_train,
            league_avgs_train,
            decay_halflife_days=params.get("TEAM_FACTORS_HALFLIFE_DAYS", 180),
            rho_max=params.get("DC_RHO_MAX", 0.15),
        )

        # 2. التنبؤ بنتائج مجموعة التحقق
        predictions = []
        for match in validation_set:
            probs = predict_match_probabilities(
                match,
                factors_attack_train,
                factors_defense_train,
                league_avgs_train,
                temp_rho
            )
            predictions.append(probs)

        # 3. حساب Log-Loss
        score = calculate_logloss(predictions, validation_set)

        return score

    except Exception as e:
        log(f"  خطأ أثناء تقييم المعاملات: {e}", "WARNING")
        return DEFAULT_BAD_LOGLOSS


def find_best_params_for_season(
    matches: List[Dict],
    prev_factors: Dict,
    season_key: str = ""
) -> Tuple[Dict, float]:
    """
    البحث عن أفضل مجموعة معاملات لموسم معين باستخدام تقييم Log-Loss حقيقي
    على مجموعة تحقق منفصلة.

    المعاملات:
        matches: جميع مباريات الموسم
        prev_factors: عوامل الموسم السابق (للبراير الهرمي)
        season_key: مفتاح الموسم (للتسجيل)

    العائد:
        tuple يحتوي على (أفضل المعاملات, أفضل قيمة Log-Loss)
    """
    log(f"بدء البحث عن أفضل المعاملات لـ {season_key}...", "INFO")

    # تقسيم البيانات إلى تدريب وتحقق
    train_set, validation_set = split_train_validation(matches)

    log(f"  مجموعة التدريب: {len(train_set)} مباراة", "INFO")
    log(f"  مجموعة التحقق: {len(validation_set)} مباراة", "INFO")

    # التحقق من كفاية مجموعة التحقق
    if len(validation_set) < MIN_VALIDATION_SET_SIZE:
        log(
            f"  مجموعة التحقق صغيرة جداً ({len(validation_set)} مباراة). "
            f"سيتم استخدام المعاملات الافتراضية.",
            "WARNING"
        )
        default_params = get_default_params()
        return default_params, DEFAULT_BAD_LOGLOSS

    # التحقق من كفاية مجموعة التدريب
    if len(train_set) < MIN_VALIDATION_SET_SIZE:
        log(
            f"  مجموعة التدريب صغيرة جداً ({len(train_set)} مباراة). "
            f"سيتم استخدام المعاملات الافتراضية.",
            "WARNING"
        )
        default_params = get_default_params()
        return default_params, DEFAULT_BAD_LOGLOSS

    # تحديد تاريخ نهاية مجموعة التدريب
    train_dates = []
    for m in train_set:
        dt = parse_date_safe(m.get("utcDate"))
        if dt is not None:
            train_dates.append(dt)

    if train_dates:
        end_date_train = max(train_dates)
    else:
        log("  لم يتم العثور على تواريخ صالحة في مجموعة التدريب.", "WARNING")
        default_params = get_default_params()
        return default_params, DEFAULT_BAD_LOGLOSS

    # توليد جميع تركيبات المعاملات
    grid = get_hyperparam_grid()
    param_combinations = generate_param_combinations(grid)

    total_combinations = len(param_combinations)
    log(f"  عدد التركيبات المطلوب تقييمها: {total_combinations}", "INFO")

    # البحث الشامل (Grid Search)
    best_params = None
    best_score = float("inf")

    for idx, params in enumerate(param_combinations, 1):
        # تقييم هذه التركيبة
        score = evaluate_params(
            params, train_set, validation_set,
            end_date_train, prev_factors
        )

        # تسجيل التقدم (كل 10 تركيبات أو آخر تركيبة)
        if idx % 10 == 0 or idx == total_combinations or idx == 1:
            log(
                f"  [{idx}/{total_combinations}] Log-Loss = {score:.4f}",
                "DEBUG"
            )

        # تحديث أفضل النتائج
        if score < best_score:
            best_score = score
            best_params = params.copy()

    # التحقق من وجود نتيجة
    if best_params is None:
        log("  لم يتم العثور على معاملات مناسبة. استخدام القيم الافتراضية.", "WARNING")
        best_params = get_default_params()

    log(
        f"  أفضل المعاملات لـ {season_key}: "
        f"Log-Loss = {best_score:.4f}",
        "INFO"
    )
    for param_name, param_value in best_params.items():
        log(f"    {param_name}: {param_value}", "INFO")

    return best_params, best_score


# -----------------------------------------------------------------------------
# القسم السادس: دوال التدريب الرئيسية
# -----------------------------------------------------------------------------

def parse_season_key(key: str) -> Tuple[str, int]:
    """
    تحليل مفتاح الموسم إلى رمز المسابقة وسنة البداية.

    المعاملات:
        key: مفتاح الموسم (مثلاً "PL_2024")

    العائد:
        tuple يحتوي على (رمز المسابقة, سنة البداية)
    """
    try:
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            comp_code = parts[0]
            year = int(parts[1])
            return comp_code, year
    except (ValueError, IndexError):
        pass

    log(
        f"فشل في تحليل مفتاح الموسم '{key}'. "
        f"سيتم استخدام ترتيب افتراضي.",
        "WARNING"
    )
    return key, 0


def get_season_end_date(matches: List[Dict]) -> Optional[datetime]:
    """
    تحديد تاريخ نهاية الموسم (آخر مباراة).

    المعاملات:
        matches: قائمة مباريات الموسم

    العائد:
        تاريخ آخر مباراة، أو None إذا لم تكن هناك تواريخ صالحة
    """
    dates = []
    for m in matches:
        dt = parse_date_safe(m.get("utcDate"))
        if dt is not None:
            dates.append(dt)

    if dates:
        return max(dates)

    return None


def train_season_model(
    season_key: str,
    matches: List[Dict],
    best_params: Dict,
    prev_factors: Dict
) -> Dict[str, Any]:
    """
    تدريب النموذج النهائي لموسم واحد على كامل بيانات الموسم
    باستخدام أفضل المعاملات.

    المعاملات:
        season_key: مفتاح الموسم
        matches: جميع مباريات الموسم
        best_params: أفضل المعاملات (من البحث أو الافتراضية)
        prev_factors: عوامل الموسم السابق (للبراير الهرمي)

    العائد:
        قاموس يحتوي على جميع مكونات النموذج المدرب:
        - league_averages
        - attack_factors
        - defense_factors
        - elo_ratings
        - rho
        - best_params
        - metadata
    """
    log(f"جارٍ تدريب النموذج النهائي لـ {season_key}...", "INFO")

    # تحديد تاريخ نهاية الموسم
    season_end_date = get_season_end_date(matches)
    if season_end_date is None:
        raise ValueError(f"لم يتم العثور على تواريخ صالحة في مباريات {season_key}")

    log(f"  تاريخ نهاية الموسم: {season_end_date}", "DEBUG")

    # 1. حساب متوسطات الدوري
    league_avgs = calculate_league_averages(matches)
    log(
        f"  متوسطات الدوري: "
        f"أهداف المضيف = {league_avgs.get('avg_home_goals', 'N/A'):.3f}, "
        f"أهداف الضيف = {league_avgs.get('avg_away_goals', 'N/A'):.3f}",
        "INFO"
    )

    # 2. بناء عوامل الفرق
    factors_attack, factors_defense = build_team_factors(
        matches,
        league_avgs,
        season_end_date,
        decay_halflife_days=best_params.get("TEAM_FACTORS_HALFLIFE_DAYS", 180),
        prior_strength=best_params.get("TEAM_FACTORS_PRIOR_GLOBAL", 0.5),
        damping=best_params.get("TEAM_FACTORS_DAMPING", 0.05),
        prior_attack=prev_factors.get("attack"),
        prior_defense=prev_factors.get("defense"),
        team_prior_weight=best_params.get("TEAM_FACTORS_TEAM_PRIOR_WEIGHT", 0.3),
    )
    log(f"  عدد الفرق بعوامل هجومية: {len(factors_attack)}", "INFO")
    log(f"  عدد الفرق بعوامل دفاعية: {len(factors_defense)}", "INFO")

    # 3. بناء تقييمات ELO
    elo_k_base = getattr(config, "ELO_K_BASE", 20)
    elo_hfa = getattr(config, "ELO_HFA", 50)

    elo_ratings = build_elo_ratings(
        matches,
        k_base=elo_k_base,
        hfa_elo=elo_hfa
    )
    log(f"  عدد الفرق بتقييم ELO: {len(elo_ratings)}", "INFO")

    # 4. حساب معامل الارتباط (rho)
    rho = fit_dc_rho_mle(
        matches,
        factors_attack,
        factors_defense,
        league_avgs,
        decay_halflife_days=best_params.get("TEAM_FACTORS_HALFLIFE_DAYS", 180),
        rho_max=best_params.get("DC_RHO_MAX", 0.15),
    )
    log(f"  معامل الارتباط (rho): {rho:.6f}", "INFO")

    # 5. بيانات وصفية
    metadata = {
        "season_key": season_key,
        "num_matches": len(matches),
        "season_end_date": season_end_date.isoformat() if season_end_date else None,
        "num_teams_attack": len(factors_attack),
        "num_teams_defense": len(factors_defense),
        "num_teams_elo": len(elo_ratings),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "league_averages": league_avgs,
        "attack_factors": factors_attack,
        "defense_factors": factors_defense,
        "elo_ratings": elo_ratings,
        "rho": rho,
        "best_params": best_params,
        "metadata": metadata,
    }


def train_all_models(
    matches_by_season: Dict[str, List[Dict]],
    skip_tuning: bool = False,
    min_matches: int = DEFAULT_MIN_MATCHES_PER_SEASON
) -> Dict[str, Dict[str, Any]]:
    """
    المنسق الرئيسي: يدير عملية التدريب الكاملة لكل المواسم.

    المعاملات:
        matches_by_season: قاموس المباريات مُجمّعة حسب الموسم
        skip_tuning: إذا True، يتم تخطي البحث عن المعاملات واستخدام الافتراضية
        min_matches: الحد الأدنى لعدد المباريات المقبولة لكل موسم

    العائد:
        قاموس يحتوي على جميع النماذج المدربة مُنظّمة حسب النوع
    """
    log("=" * 70, "INFO")
    log("بدء عملية التدريب لجميع المواسم", "INFO")
    log("=" * 70, "INFO")

    # بنية البيانات النهائية
    trained_models = {
        "team_factors": {},
        "elo_ratings": {},
        "league_averages": {},
        "rho_values": {},
        "best_params": {},
        "training_metadata": {},
    }

    # ترتيب المواسم زمنياً (مهم للبراير الهرمي)
    sorted_seasons = sorted(
        matches_by_season.items(),
        key=lambda kv: parse_season_key(kv[0])
    )

    log(f"عدد المواسم: {len(sorted_seasons)}", "INFO")
    log(
        f"الترتيب الزمني: {[k for k, _ in sorted_seasons]}",
        "INFO"
    )

    if skip_tuning:
        log("⚠ تم تخطي البحث عن المعاملات (--skip-tuning). سيتم استخدام القيم الافتراضية.", "WARNING")

    # تتبع عوامل الموسم السابق لكل مسابقة (للبراير الهرمي)
    last_factors_by_comp: Dict[str, Dict] = defaultdict(dict)

    # عدادات
    trained_count = 0
    skipped_count = 0

    for season_key, matches in sorted_seasons:
        print("")
        log(f"{'=' * 50}", "INFO")
        log(f"الموسم: {season_key} ({len(matches)} مباراة)", "INFO")
        log(f"{'=' * 50}", "INFO")

        # التحقق من كفاية عدد المباريات
        if len(matches) < min_matches:
            log(
                f"تجاهل الموسم {season_key}: "
                f"عدد المباريات ({len(matches)}) أقل من الحد الأدنى ({min_matches}).",
                "WARNING"
            )
            skipped_count += 1
            continue

        # تحديد رمز المسابقة
        comp_code, season_year = parse_season_key(season_key)

        # استرجاع عوامل الموسم السابق (لنفس المسابقة)
        prev_factors = last_factors_by_comp.get(comp_code, {})

        if prev_factors:
            log(f"  استخدام عوامل الموسم السابق كـ prior لـ {comp_code}.", "INFO")
        else:
            log(f"  لا توجد عوامل سابقة لـ {comp_code} (أول موسم).", "INFO")

        try:
            # --- الخطوة 1: البحث عن أفضل المعاملات ---
            if skip_tuning:
                best_params = get_default_params()
                best_logloss = None
                log("  استخدام المعاملات الافتراضية (تم تخطي البحث).", "INFO")
            else:
                best_params, best_logloss = find_best_params_for_season(
                    matches, prev_factors, season_key
                )

            # --- الخطوة 2: تدريب النموذج النهائي ---
            season_model = train_season_model(
                season_key, matches, best_params, prev_factors
            )

            # --- الخطوة 3: حفظ مكونات النموذج ---
            trained_models["league_averages"][season_key] = season_model["league_averages"]

            trained_models["team_factors"][season_key] = {
                "attack": season_model["attack_factors"],
                "defense": season_model["defense_factors"],
            }

            trained_models["elo_ratings"][season_key] = season_model["elo_ratings"]

            trained_models["rho_values"][season_key] = season_model["rho"]

            trained_models["best_params"][season_key] = best_params

            trained_models["training_metadata"][season_key] = {
                **season_model["metadata"],
                "best_logloss": best_logloss,
                "tuning_skipped": skip_tuning,
            }

            # --- الخطوة 4: تحديث عوامل الموسم السابق ---
            last_factors_by_comp[comp_code] = {
                "attack": season_model["attack_factors"],
                "defense": season_model["defense_factors"],
            }

            trained_count += 1
            log(f"✅ اكتمل تدريب {season_key} بنجاح.", "INFO")

        except Exception as e:
            log(f"❌ فشل تدريب الموسم {season_key}: {e}", "ERROR")
            traceback.print_exc()
            skipped_count += 1
            continue

    # ملخص
    print("")
    log("=" * 70, "INFO")
    log("ملخص التدريب", "INFO")
    log(f"  تم تدريب: {trained_count} موسم", "INFO")
    log(f"  تم تخطي: {skipped_count} موسم", "INFO")
    log("=" * 70, "INFO")

    return trained_models


# -----------------------------------------------------------------------------
# القسم السابع: حفظ النماذج المدربة
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
        log(f"  نسخة احتياطية: {backup_path.name}", "DEBUG")
        return backup_path
    except OSError as e:
        log(f"  فشل إنشاء النسخة الاحتياطية: {e}", "WARNING")
        return None


def cleanup_old_backups(directory: Path, base_name: str, keep_last: int = MAX_BACKUPS):
    """
    حذف النسخ الاحتياطية القديمة والإبقاء على آخر N نسخة.

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


def save_json_safely(data: Any, file_path: Path, description: str) -> bool:
    """
    حفظ بيانات JSON بأمان عبر ملف مؤقت ثم إعادة تسمية.

    المعاملات:
        data: البيانات المراد حفظها
        file_path: مسار الملف النهائي
        description: وصف الملف (للتسجيل)

    العائد:
        True إذا تم الحفظ بنجاح
    """
    temp_path = file_path.with_suffix(".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)

        # التحقق من صحة الملف
        with open(temp_path, "r", encoding="utf-8") as f:
            json.load(f)

        # نقل الملف المؤقت ليحلّ محل النهائي
        temp_path.replace(file_path)

        file_size = file_path.stat().st_size
        log(f"  ✅ {description}: {file_path.name} ({file_size:,} بايت)", "INFO")
        return True

    except Exception as e:
        log(f"  ❌ فشل حفظ {description}: {e}", "ERROR")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def save_models(
    trained_models: Dict[str, Dict[str, Any]],
    create_backups_flag: bool = True
) -> bool:
    """
    حفظ جميع النماذج المدربة في ملفات JSON منفصلة.

    المعاملات:
        trained_models: قاموس يحتوي على جميع النماذج المدربة
        create_backups_flag: إنشاء نسخ احتياطية قبل الكتابة

    العائد:
        True إذا تم حفظ جميع الملفات بنجاح
    """
    log("", "INFO")
    log("=" * 70, "INFO")
    log("حفظ النماذج المدربة", "INFO")
    log("=" * 70, "INFO")

    # التأكد من وجود مجلد النماذج
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log(f"فشل في إنشاء مجلد النماذج: {e}", "CRITICAL")
        return False

    # قائمة الملفات المراد حفظها
    files_to_save = [
        (
            "league_averages.json",
            trained_models.get("league_averages", {}),
            "متوسطات الدوري",
        ),
        (
            "team_factors.json",
            trained_models.get("team_factors", {}),
            "عوامل الفرق",
        ),
        (
            "elo_ratings.json",
            trained_models.get("elo_ratings", {}),
            "تقييمات ELO",
        ),
        (
            "rho_values.json",
            trained_models.get("rho_values", {}),
            "قيم rho",
        ),
    ]

    all_saved = True

    for filename, data, description in files_to_save:
        file_path = config.MODELS_DIR / filename

        # إنشاء نسخة احتياطية
        if create_backups_flag:
            create_backup(file_path)
            cleanup_old_backups(config.MODELS_DIR, file_path.stem)

        # حفظ الملف
        saved = save_json_safely(data, file_path, description)
        if not saved:
            all_saved = False

    # حفظ بيانات وصفية إضافية (اختيارية)
    metadata_to_save = {
        "training_completed_at": datetime.now(timezone.utc).isoformat(),
        "version": getattr(config, "VERSION", "N/A"),
        "seasons_trained": list(trained_models.get("league_averages", {}).keys()),
        "best_params_per_season": trained_models.get("best_params", {}),
        "training_metadata": trained_models.get("training_metadata", {}),
    }

    metadata_path = config.MODELS_DIR / "training_metadata.json"
    save_json_safely(metadata_to_save, metadata_path, "البيانات الوصفية للتدريب")

    if all_saved:
        log("✅ تم حفظ جميع النماذج بنجاح.", "INFO")
    else:
        log("⚠ فشل حفظ بعض الملفات.", "WARNING")

    return all_saved


# -----------------------------------------------------------------------------
# القسم الثامن: التقييم النهائي (اختياري)
# -----------------------------------------------------------------------------

def evaluate_final_models(
    trained_models: Dict[str, Dict[str, Any]],
    matches_by_season: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    """
    تقييم النماذج المدربة على البيانات الكاملة لكل موسم
    (للحصول على تقدير تقريبي للأداء — ليس تقييماً غير متحيز).

    المعاملات:
        trained_models: النماذج المدربة
        matches_by_season: المباريات المُجمّعة حسب الموسم

    العائد:
        قاموس نتائج التقييم لكل موسم
    """
    log("", "INFO")
    log("تقييم النماذج المدربة (تقييم تقريبي على بيانات التدريب):", "INFO")

    evaluation_results = {}

    for season_key in sorted(trained_models.get("league_averages", {}).keys()):
        matches = matches_by_season.get(season_key, [])
        if not matches:
            continue

        league_avgs = trained_models["league_averages"].get(season_key, {})
        team_factors = trained_models["team_factors"].get(season_key, {})
        rho = trained_models["rho_values"].get(season_key, 0.0)

        factors_attack = team_factors.get("attack", {})
        factors_defense = team_factors.get("defense", {})

        # التنبؤ بجميع المباريات
        predictions = []
        for match in matches:
            probs = predict_match_probabilities(
                match, factors_attack, factors_defense, league_avgs, rho
            )
            predictions.append(probs)

        # حساب Log-Loss
        logloss = calculate_logloss(predictions, matches)

        # حساب الدقة (Accuracy)
        correct = 0
        total = 0
        for i, match in enumerate(matches):
            result_index = get_actual_result_index(match)
            if result_index is None:
                continue

            probs = predictions[i]
            predicted_index = int(np.argmax(probs))

            if predicted_index == result_index:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        evaluation_results[season_key] = {
            "logloss": round(float(logloss), 4),
            "accuracy": round(float(accuracy), 4),
            "num_matches": total,
        }

        log(
            f"  {season_key}: "
            f"Log-Loss = {logloss:.4f}, "
            f"Accuracy = {accuracy:.2%} "
            f"({total} مباراة)",
            "INFO"
        )

    return evaluation_results


# -----------------------------------------------------------------------------
# الدالة الرئيسية
# -----------------------------------------------------------------------------

def main(
    skip_tuning: bool = False,
    min_matches: int = DEFAULT_MIN_MATCHES_PER_SEASON,
    evaluate: bool = True,
    create_backups_flag: bool = True,
    dry_run: bool = False,
):
    """
    الدالة الرئيسية التي تنسق عملية تدريب النماذج.

    المعاملات:
        skip_tuning: تخطي البحث عن المعاملات
        min_matches: الحد الأدنى لعدد المباريات لقبول الموسم
        evaluate: تقييم النماذج بعد التدريب
        create_backups_flag: إنشاء نسخ احتياطية
        dry_run: تشغيل بدون حفظ
    """
    start_time = datetime.now(timezone.utc)

    log("=" * 70, "INFO")
    log("بدء عملية تدريب النماذج (Model Trainer)", "INFO")
    log(f"الوقت: {start_time.isoformat()}", "INFO")
    log(f"الإصدار: {getattr(config, 'VERSION', 'N/A')}", "INFO")
    log("=" * 70, "INFO")

    if dry_run:
        log("⚠ وضع التشغيل الجاف (Dry Run): لن يتم حفظ أي ملفات.", "WARNING")

    try:
        # =====================================================================
        # المرحلة 1: تحميل وتجميع البيانات
        # =====================================================================
        log("--- المرحلة 1: تحميل وتجميع البيانات ---", "INFO")
        matches_by_season = load_and_group_matches()

        if not matches_by_season:
            log("لا توجد مباريات مُجمّعة. لا يمكن المتابعة.", "CRITICAL")
            return

        # =====================================================================
        # المرحلة 2: تدريب النماذج
        # =====================================================================
        log("--- المرحلة 2: تدريب النماذج ---", "INFO")
        trained_models = train_all_models(
            matches_by_season,
            skip_tuning=skip_tuning,
            min_matches=min_matches,
        )

        # التحقق من وجود نماذج مدربة
        trained_seasons = list(trained_models.get("league_averages", {}).keys())
        if not trained_seasons:
            log("لم يتم تدريب أي نموذج. تحقق من البيانات والإعدادات.", "CRITICAL")
            return

        log(f"تم تدريب نماذج لـ {len(trained_seasons)} موسم: {trained_seasons}", "INFO")

        # =====================================================================
        # المرحلة 3: تقييم النماذج (اختياري)
        # =====================================================================
        if evaluate:
            log("--- المرحلة 3: تقييم النماذج ---", "INFO")
            evaluation_results = evaluate_final_models(trained_models, matches_by_season)

            # إضافة نتائج التقييم إلى البيانات الوصفية
            trained_models["evaluation"] = evaluation_results
        else:
            log("--- المرحلة 3: تخطي التقييم ---", "INFO")

        # =====================================================================
        # المرحلة 4: حفظ النماذج
        # =====================================================================
        if dry_run:
            log("--- المرحلة 4: تخطي الحفظ (وضع تجريبي) ---", "INFO")
            log(f"[DRY RUN] كان سيتم حفظ نماذج لـ {len(trained_seasons)} موسم.", "INFO")
        else:
            log("--- المرحلة 4: حفظ النماذج ---", "INFO")
            save_success = save_models(
                trained_models,
                create_backups_flag=create_backups_flag,
            )

            if not save_success:
                log("⚠ فشل حفظ بعض الملفات.", "WARNING")

    except FileNotFoundError as e:
        log(f"ملف مطلوب غير موجود: {e}", "CRITICAL")
        return

    except ValueError as e:
        log(f"خطأ في البيانات: {e}", "CRITICAL")
        return

    except Exception as e:
        log(f"حدث خطأ غير متوقع أثناء عملية التدريب: {e}", "CRITICAL")
        traceback.print_exc()
        return

    # =========================================================================
    # ملخص نهائي
    # =========================================================================
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    print("")
    log("=" * 70, "INFO")
    log("انتهت عملية تدريب النماذج بنجاح ✅", "INFO")
    log(f"المدة الإجمالية: {format_duration(duration)}", "INFO")
    log("=" * 70, "INFO")


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


# -----------------------------------------------------------------------------
# نقطة الدخول
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="تدريب النماذج الإحصائية (Dixon-Coles + ELO + Team Factors)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  python 02_trainer.py                      # تدريب كامل مع بحث عن المعاملات
  python 02_trainer.py --skip-tuning        # تدريب سريع بدون بحث
  python 02_trainer.py --min-matches 30     # تقليل الحد الأدنى للمباريات
  python 02_trainer.py --no-eval            # بدون تقييم بعد التدريب
  python 02_trainer.py --dry-run            # تشغيل تجريبي بدون حفظ
        """
    )

    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        default=False,
        help="تخطي البحث عن أفضل المعاملات (Hyperparameter Tuning) واستخدام القيم الافتراضية."
    )

    parser.add_argument(
        "--min-matches",
        type=int,
        default=DEFAULT_MIN_MATCHES_PER_SEASON,
        help=f"الحد الأدنى لعدد المباريات المطلوبة لقبول الموسم للتدريب. (افتراضي: {DEFAULT_MIN_MATCHES_PER_SEASON})"
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="تخطي تقييم النماذج بعد التدريب."
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
        help="تشغيل تجريبي: تنفيذ التدريب بدون حفظ أي ملفات."
    )

    args = parser.parse_args()

    try:
        main(
            skip_tuning=args.skip_tuning,
            min_matches=args.min_matches,
            evaluate=not args.no_eval,
            create_backups_flag=not args.no_backup,
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
