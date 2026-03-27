# 03_backtester.py
# -----------------------------------------------------------------------------
# الوصف:
# - باكتيستر زمني للموديل الإحصائي (Dixon–Coles + Team Factors + ELO).
# - يقيس LogLoss/Brier/ECE عبر مواسم/دوريات بنهج نافذة زمنية متوسعة
#   (expanding window).
# - توليف بسيط لمعاملات:
#     * TEAM_FACTORS_HALFLIFE_DAYS
#     * TEAM_FACTORS_PRIOR_GLOBAL
#     * TEAM_FACTORS_TEAM_PRIOR_WEIGHT
#     * DC_RHO_MAX
# - يطبع أفضل الإعدادات ويقترح قيمًا لتثبيتها في common.config.
#
# التحسينات:
# - إصلاح خلط احتمالات فوز المضيف/الضيف (triu vs tril)
# - إصلاح عدم تحديث prior الموسم السابق (prev_prior_by_comp)
# - إصلاح المتغيرات المحسوبة وغير المستخدمة في compute_lambdas_for_match
# - إضافة تجميع دقيق للاحتمالات والملصقات بدلاً من المعدل المرجّح فقط
# - حماية من القيم غير الصالحة (None, NaN, Inf)
# - إضافة نسخ احتياطية قبل الحفظ
# - إضافة تقرير مفصّل بعد انتهاء العملية
# - إضافة شريط تقدم وتسجيل أوضح
# - إضافة معالجة أخطاء تفصيلية
# - إضافة التحقق من صحة البيانات
# - إضافة دعم --dry-run
# - دعم حفظ تقرير نصي قابل للقراءة
#
# الاستخدام:
#   python 03_backtester.py --save
#   python 03_backtester.py --comps PL PD --use-elo --save
#   python 03_backtester.py --grid-halflife 90,180,365 --grid-prior-global 2.0,3.0 --save
#   python 03_backtester.py --min-train 120 --block-size 40 --limit-seasons 3 --save
# -----------------------------------------------------------------------------

import sys
import os
import json
import shutil
import argparse
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# مكتبات علمية
try:
    import numpy as np
except ImportError:
    print("خطأ: مكتبة numpy غير مثبتة. يرجى تثبيتها: pip install numpy")
    sys.exit(1)

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe, parse_score

# استيراد دوال النمذجة
try:
    from common.modeling import (
        calculate_league_averages,
        build_team_factors,
        build_elo_ratings,
        fit_dc_rho_mle,
        poisson_matrix_dc,
        suggest_goal_cutoff,
    )
except ImportError as e:
    log(f"فشل استيراد وحدة النمذجة (common.modeling): {e}", "CRITICAL")
    sys.exit(1)


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# الحد الأدنى لعدد المباريات في الموسم لقبوله في الباكتيست
DEFAULT_MIN_SEASON_MATCHES = 30

# الحد الأدنى لعدد مباريات التدريب قبل أول تقييم
DEFAULT_MIN_TRAIN = 120

# حجم كتلة الاختبار الافتراضي
DEFAULT_BLOCK_SIZE = 40

# عدد صناديق ECE الافتراضي
DEFAULT_ECE_BINS = 10

# قيمة صغيرة لمنع log(0)
LOG_EPSILON = 1e-15

# اسم ملف النسخة الاحتياطية
BACKUP_SUFFIX = ".backup"

# عدد النسخ الاحتياطية المحتفظ بها
MAX_BACKUPS = 3

# القيم الافتراضية لشبكة البحث
DEFAULT_GRID_HALFLIFE = [90, 180, 365]
DEFAULT_GRID_PRIOR_GLOBAL = [2.0, 3.0, 5.0]
DEFAULT_GRID_TEAM_PRIOR_WEIGHT = [0.0, 5.0]
DEFAULT_GRID_RHO_MAX = [0.15, 0.20]
DEFAULT_RHO_STEP = 0.002


# -----------------------------------------------------------------------------
# القسم الأول: أدوات تحميل البيانات وتجميعها
# -----------------------------------------------------------------------------

def load_matches(path: Path) -> List[Dict[str, Any]]:
    """
    تحميل بيانات المباريات من ملف JSON.

    المعاملات:
        path: مسار ملف matches.json

    العائد:
        قائمة المباريات

    الاستثناءات:
        FileNotFoundError: إذا لم يكن الملف موجوداً
        json.JSONDecodeError: إذا كان الملف تالفاً
    """
    if not path.exists():
        raise FileNotFoundError(
            f"ملف المباريات غير موجود: {path}. "
            f"يرجى تشغيل 01_pipeline.py أولاً."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"تنسيق ملف المباريات غير متوقع: المتوقع قائمة، الموجود {type(data).__name__}"
            )

        log(f"تم تحميل {len(data)} مباراة من: {path}", "INFO")
        return data

    except json.JSONDecodeError as e:
        log(f"خطأ في تحليل ملف المباريات: {e}", "CRITICAL")
        raise
    except Exception as e:
        log(f"تعذّر تحميل matches.json: {e}", "CRITICAL")
        raise


def group_matches_by_season(
    matches: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    تجميع المباريات في قاموس حسب الموسم.
    كل مفتاح يمثّل "رمز_المسابقة_سنة_البداية" (مثلاً "PL_2024").

    يتم استخدام season.startDate لتحديد سنة بداية الموسم.
    المباريات داخل كل موسم تُرتّب زمنياً.

    المعاملات:
        matches: قائمة جميع المباريات

    العائد:
        قاموس: مفتاح الموسم -> قائمة المباريات (مرتبة زمنياً)
    """
    by_season: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for m in matches:
        # استخراج رمز المسابقة
        competition = m.get("competition", {}) or {}
        comp_code = competition.get("code", "UNK")

        # استخراج سنة بداية الموسم
        season = m.get("season", {}) or {}
        start_date_str = season.get("startDate", "")

        if not start_date_str or len(start_date_str) < 4:
            skipped += 1
            continue

        season_year = start_date_str[:4]

        # بناء مفتاح الموسم
        key = f"{comp_code}_{season_year}"
        by_season[key].append(m)

    if skipped > 0:
        log(
            f"تم تخطي {skipped} مباراة لعدم وجود بيانات موسم صالحة.",
            "WARNING"
        )

    # ترتيب المباريات داخل كل موسم زمنياً
    default_date = parse_date_safe("1900-01-01")
    for key in by_season:
        by_season[key].sort(
            key=lambda x: parse_date_safe(x.get("utcDate")) or default_date
        )

    log(f"تم تجميع المباريات في {len(by_season)} موسم فريد.", "INFO")

    # عرض تفاصيل كل موسم
    for key in sorted(by_season.keys()):
        count = len(by_season[key])
        log(f"  {key}: {count} مباراة", "DEBUG")

    return dict(by_season)


def season_key_parts(sk: str) -> Tuple[str, int]:
    """
    تحليل مفتاح الموسم إلى رمز المسابقة وسنة البداية.

    المعاملات:
        sk: مفتاح الموسم (مثلاً "PL_2024")

    العائد:
        tuple يحتوي على (رمز المسابقة, سنة البداية)
    """
    try:
        parts = sk.rsplit("_", 1)
        if len(parts) == 2:
            comp = parts[0]
            yr = int(parts[1])
            return comp, yr
    except (ValueError, IndexError):
        pass

    return sk, 0


# -----------------------------------------------------------------------------
# القسم الثاني: أدوات تحليل سطر الأوامر
# -----------------------------------------------------------------------------

def parse_grid_list_floats(s: str) -> List[float]:
    """
    تحليل سلسلة نصية مفصولة بفواصل إلى قائمة أرقام عشرية.

    المعاملات:
        s: السلسلة النصية (مثلاً "2.0,3.0,5.0")

    العائد:
        قائمة أرقام عشرية
    """
    result = []
    for x in s.split(","):
        x = x.strip()
        if x:
            try:
                result.append(float(x))
            except ValueError:
                log(f"تحذير: قيمة غير صالحة في الشبكة: '{x}'", "WARNING")
    return result


def parse_grid_list_ints(s: str) -> List[int]:
    """
    تحليل سلسلة نصية مفصولة بفواصل إلى قائمة أعداد صحيحة.

    المعاملات:
        s: السلسلة النصية (مثلاً "90,180,365")

    العائد:
        قائمة أعداد صحيحة
    """
    result = []
    for x in s.split(","):
        x = x.strip()
        if x:
            try:
                result.append(int(x))
            except ValueError:
                log(f"تحذير: قيمة غير صالحة في الشبكة: '{x}'", "WARNING")
    return result


# -----------------------------------------------------------------------------
# القسم الثالث: تحديد نتيجة المباراة
# -----------------------------------------------------------------------------

def outcome_label(home_goals: int, away_goals: int) -> int:
    """
    تحديد تصنيف نتيجة المباراة بناءً على الأهداف.

    المعاملات:
        home_goals: أهداف المضيف
        away_goals: أهداف الضيف

    العائد:
        0 = فوز المضيف، 1 = تعادل، 2 = فوز الضيف
    """
    if home_goals > away_goals:
        return 0  # فوز المضيف
    elif home_goals == away_goals:
        return 1  # تعادل
    else:
        return 2  # فوز الضيف


# -----------------------------------------------------------------------------
# القسم الرابع: حساب مقاييس التقييم
# -----------------------------------------------------------------------------

def compute_metrics(
    probs: List[Tuple[float, float, float]],
    labels: List[int],
    ece_bins: int = DEFAULT_ECE_BINS
) -> Dict[str, Any]:
    """
    حساب مقاييس أداء النموذج: LogLoss, Brier Score, Accuracy, ECE.

    المعاملات:
        probs: قائمة التنبؤات (كل عنصر هو tuple من 3 احتمالات:
               فوز المضيف، تعادل، فوز الضيف)
        labels: قائمة التصنيفات الفعلية
               (0 = فوز المضيف، 1 = تعادل، 2 = فوز الضيف)
        ece_bins: عدد صناديق ECE

    العائد:
        قاموس يحتوي على المقاييس: n, logloss, brier, accuracy, ece
    """
    n = len(labels)

    if n == 0:
        return {
            "n": 0,
            "logloss": None,
            "brier": None,
            "accuracy": None,
            "ece": None,
        }

    if len(probs) != n:
        log(
            f"تحذير: عدد التنبؤات ({len(probs)}) لا يطابق عدد الملصقات ({n})",
            "WARNING"
        )
        n = min(len(probs), n)

    # متراكمات
    logloss_sum = 0.0
    brier_sum = 0.0
    correct = 0

    # صناديق ECE
    num_bins = max(1, ece_bins)
    bins: List[List[Tuple[bool, float]]] = [[] for _ in range(num_bins)]

    for i in range(n):
        prob_tuple = probs[i]
        y = labels[i]

        # استخراج الاحتمالات وضمان أنها موجبة
        ph = max(LOG_EPSILON, float(prob_tuple[0]))
        pd = max(LOG_EPSILON, float(prob_tuple[1]))
        pa = max(LOG_EPSILON, float(prob_tuple[2]))

        # تطبيع الاحتمالات
        p_arr = np.array([ph, pd, pa], dtype=np.float64)
        p_sum = p_arr.sum()
        if p_sum > 0:
            p_arr = p_arr / p_sum
        else:
            p_arr = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # --- LogLoss ---
        # LogLoss = -log(p_actual)
        logloss_sum += -math.log(max(float(p_arr[y]), LOG_EPSILON))

        # --- Brier Score (multi-class) ---
        # Brier = sum((p_i - y_i)^2) حيث y_i = 1 للفئة الصحيحة و0 لغيرها
        y_vec = np.zeros(3, dtype=np.float64)
        y_vec[y] = 1.0
        brier_sum += float(np.sum((p_arr - y_vec) ** 2))

        # --- Accuracy ---
        pred = int(np.argmax(p_arr))
        if pred == y:
            correct += 1

        # --- ECE ---
        # نستخدم أعلى ثقة (top-class confidence) لتحديد الصندوق
        conf = float(np.max(p_arr))
        bin_idx = min(num_bins - 1, int(conf * num_bins))
        is_correct = (pred == y)
        bins[bin_idx].append((is_correct, conf))

    # حساب ECE (Expected Calibration Error)
    ece_sum = 0.0
    for b in bins:
        if not b:
            continue
        # دقة الصندوق (accuracy)
        acc_bin = sum(1.0 if item[0] else 0.0 for item in b) / len(b)
        # متوسط الثقة في الصندوق
        conf_bin = sum(item[1] for item in b) / len(b)
        # مساهمة الصندوق في ECE (مرجّحة بحجم الصندوق)
        ece_sum += abs(acc_bin - conf_bin) * (len(b) / n)

    return {
        "n": n,
        "logloss": logloss_sum / n,
        "brier": brier_sum / n,
        "accuracy": correct / n,
        "ece": ece_sum,
    }


# -----------------------------------------------------------------------------
# القسم الخامس: حساب λ والتنبؤ بالاحتمالات
# -----------------------------------------------------------------------------

def compute_lambdas_for_match(
    match: Dict[str, Any],
    factors_attack: Dict[str, float],
    factors_defense: Dict[str, float],
    league_avgs: Dict[str, Any],
) -> Tuple[float, float]:
    """
    حساب معدّلات الأهداف المتوقعة (λ) لمباراة باستخدام نموذج Dixon-Coles.

    λ_home = avg_home_goals * A_home * D_away
    λ_away = avg_away_goals * A_away * D_home

    المعاملات:
        match: بيانات المباراة
        factors_attack: عوامل الهجوم لكل فريق (مفتاح = ID الفريق كنص)
        factors_defense: عوامل الدفاع لكل فريق (مفتاح = ID الفريق كنص)
        league_avgs: متوسطات الدوري

    العائد:
        tuple يحتوي على (lambda_home, lambda_away)
    """
    # استخراج معرّفات الفريقين
    home_id = str(match.get("homeTeam", {}).get("id", "unknown"))
    away_id = str(match.get("awayTeam", {}).get("id", "unknown"))

    # استخراج متوسطات الدوري
    avg_home_goals = float(league_avgs.get("avg_home_goals", 1.40))
    avg_away_goals = float(league_avgs.get("avg_away_goals", 1.10))

    # استخراج عوامل الهجوم والدفاع
    attack_home = float(factors_attack.get(home_id, 1.0))
    defense_home = float(factors_defense.get(home_id, 1.0))
    attack_away = float(factors_attack.get(away_id, 1.0))
    defense_away = float(factors_defense.get(away_id, 1.0))

    # حساب λ
    # λ_home = متوسط أهداف المضيف × هجوم المضيف × دفاع الضيف
    lambda_home = avg_home_goals * attack_home * defense_away

    # λ_away = متوسط أهداف الضيف × هجوم الضيف × دفاع المضيف
    lambda_away = avg_away_goals * attack_away * defense_home

    # ضمان قيم موجبة
    lambda_home = max(lambda_home, 1e-6)
    lambda_away = max(lambda_away, 1e-6)

    return lambda_home, lambda_away


def adjust_lambdas_with_elo(
    lambda_home: float,
    lambda_away: float,
    elo_home: float,
    elo_away: float
) -> Tuple[float, float]:
    """
    تعديل معدّلات الأهداف (λ) باستخدام فرق تقييمات ELO.

    يضيف عامل أفضلية الأرض (HFA) ثم يحسب عامل التعديل بناءً على
    الفرق في ELO.

    المعاملات:
        lambda_home: معدّل أهداف المضيف الأصلي
        lambda_away: معدّل أهداف الضيف الأصلي
        elo_home: تقييم ELO للمضيف
        elo_away: تقييم ELO للضيف

    العائد:
        tuple يحتوي على (lambda_home_adjusted, lambda_away_adjusted)
    """
    # استخراج إعدادات ELO من config
    elo_hfa = float(getattr(config, "ELO_HFA", 60.0))
    elo_lambda_scale = float(getattr(config, "ELO_LAMBDA_SCALE", 400.0))

    # حساب الأفضلية الإجمالية (فرق ELO + أفضلية الأرض)
    edge = (float(elo_home) - float(elo_away)) + elo_hfa

    # حساب عامل التعديل
    # factor > 1 يعني المضيف أقوى، factor < 1 يعني الضيف أقوى
    factor = 10.0 ** (edge / elo_lambda_scale)

    # تعديل λ
    adjusted_home = lambda_home * factor
    adjusted_away = max(1e-6, lambda_away / factor)

    return adjusted_home, adjusted_away


def predict_probs_for_match(
    match: Dict[str, Any],
    models: Dict[str, Any],
    use_elo: bool = True
) -> Tuple[float, float, float]:
    """
    التنبؤ باحتمالات (فوز المضيف، تعادل، فوز الضيف) لمباراة واحدة.

    يستخدم نموذج Dixon-Coles مع إمكانية تعديل بواسطة ELO.

    ملاحظة مهمة حول اتجاه المصفوفة:
    - mat[i, j] = P(أهداف المضيف = i، أهداف الضيف = j)
    - فوز المضيف: i > j → المثلث السفلي (tril مع k=-1)
    - تعادل: i = j → القطر (trace)
    - فوز الضيف: j > i → المثلث العلوي (triu مع k=1)

    المعاملات:
        match: بيانات المباراة
        models: قاموس يحتوي على مكونات النموذج
        use_elo: تفعيل تعديل ELO

    العائد:
        tuple يحتوي على (prob_home_win, prob_draw, prob_away_win)
    """
    # استخراج مكونات النموذج
    factors_attack = models.get("factors_A", {})
    factors_defense = models.get("factors_D", {})
    league_avgs = models.get("league_avgs", {})
    rho = float(models.get("rho", 0.0))
    elo_ratings = models.get("elo", {})

    # حساب λ الأساسي
    lambda_home, lambda_away = compute_lambdas_for_match(
        match, factors_attack, factors_defense, league_avgs
    )

    # تعديل بواسطة ELO إذا مطلوب
    if use_elo and elo_ratings:
        home_id = str(match.get("homeTeam", {}).get("id", "unknown"))
        away_id = str(match.get("awayTeam", {}).get("id", "unknown"))

        elo_start = float(getattr(config, "ELO_START", 1500.0))
        elo_home = float(elo_ratings.get(home_id, elo_start))
        elo_away = float(elo_ratings.get(away_id, elo_start))

        lambda_home, lambda_away = adjust_lambdas_with_elo(
            lambda_home, lambda_away, elo_home, elo_away
        )

    # حساب الحد الأقصى للأهداف
    goal_cutoff = suggest_goal_cutoff(lambda_home, lambda_away)

    # حساب مصفوفة الاحتمالات (Dixon-Coles)
    mat = poisson_matrix_dc(lambda_home, lambda_away, rho, max_goals=goal_cutoff)

    # استخراج الاحتمالات الثلاثة
    # mat[i, j] = P(home=i, away=j)
    # فوز المضيف: أهداف المضيف > أهداف الضيف → المثلث السفلي
    prob_home_win = float(np.tril(mat, k=-1).sum())

    # تعادل: أهداف المضيف = أهداف الضيف → القطر
    prob_draw = float(np.trace(mat))

    # فوز الضيف: أهداف الضيف > أهداف المضيف → المثلث العلوي
    prob_away_win = float(np.triu(mat, k=1).sum())

    # تطبيع الاحتمالات
    total = prob_home_win + prob_draw + prob_away_win
    if total > 0:
        prob_home_win /= total
        prob_draw /= total
        prob_away_win /= total
    else:
        # حالة نادرة جداً
        prob_home_win = 1.0 / 3.0
        prob_draw = 1.0 / 3.0
        prob_away_win = 1.0 / 3.0

    return prob_home_win, prob_draw, prob_away_win


# -----------------------------------------------------------------------------
# القسم السادس: تدريب النماذج لنافذة زمنية
# -----------------------------------------------------------------------------

def get_end_date_from_matches(matches: List[Dict]) -> Optional[datetime]:
    """
    استخراج تاريخ آخر مباراة من قائمة مباريات.

    المعاملات:
        matches: قائمة المباريات

    العائد:
        تاريخ آخر مباراة، أو None
    """
    dates = []
    for m in matches:
        dt = parse_date_safe(m.get("utcDate"))
        if dt is not None:
            dates.append(dt)

    if dates:
        return max(dates)

    return None


def train_models_for_window(
    train_matches: List[Dict[str, Any]],
    halflife_days: int,
    prior_global: float,
    team_prior_weight: float,
    rho_max: float,
    rho_step: float,
    damping: float = 0.5,
    prev_season_prior: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    تدريب مجموعة كاملة من النماذج على نافذة تدريب معيّنة.

    المعاملات:
        train_matches: مباريات التدريب (مرتبة زمنياً)
        halflife_days: نصف عمر الانحلال الزمني بالأيام
        prior_global: قوة الانكماش نحو 1.0 (Gamma prior)
        team_prior_weight: وزن الانكماش الهرمي نحو الموسم السابق
        rho_max: الحد الأقصى لقيمة |rho|
        rho_step: دقة شبكة البحث عن rho
        damping: معامل التخميد لعوامل الفرق
        prev_season_prior: عوامل الموسم السابق (للبراير الهرمي)

    العائد:
        قاموس يحتوي على مكونات النموذج، أو None في حالة الفشل
    """
    if not train_matches:
        return None

    try:
        # تحديد تاريخ نهاية نافذة التدريب
        season_end_date = get_end_date_from_matches(train_matches)

        # حساب متوسطات الدوري
        league_avgs = calculate_league_averages(train_matches)

        # استخراج عوامل الموسم السابق (إن وُجدت)
        prior_attack = None
        prior_defense = None
        if prev_season_prior:
            prior_attack = prev_season_prior.get("attack")
            prior_defense = prev_season_prior.get("defense")

        # بناء عوامل الفرق
        factors_attack, factors_defense = build_team_factors(
            train_matches,
            league_avgs,
            season_end_date,
            decay_halflife_days=halflife_days,
            prior_strength=prior_global,
            damping=damping,
            prior_attack=prior_attack,
            prior_defense=prior_defense,
            team_prior_weight=team_prior_weight,
        )

        # بناء تقييمات ELO
        elo_start = float(getattr(config, "ELO_START", 1500.0))
        elo_k_base = float(getattr(config, "ELO_K_BASE", 24.0))
        elo_hfa = float(getattr(config, "ELO_HFA", 60.0))
        elo_scale = float(getattr(config, "ELO_SCALE", 400.0))
        elo_halflife = int(getattr(config, "ELO_HALFLIFE_DAYS", 365))

        elo_ratings = build_elo_ratings(
            train_matches,
            start_rating=elo_start,
            k_base=elo_k_base,
            hfa_elo=elo_hfa,
            scale=elo_scale,
            decay_halflife_days=elo_halflife,
        )

        # حساب معامل الارتباط (rho)
        rho = fit_dc_rho_mle(
            train_matches,
            factors_attack,
            factors_defense,
            league_avgs,
            decay_halflife_days=halflife_days,
            rho_min=-abs(rho_max),
            rho_max=abs(rho_max),
            rho_step=rho_step,
        )

        return {
            "league_avgs": league_avgs,
            "factors_A": factors_attack,
            "factors_D": factors_defense,
            "elo": elo_ratings,
            "rho": rho,
        }

    except Exception as e:
        log(f"خطأ أثناء تدريب نماذج النافذة: {e}", "WARNING")
        return None


def train_prior_for_next_season(
    matches: List[Dict[str, Any]],
    halflife_days: int,
    prior_global: float,
    damping: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    تدريب عوامل كاملة لموسم لاستخدامها كبراير للموسم التالي.

    يتم التدريب بدون استخدام براير من موسم سابق (team_prior_weight=0.0)
    للحصول على عوامل مستقلة تماماً لهذا الموسم.

    المعاملات:
        matches: جميع مباريات الموسم
        halflife_days: نصف عمر الانحلال الزمني
        prior_global: قوة الانكماش نحو 1.0
        damping: معامل التخميد

    العائد:
        قاموس يحتوي على {"attack": {...}, "defense": {...}}
    """
    try:
        season_end_date = get_end_date_from_matches(matches)
        league_avgs = calculate_league_averages(matches)

        factors_attack, factors_defense = build_team_factors(
            matches,
            league_avgs,
            season_end_date,
            decay_halflife_days=halflife_days,
            prior_strength=prior_global,
            damping=damping,
            prior_attack=None,
            prior_defense=None,
            team_prior_weight=0.0,
        )

        return {
            "attack": factors_attack,
            "defense": factors_defense,
        }

    except Exception as e:
        log(f"فشل حساب براير الموسم: {e}", "WARNING")
        return {}


# -----------------------------------------------------------------------------
# القسم السابع: منطق الباكتيست (Expanding Window)
# -----------------------------------------------------------------------------

def backtest_season_expanding(
    season_key: str,
    matches: List[Dict[str, Any]],
    halflife_days: int,
    prior_global: float,
    team_prior_weight: float,
    rho_max: float,
    rho_step: float,
    min_train: int,
    block_size: int,
    use_elo: bool,
    prev_season_prior: Optional[Dict[str, Dict[str, float]]] = None,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> Dict[str, Any]:
    """
    تنفيذ باكتيست بنافذة متوسعة (expanding window) على موسم واحد.

    الخوارزمية:
    1. نبدأ بنافذة تدريب بحجم min_train
    2. ندرّب النموذج على نافذة التدريب
    3. نتنبأ بنتائج الكتلة التالية (block_size)
    4. نوسّع النافذة لتشمل الكتلة المختبرة
    5. نكرر حتى نفاد المباريات

    المعاملات:
        season_key: مفتاح الموسم (للتسجيل)
        matches: جميع مباريات الموسم (مرتبة زمنياً)
        halflife_days: نصف عمر الانحلال
        prior_global: قوة الانكماش العام
        team_prior_weight: وزن البراير الهرمي
        rho_max: الحد الأقصى لقيمة |rho|
        rho_step: دقة شبكة rho
        min_train: الحد الأدنى لعدد مباريات التدريب
        block_size: حجم كتلة الاختبار
        use_elo: تفعيل ELO
        prev_season_prior: عوامل الموسم السابق
        ece_bins: عدد صناديق ECE

    العائد:
        قاموس يحتوي على:
        - season_key: مفتاح الموسم
        - n: عدد التنبؤات
        - metrics: مقاييس الأداء
        - all_probs: جميع الاحتمالات (لتجميع شامل لاحقاً)
        - all_labels: جميع الملصقات
    """
    n_total = len(matches)

    # التحقق من كفاية المباريات
    min_required = max(DEFAULT_MIN_SEASON_MATCHES, min_train + 5)
    if n_total < min_required:
        log(
            f"  {season_key}: عدد المباريات ({n_total}) "
            f"أقل من الحد الأدنى ({min_required}). تم التخطي.",
            "WARNING"
        )
        return {
            "season_key": season_key,
            "n": 0,
            "metrics": {},
            "all_probs": [],
            "all_labels": [],
        }

    # تجميع التنبؤات والملصقات
    season_probs: List[Tuple[float, float, float]] = []
    season_labels: List[int] = []

    # بدء النافذة المتوسعة
    train_end = min_train
    window_count = 0

    while train_end < n_total:
        # تحديد حدود كتلة الاختبار
        test_start = train_end
        test_end = min(n_total, train_end + block_size)

        # استخراج مجموعتي التدريب والاختبار
        train_subset = matches[:train_end]
        test_subset = matches[test_start:test_end]

        if not test_subset:
            break

        window_count += 1

        # تدريب النماذج على نافذة التدريب
        models = train_models_for_window(
            train_subset,
            halflife_days=halflife_days,
            prior_global=prior_global,
            team_prior_weight=team_prior_weight,
            rho_max=rho_max,
            rho_step=rho_step,
            prev_season_prior=prev_season_prior,
        )

        if models is None:
            # فشل التدريب — ننتقل لتوسيع النافذة
            train_end = test_end
            continue

        # التنبؤ بنتائج كتلة الاختبار
        for m in test_subset:
            # استخراج النتيجة الفعلية
            home_goals, away_goals = parse_score(m)
            if home_goals is None or away_goals is None:
                continue

            # التنبؤ
            try:
                probs = predict_probs_for_match(m, models, use_elo=use_elo)

                # التحقق من صحة الاحتمالات
                if any(math.isnan(p) or math.isinf(p) for p in probs):
                    continue

                label = outcome_label(home_goals, away_goals)

                season_probs.append(probs)
                season_labels.append(label)

            except Exception:
                # تخطي المباريات التي تفشل في التنبؤ
                continue

        # توسيع النافذة
        train_end = test_end

    # حساب المقاييس
    metrics = compute_metrics(season_probs, season_labels, ece_bins=ece_bins)

    log(
        f"  {season_key}: "
        f"N={metrics['n']}, "
        f"نوافذ={window_count}, "
        f"LogLoss={metrics.get('logloss', 'N/A')}, "
        f"Acc={metrics.get('accuracy', 'N/A')}",
        "DEBUG"
    )

    return {
        "season_key": season_key,
        "n": metrics["n"],
        "metrics": metrics,
        "all_probs": season_probs,
        "all_labels": season_labels,
    }


# -----------------------------------------------------------------------------
# القسم الثامن: تشغيل الشبكة (Grid Search) وتجميع النتائج
# -----------------------------------------------------------------------------

def generate_param_combinations(
    grid_halflife: List[int],
    grid_prior_global: List[float],
    grid_team_prior_w: List[float],
    grid_rho_max: List[float],
    rho_step: float,
) -> List[Dict[str, Any]]:
    """
    توليد جميع تركيبات المعاملات من الشبكة.

    المعاملات:
        grid_halflife: قائمة قيم نصف العمر
        grid_prior_global: قائمة قيم قوة الانكماش
        grid_team_prior_w: قائمة أوزان البراير الهرمي
        grid_rho_max: قائمة قيم الحد الأقصى لـ rho
        rho_step: دقة شبكة rho

    العائد:
        قائمة تركيبات المعاملات
    """
    combos = []

    for hf in grid_halflife:
        for pg in grid_prior_global:
            for tpw in grid_team_prior_w:
                for rmax in grid_rho_max:
                    combos.append({
                        "halflife": int(hf),
                        "prior_global": float(pg),
                        "team_prior_w": float(tpw),
                        "rho_max": float(rmax),
                        "rho_step": float(rho_step),
                    })

    return combos


def organize_seasons_by_competition(
    by_season: Dict[str, List[Dict[str, Any]]],
    comps: Optional[List[str]],
    limit_seasons: Optional[int],
) -> Dict[str, List[Tuple[str, List[Dict[str, Any]]]]]:
    """
    تنظيم المواسم حسب المسابقة مع تصفية وتقليص اختياري.

    المعاملات:
        by_season: جميع المواسم المُجمّعة
        comps: قائمة المسابقات المطلوبة (None = الكل)
        limit_seasons: حد أقصى لعدد المواسم الأخيرة لكل مسابقة

    العائد:
        قاموس: رمز المسابقة -> قائمة (مفتاح الموسم, المباريات) مرتبة زمنياً
    """
    # ترتيب المواسم
    items = list(by_season.items())
    items.sort(
        key=lambda kv: (season_key_parts(kv[0])[0], season_key_parts(kv[0])[1])
    )

    # تجميع حسب المسابقة
    comps_to_seasons: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = defaultdict(list)

    for sk, matches in items:
        comp, yr = season_key_parts(sk)

        # تصفية المسابقات إذا حُددت
        if comps and comp not in comps:
            continue

        comps_to_seasons[comp].append((sk, matches))

    # تقليص عدد المواسم إذا طُلب
    if limit_seasons and limit_seasons > 0:
        for comp in list(comps_to_seasons.keys()):
            seasons_list = comps_to_seasons[comp]
            if len(seasons_list) > limit_seasons:
                comps_to_seasons[comp] = seasons_list[-limit_seasons:]

    return dict(comps_to_seasons)


def evaluate_single_combination(
    combo: Dict[str, Any],
    comps_to_seasons: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]],
    min_train: int,
    block_size: int,
    use_elo: bool,
    ece_bins: int,
) -> Dict[str, Any]:
    """
    تقييم تركيبة معاملات واحدة عبر جميع المسابقات والمواسم.

    المعاملات:
        combo: تركيبة المعاملات
        comps_to_seasons: المواسم مُنظّمة حسب المسابقة
        min_train: الحد الأدنى لمباريات التدريب
        block_size: حجم كتلة الاختبار
        use_elo: تفعيل ELO
        ece_bins: عدد صناديق ECE

    العائد:
        قاموس يحتوي على نتائج التقييم الشاملة والتفصيلية
    """
    # تجميع شامل لجميع الاحتمالات والملصقات عبر كل المواسم
    all_probs_global: List[Tuple[float, float, float]] = []
    all_labels_global: List[int] = []

    # نتائج تفصيلية لكل موسم
    season_details: List[Dict[str, Any]] = []

    # تتبع براير الموسم السابق لكل مسابقة
    prev_prior_by_comp: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {}

    for comp, seasons in sorted(comps_to_seasons.items()):
        for sk, matches in seasons:
            # استرجاع براير الموسم السابق لهذه المسابقة
            prev_prior = prev_prior_by_comp.get(comp)

            # تنفيذ الباكتيست لهذا الموسم
            result = backtest_season_expanding(
                season_key=sk,
                matches=matches,
                halflife_days=combo["halflife"],
                prior_global=combo["prior_global"],
                team_prior_weight=combo["team_prior_w"],
                rho_max=combo["rho_max"],
                rho_step=combo["rho_step"],
                min_train=min_train,
                block_size=block_size,
                use_elo=use_elo,
                prev_season_prior=prev_prior,
                ece_bins=ece_bins,
            )

            # تجميع النتائج
            n = result.get("n", 0)
            mx = result.get("metrics", {})

            season_details.append({
                "season_key": sk,
                "n": n,
                "logloss": mx.get("logloss"),
                "brier": mx.get("brier"),
                "accuracy": mx.get("accuracy"),
                "ece": mx.get("ece"),
            })

            # تجميع شامل من الاحتمالات والملصقات الفعلية
            season_probs = result.get("all_probs", [])
            season_labels = result.get("all_labels", [])
            all_probs_global.extend(season_probs)
            all_labels_global.extend(season_labels)

            # تحديث براير الموسم السابق لهذه المسابقة
            # يتم تدريب عوامل كاملة على كل مباريات الموسم الحالي
            # لاستخدامها كبراير للموسم التالي
            updated_prior = train_prior_for_next_season(
                matches=matches,
                halflife_days=combo["halflife"],
                prior_global=combo["prior_global"],
            )

            if updated_prior:
                prev_prior_by_comp[comp] = updated_prior

    # حساب المقاييس الشاملة بدقة (من الاحتمالات والملصقات المُجمّعة)
    global_metrics = compute_metrics(
        all_probs_global, all_labels_global, ece_bins=ece_bins
    )

    # بناء قاموس النتيجة
    result = {
        "halflife": combo["halflife"],
        "prior_global": combo["prior_global"],
        "team_prior_w": combo["team_prior_w"],
        "rho_max": combo["rho_max"],
        "rho_step": combo["rho_step"],
        "use_elo": use_elo,
        "total_samples": global_metrics.get("n", 0),
        "logloss": global_metrics.get("logloss"),
        "brier": global_metrics.get("brier"),
        "accuracy": global_metrics.get("accuracy"),
        "ece": global_metrics.get("ece"),
        "by_season": season_details,
    }

    return result


def format_metric(value: Optional[float], fmt: str = ".5f") -> str:
    """
    تنسيق قيمة مقياس للعرض.

    المعاملات:
        value: القيمة (قد تكون None)
        fmt: تنسيق العرض

    العائد:
        نص منسّق
    """
    if value is None:
        return "N/A"
    return f"{value:{fmt}}"


def print_combination_result(
    combo_index: int,
    total_combos: int,
    combo: Dict[str, Any],
    result: Dict[str, Any],
):
    """
    طباعة نتائج تركيبة واحدة.

    المعاملات:
        combo_index: رقم التركيبة (يبدأ من 1)
        total_combos: إجمالي عدد التركيبات
        combo: تركيبة المعاملات
        result: نتائج التقييم
    """
    log(
        f"[{combo_index}/{total_combos}] "
        f"halflife={combo['halflife']}, "
        f"prior={combo['prior_global']}, "
        f"team_prior_w={combo['team_prior_w']}, "
        f"rho_max={combo['rho_max']}",
        "INFO"
    )

    ll = format_metric(result.get("logloss"), ".5f")
    br = format_metric(result.get("brier"), ".5f")
    ac = format_metric(result.get("accuracy"), ".3f")
    ec = format_metric(result.get("ece"), ".4f")
    n = result.get("total_samples", 0)

    log(
        f"  النتائج: N={n}, LogLoss={ll} | Brier={br} | Acc={ac} | ECE={ec}",
        "INFO"
    )


def print_best_result(best: Dict[str, Any]):
    """
    طباعة أفضل نتيجة مع اقتراح الإعدادات.

    المعاملات:
        best: أفضل نتيجة
    """
    print("")
    log("—" * 60, "INFO")
    log("أفضل إعدادات حسب LogLoss:", "INFO")
    log(
        f"  halflife={best['halflife']} | "
        f"prior_global={best['prior_global']} | "
        f"team_prior_w={best['team_prior_w']} | "
        f"rho_max={best['rho_max']} | "
        f"rho_step={best['rho_step']} | "
        f"use_elo={best['use_elo']}",
        "INFO"
    )
    log(
        f"  المقاييس: N={best['total_samples']} | "
        f"LogLoss={format_metric(best.get('logloss'), '.5f')} | "
        f"Brier={format_metric(best.get('brier'), '.5f')} | "
        f"Acc={format_metric(best.get('accuracy'), '.3f')} | "
        f"ECE={format_metric(best.get('ece'), '.4f')}",
        "INFO"
    )

    # طباعة اقتراح الإعدادات
    log("", "INFO")
    log("اقتراح القيم لتثبيتها في common/config.py:", "INFO")
    print(
        f"""
# common/config.py (اقتراح بناءً على نتائج الباكتيست)
TEAM_FACTORS_HALFLIFE_DAYS = {best['halflife']}
TEAM_FACTORS_PRIOR_GLOBAL = {best['prior_global']}
TEAM_FACTORS_TEAM_PRIOR_WEIGHT = {best['team_prior_w']}
DC_RHO_MIN = {-best['rho_max']:.3f}
DC_RHO_MAX = {best['rho_max']:.3f}
DC_RHO_STEP = {best['rho_step']}
# استخدام ELO في Predictor عبر واجهة التطبيق (use_elo={best['use_elo']})
"""
    )

    # عرض نتائج كل موسم
    log("", "INFO")
    log("تفاصيل كل موسم (أفضل تركيبة):", "INFO")
    for s in best.get("by_season", []):
        n = s.get("n", 0)
        if n <= 0:
            log(f"  {s.get('season_key', '?')}: لا توجد بيانات كافية", "DEBUG")
            continue

        log(
            f"  {s.get('season_key', '?')}: "
            f"N={n}, "
            f"LogLoss={format_metric(s.get('logloss'), '.5f')}, "
            f"Brier={format_metric(s.get('brier'), '.5f')}, "
            f"Acc={format_metric(s.get('accuracy'), '.3f')}, "
            f"ECE={format_metric(s.get('ece'), '.4f')}",
            "INFO"
        )


# -----------------------------------------------------------------------------
# القسم التاسع: حفظ النتائج
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
    keep_last: int = MAX_BACKUPS
):
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


def save_backtest_results(
    results_data: Dict[str, Any],
    save_path: Path,
    create_backups_flag: bool = True,
) -> bool:
    """
    حفظ نتائج الباكتيست في ملف JSON.

    المعاملات:
        results_data: بيانات النتائج
        save_path: مسار ملف الحفظ
        create_backups_flag: إنشاء نسخة احتياطية

    العائد:
        True إذا تم الحفظ بنجاح
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # إنشاء نسخة احتياطية
        if create_backups_flag:
            create_backup(save_path)
            cleanup_old_backups(save_path.parent, save_path.stem)

        # كتابة في ملف مؤقت أولاً
        temp_path = save_path.with_suffix(".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)

        # التحقق من صحة الملف
        with open(temp_path, "r", encoding="utf-8") as f:
            json.load(f)

        # نقل الملف المؤقت ليحلّ محل النهائي
        temp_path.replace(save_path)

        file_size = save_path.stat().st_size
        log(f"تم حفظ نتائج الباكتيست في: {save_path} ({file_size:,} بايت)", "INFO")
        return True

    except Exception as e:
        log(f"فشل حفظ نتائج الباكتيست: {e}", "ERROR")
        # تنظيف الملف المؤقت
        temp_path = save_path.with_suffix(".tmp")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def save_backtest_report(
    report_path: Path,
    best_result: Optional[Dict],
    all_results: List[Dict],
    run_config: Dict,
    duration_seconds: float,
) -> bool:
    """
    حفظ تقرير الباكتيست في ملف نصي قابل للقراءة.

    المعاملات:
        report_path: مسار ملف التقرير
        best_result: أفضل نتيجة
        all_results: جميع النتائج
        run_config: إعدادات التشغيل
        duration_seconds: مدة العملية

    العائد:
        True إذا تم الحفظ بنجاح
    """
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 70)
        lines.append("تقرير الاختبار التاريخي (Backtesting Report)")
        lines.append(f"التاريخ: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"المدة: {format_duration(duration_seconds)}")
        lines.append("=" * 70)
        lines.append("")

        # إعدادات التشغيل
        lines.append("إعدادات التشغيل:")
        for key, value in run_config.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # أفضل نتيجة
        if best_result:
            lines.append("أفضل إعدادات:")
            lines.append(f"  halflife: {best_result.get('halflife')}")
            lines.append(f"  prior_global: {best_result.get('prior_global')}")
            lines.append(f"  team_prior_w: {best_result.get('team_prior_w')}")
            lines.append(f"  rho_max: {best_result.get('rho_max')}")
            lines.append(f"  rho_step: {best_result.get('rho_step')}")
            lines.append("")

            lines.append("أفضل المقاييس:")
            lines.append(f"  N: {best_result.get('total_samples', 0)}")
            lines.append(f"  LogLoss: {format_metric(best_result.get('logloss'), '.5f')}")
            lines.append(f"  Brier: {format_metric(best_result.get('brier'), '.5f')}")
            lines.append(f"  Accuracy: {format_metric(best_result.get('accuracy'), '.3f')}")
            lines.append(f"  ECE: {format_metric(best_result.get('ece'), '.4f')}")
            lines.append("")

            # تفاصيل كل موسم
            lines.append("تفاصيل كل موسم (أفضل تركيبة):")
            for s in best_result.get("by_season", []):
                n = s.get("n", 0)
                if n <= 0:
                    lines.append(f"  {s.get('season_key', '?')}: لا توجد بيانات كافية")
                else:
                    lines.append(
                        f"  {s.get('season_key', '?')}: "
                        f"N={n}, "
                        f"LL={format_metric(s.get('logloss'), '.5f')}, "
                        f"Br={format_metric(s.get('brier'), '.5f')}, "
                        f"Acc={format_metric(s.get('accuracy'), '.3f')}, "
                        f"ECE={format_metric(s.get('ece'), '.4f')}"
                    )
            lines.append("")

        # جميع النتائج
        lines.append(f"جميع التركيبات ({len(all_results)}):")
        lines.append("-" * 70)

        # ترتيب حسب LogLoss
        sorted_results = sorted(
            all_results,
            key=lambda x: x.get("logloss") if x.get("logloss") is not None else 999.0
        )

        for i, r in enumerate(sorted_results, 1):
            lines.append(
                f"  #{i}: "
                f"hf={r.get('halflife')}, "
                f"pg={r.get('prior_global')}, "
                f"tpw={r.get('team_prior_w')}, "
                f"rmax={r.get('rho_max')} → "
                f"N={r.get('total_samples', 0)}, "
                f"LL={format_metric(r.get('logloss'), '.5f')}, "
                f"Acc={format_metric(r.get('accuracy'), '.3f')}"
            )

        lines.append("")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        log(f"تم حفظ تقرير الباكتيست في: {report_path}", "INFO")
        return True

    except Exception as e:
        log(f"فشل حفظ تقرير الباكتيست: {e}", "WARNING")
        return False


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
# القسم العاشر: الدالة الرئيسية لتشغيل الباكتيست
# -----------------------------------------------------------------------------

def run_backtester(
    comps: Optional[List[str]],
    min_train: int,
    block_size: int,
    grid_halflife: List[int],
    grid_prior_global: List[float],
    grid_team_prior_w: List[float],
    grid_rho_max: List[float],
    rho_step: float,
    ece_bins: int,
    limit_seasons: Optional[int],
    use_elo: bool,
    save: bool,
    dry_run: bool = False,
) -> None:
    """
    الدالة الرئيسية لتشغيل الباكتيست.

    تقوم بـ:
    1. تحميل البيانات وتنظيمها
    2. توليد تركيبات المعاملات
    3. تقييم كل تركيبة عبر جميع المواسم
    4. تحديد أفضل التركيبات
    5. حفظ النتائج (اختياري)

    المعاملات:
        comps: قائمة المسابقات المطلوبة (None = الكل)
        min_train: الحد الأدنى لمباريات التدريب
        block_size: حجم كتلة الاختبار
        grid_halflife: شبكة قيم نصف العمر
        grid_prior_global: شبكة قيم قوة الانكماش
        grid_team_prior_w: شبكة أوزان البراير الهرمي
        grid_rho_max: شبكة قيم الحد الأقصى لـ rho
        rho_step: دقة شبكة rho
        ece_bins: عدد صناديق ECE
        limit_seasons: حد أقصى لعدد المواسم
        use_elo: تفعيل ELO
        save: حفظ النتائج
        dry_run: تشغيل تجريبي
    """
    start_time = datetime.now(timezone.utc)

    log("=" * 70, "INFO")
    log("بدء الاختبار التاريخي (Backtesting)", "INFO")
    log(f"الوقت: {start_time.isoformat()}", "INFO")
    log("=" * 70, "INFO")

    if dry_run:
        log("⚠ وضع التشغيل الجاف (Dry Run): لن يتم حفظ أي ملفات.", "WARNING")

    # =========================================================================
    # 1. تحميل البيانات
    # =========================================================================
    log("--- المرحلة 1: تحميل البيانات ---", "INFO")

    try:
        all_matches = load_matches(config.DATA_DIR / "matches.json")
    except Exception as e:
        log(f"فشل تحميل البيانات: {e}", "CRITICAL")
        return

    if not all_matches:
        log("لا توجد مباريات. لا يمكن إجراء الباكتيست.", "CRITICAL")
        return

    # =========================================================================
    # 2. تجميع المباريات حسب الموسم
    # =========================================================================
    log("--- المرحلة 2: تجميع المباريات ---", "INFO")

    by_season = group_matches_by_season(all_matches)

    if not by_season:
        log("لم يتم العثور على مواسم. لا يمكن إجراء الباكتيست.", "CRITICAL")
        return

    # =========================================================================
    # 3. تنظيم المواسم حسب المسابقة
    # =========================================================================
    log("--- المرحلة 3: تنظيم المواسم ---", "INFO")

    comps_to_seasons = organize_seasons_by_competition(
        by_season, comps, limit_seasons
    )

    if not comps_to_seasons:
        log("لا توجد مسابقات مطابقة. تحقق من قيم --comps.", "CRITICAL")
        return

    # عرض المسابقات والمواسم المختارة
    for comp, seasons in sorted(comps_to_seasons.items()):
        season_keys = [sk for sk, _ in seasons]
        log(f"  {comp}: {len(seasons)} موسم — {season_keys}", "INFO")

    # =========================================================================
    # 4. توليد تركيبات المعاملات
    # =========================================================================
    log("--- المرحلة 4: توليد التركيبات ---", "INFO")

    combos = generate_param_combinations(
        grid_halflife=grid_halflife,
        grid_prior_global=grid_prior_global,
        grid_team_prior_w=grid_team_prior_w,
        grid_rho_max=grid_rho_max,
        rho_step=rho_step,
    )

    total_combos = len(combos)
    log(f"عدد التركيبات: {total_combos}", "INFO")
    log(f"  halflife: {grid_halflife}", "DEBUG")
    log(f"  prior_global: {grid_prior_global}", "DEBUG")
    log(f"  team_prior_w: {grid_team_prior_w}", "DEBUG")
    log(f"  rho_max: {grid_rho_max}", "DEBUG")
    log(f"  rho_step: {rho_step}", "DEBUG")

    if dry_run:
        log(
            f"[DRY RUN] سيتم تقييم {total_combos} تركيبة "
            f"عبر {sum(len(s) for s in comps_to_seasons.values())} موسم.",
            "INFO"
        )
        return

    # =========================================================================
    # 5. تقييم كل تركيبة
    # =========================================================================
    log("--- المرحلة 5: تقييم التركيبات ---", "INFO")

    overall_results: List[Dict[str, Any]] = []
    best_candidates: List[Dict[str, Any]] = []

    for ci, combo in enumerate(combos, start=1):
        print("")
        log(f"{'─' * 50}", "INFO")
        log(f"التركيبة [{ci}/{total_combos}]:", "INFO")

        try:
            result = evaluate_single_combination(
                combo=combo,
                comps_to_seasons=comps_to_seasons,
                min_train=min_train,
                block_size=block_size,
                use_elo=use_elo,
                ece_bins=ece_bins,
            )

            # طباعة النتائج
            print_combination_result(ci, total_combos, combo, result)

            # حفظ النتيجة
            overall_results.append(result)

            # إضافة للمرشحين إذا كانت LogLoss صالحة
            if result.get("logloss") is not None and result.get("total_samples", 0) > 0:
                best_candidates.append(result)

        except Exception as e:
            log(f"خطأ أثناء تقييم التركيبة [{ci}]: {e}", "ERROR")
            traceback.print_exc()
            continue

    # =========================================================================
    # 6. اختيار الأفضل
    # =========================================================================
    log("", "INFO")
    log("--- المرحلة 6: اختيار أفضل الإعدادات ---", "INFO")

    best_result = None

    if best_candidates:
        # ترتيب حسب LogLoss (الأقل أفضل)، ثم ECE (الأقل أفضل)
        best_candidates.sort(
            key=lambda x: (
                x.get("logloss", 999.0),
                x.get("ece", 1e9) if x.get("ece") is not None else 1e9,
            )
        )

        best_result = best_candidates[0]
        print_best_result(best_result)

        # عرض أفضل 3 تركيبات (إن وُجدت)
        if len(best_candidates) > 1:
            print("")
            log("أفضل 3 تركيبات:", "INFO")
            for i, candidate in enumerate(best_candidates[:3], 1):
                log(
                    f"  #{i}: "
                    f"hf={candidate['halflife']}, "
                    f"pg={candidate['prior_global']}, "
                    f"tpw={candidate['team_prior_w']}, "
                    f"rmax={candidate['rho_max']} → "
                    f"LL={format_metric(candidate.get('logloss'), '.5f')}, "
                    f"Acc={format_metric(candidate.get('accuracy'), '.3f')}, "
                    f"ECE={format_metric(candidate.get('ece'), '.4f')}",
                    "INFO"
                )
    else:
        log("لم يتم العثور على نتائج صالحة.", "WARNING")

    # =========================================================================
    # 7. حفظ النتائج
    # =========================================================================
    if save and not dry_run:
        log("", "INFO")
        log("--- المرحلة 7: حفظ النتائج ---", "INFO")

        # إعدادات التشغيل
        run_config = {
            "comps": comps,
            "min_train": min_train,
            "block_size": block_size,
            "ece_bins": ece_bins,
            "use_elo": use_elo,
            "grid_halflife": grid_halflife,
            "grid_prior_global": grid_prior_global,
            "grid_team_prior_w": grid_team_prior_w,
            "grid_rho_max": grid_rho_max,
            "rho_step": rho_step,
            "limit_seasons": limit_seasons,
        }

        # حفظ JSON
        results_data = {
            "version": getattr(config, "VERSION", "N/A"),
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "run_config": run_config,
            "best_result": best_result,
            "all_results": overall_results,
            "total_combinations": total_combos,
        }

        save_path = config.DATA_DIR / "backtest_results.json"
        save_backtest_results(results_data, save_path)

        # حفظ تقرير نصي
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        report_path = config.DATA_DIR / "backtest_report.txt"
        save_backtest_report(
            report_path=report_path,
            best_result=best_result,
            all_results=overall_results,
            run_config=run_config,
            duration_seconds=duration,
        )

    # =========================================================================
    # ملخص نهائي
    # =========================================================================
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    print("")
    log("=" * 70, "INFO")
    log("انتهى الاختبار التاريخي", "INFO")
    log(f"المدة: {format_duration(duration)}", "INFO")
    log(f"عدد التركيبات المُقيّمة: {len(overall_results)}/{total_combos}", "INFO")

    if best_result:
        log(
            f"أفضل LogLoss: {format_metric(best_result.get('logloss'), '.5f')} "
            f"(N={best_result.get('total_samples', 0)})",
            "INFO"
        )

    log("=" * 70, "INFO")


# -----------------------------------------------------------------------------
# نقطة الدخول (CLI)
# -----------------------------------------------------------------------------

def main():
    """
    نقطة الدخول الرئيسية — تحليل سطر الأوامر وتشغيل الباكتيست.
    """
    parser = argparse.ArgumentParser(
        description="اختبار تاريخي للنموذج الإحصائي (Dixon–Coles + Team Factors + ELO).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  python 03_backtester.py --save
  python 03_backtester.py --comps PL PD --use-elo --save
  python 03_backtester.py --grid-halflife 90,180,365 --grid-prior-global 2.0,3.0 --save
  python 03_backtester.py --min-train 100 --block-size 30 --limit-seasons 3 --save
  python 03_backtester.py --dry-run
        """
    )

    # إعدادات المسابقات
    parser.add_argument(
        "--comps",
        nargs="*",
        default=None,
        help="رموز المسابقات المطلوب اختبارها (مثل: PL PD SA BL1 FL1). "
             "إذا لم تُحدد، تُستخدم TARGET_COMPETITIONS من الإعدادات."
    )

    # إعدادات النافذة
    parser.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN,
        help=f"أدنى عدد مباريات للتدريب قبل أول تقييم. (افتراضي: {DEFAULT_MIN_TRAIN})"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"حجم كتلة الاختبار في كل خطوة. (افتراضي: {DEFAULT_BLOCK_SIZE})"
    )

    # شبكة المعاملات
    parser.add_argument(
        "--grid-halflife",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GRID_HALFLIFE),
        help=f"قائمة نصف العمر بالأيام (مفصولة بفواصل). (افتراضي: {DEFAULT_GRID_HALFLIFE})"
    )

    parser.add_argument(
        "--grid-prior-global",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GRID_PRIOR_GLOBAL),
        help=f"قائمة قوة الانكماش Gamma. (افتراضي: {DEFAULT_GRID_PRIOR_GLOBAL})"
    )

    parser.add_argument(
        "--grid-team-prior-weight",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GRID_TEAM_PRIOR_WEIGHT),
        help=f"أوزان الانكماش الهرمي. (افتراضي: {DEFAULT_GRID_TEAM_PRIOR_WEIGHT})"
    )

    parser.add_argument(
        "--grid-rho-max",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GRID_RHO_MAX),
        help=f"قيم الحد الأقصى |ρ|. (افتراضي: {DEFAULT_GRID_RHO_MAX})"
    )

    parser.add_argument(
        "--rho-step",
        type=float,
        default=DEFAULT_RHO_STEP,
        help=f"دقة شبكة rho. (افتراضي: {DEFAULT_RHO_STEP})"
    )

    # إعدادات التقييم
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=DEFAULT_ECE_BINS,
        help=f"عدد صناديق ECE. (افتراضي: {DEFAULT_ECE_BINS})"
    )

    parser.add_argument(
        "--limit-seasons",
        type=int,
        default=0,
        help="حصر عدد المواسم الأخيرة لكل دوري (0 = كل المواسم). (افتراضي: 0)"
    )

    # خيارات التشغيل
    parser.add_argument(
        "--use-elo",
        action="store_true",
        help="تفعيل استخدام ELO لتعديل λ أثناء التوقع."
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="حفظ نتائج الباكتيست في data/backtest_results.json"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="تشغيل تجريبي: عرض الإعدادات بدون تنفيذ فعلي."
    )

    args = parser.parse_args()

    # تحديد المسابقات
    comps = args.comps
    if not comps:
        # محاولة الحصول على المسابقات من الإعدادات
        target_comps = getattr(config, "TARGET_COMPETITIONS", None)
        if target_comps:
            if isinstance(target_comps, list):
                comps = target_comps
            elif isinstance(target_comps, dict):
                comps = list(target_comps.keys())

    # تحليل شبكات المعاملات
    halflife_values = parse_grid_list_ints(args.grid_halflife)
    prior_global_values = parse_grid_list_floats(args.grid_prior_global)
    team_prior_w_values = parse_grid_list_floats(args.grid_team_prior_weight)
    rho_max_values = parse_grid_list_floats(args.grid_rho_max)

    # التحقق من وجود قيم في الشبكات
    if not halflife_values:
        halflife_values = DEFAULT_GRID_HALFLIFE
        log("شبكة halflife فارغة. استخدام القيم الافتراضية.", "WARNING")

    if not prior_global_values:
        prior_global_values = DEFAULT_GRID_PRIOR_GLOBAL
        log("شبكة prior_global فارغة. استخدام القيم الافتراضية.", "WARNING")

    if not team_prior_w_values:
        team_prior_w_values = DEFAULT_GRID_TEAM_PRIOR_WEIGHT
        log("شبكة team_prior_weight فارغة. استخدام القيم الافتراضية.", "WARNING")

    if not rho_max_values:
        rho_max_values = DEFAULT_GRID_RHO_MAX
        log("شبكة rho_max فارغة. استخدام القيم الافتراضية.", "WARNING")

    # تحديد حد المواسم
    limit_seasons = args.limit_seasons if args.limit_seasons > 0 else None

    # عرض الإعدادات
    log("إعدادات الباكتيست:", "INFO")
    log(f"  المسابقات: {comps or 'الكل'}", "INFO")
    log(f"  min_train: {args.min_train}", "INFO")
    log(f"  block_size: {args.block_size}", "INFO")
    log(f"  use_elo: {args.use_elo}", "INFO")
    log(f"  limit_seasons: {limit_seasons or 'الكل'}", "INFO")
    log(f"  ece_bins: {args.ece_bins}", "INFO")
    log(f"  شبكة halflife: {halflife_values}", "INFO")
    log(f"  شبكة prior_global: {prior_global_values}", "INFO")
    log(f"  شبكة team_prior_w: {team_prior_w_values}", "INFO")
    log(f"  شبكة rho_max: {rho_max_values}", "INFO")
    log(f"  rho_step: {args.rho_step}", "INFO")

    total_combos = (
        len(halflife_values)
        * len(prior_global_values)
        * len(team_prior_w_values)
        * len(rho_max_values)
    )
    log(f"  إجمالي التركيبات: {total_combos}", "INFO")

    # تشغيل الباكتيست
    try:
        run_backtester(
            comps=comps,
            min_train=args.min_train,
            block_size=args.block_size,
            grid_halflife=halflife_values,
            grid_prior_global=prior_global_values,
            grid_team_prior_w=team_prior_w_values,
            grid_rho_max=rho_max_values,
            rho_step=args.rho_step,
            ece_bins=args.ece_bins,
            limit_seasons=limit_seasons,
            use_elo=args.use_elo,
            save=args.save,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("")
        log("تم إيقاف الباكتيست بواسطة المستخدم (Ctrl+C).", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"خطأ غير متوقع: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
