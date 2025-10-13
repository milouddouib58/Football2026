# 02_trainer.py
# -----------------------------------------------------------------------------
# الوصف:
# هذا السكريبت هو "العقل" الخاص بالمشروع. يقوم بتنفيذ الخطوات التالية:
# 1. تحميل بيانات المباريات الخام.
# 2. تجميع المباريات حسب الموسم الحقيقي لكل دوري.
# 3. لكل موسم على حدة:
#    أ. البحث عن أفضل مجموعة من المعاملات (Hyperparameters) باستخدام تقييم Log-Loss
#       حقيقي على مجموعة تحقق منفصلة.
#    ب. تدريب النماذج الإحصائية النهائية على كامل بيانات الموسم باستخدام أفضل
#       المعاملات التي تم العثور عليها.
# 4. حفظ النماذج المدربة النهائية في مجلد `models/`.
# -----------------------------------------------------------------------------
import json
import itertools
import numpy as np
from scipy.stats import poisson
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# استيراد الوحدات المشتركة
from common.config import config
from common.utils import log, parse_date_safe
from common.modeling import (
    calculate_league_averages,
    build_team_factors,
    build_elo_ratings,
    fit_dc_rho_mle,
)

# ----------------------------------------------------------------------------------
# ✅ --- القسم الأول: دوال التقييم والتنبؤ (إضافات جديدة وحاسمة) ---
# ----------------------------------------------------------------------------------

def predict_match_probabilities(match: Dict, factors_A: Dict, factors_D: Dict,
                                league_avgs: Dict, rho: float) -> Tuple[float, float, float]:
    """
    تتنبأ باحتمالات (فوز المضيف، تعادل، فوز الضيف) لمباراة واحدة باستخدام نموذج Dixon-Coles.
    """
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    
    avg_home_goals = league_avgs['avg_home_goals']
    avg_away_goals = league_avgs['avg_away_goals']
    
    # استخدام 1.0 كقيمة افتراضية للفرق الجديدة التي قد لا تكون في مجموعة التدريب
    home_attack = factors_A.get(home_team, 1.0)
    away_defense = factors_D.get(away_team, 1.0)
    away_attack = factors_A.get(away_team, 1.0)
    home_defense = factors_D.get(home_team, 1.0)
    
    lambda_home = home_attack * away_defense * avg_home_goals
    lambda_away = away_attack * home_defense * avg_away_goals
    
    # بناء مصفوفة احتمالات الأهداف
    max_goals = 10  # حد أقصى معقول لعدد الأهداف
    home_goals_pmf = poisson.pmf(np.arange(0, max_goals + 1), lambda_home)
    away_goals_pmf = poisson.pmf(np.arange(0, max_goals + 1), lambda_away)
    
    score_matrix = np.outer(home_goals_pmf, away_goals_pmf)
    
    # تطبيق معامل الارتباط rho (Dixon-Coles adjustment)
    if rho != 0:
        tau = np.ones((max_goals + 1, max_goals + 1))
        tau[0, 0] = 1 - (lambda_home * lambda_away * rho)
        tau[0, 1] = 1 + (lambda_home * rho)
        tau[1, 0] = 1 + (lambda_away * rho)
        tau[1, 1] = 1 - rho
        score_matrix *= tau

    prob_home_win = np.sum(np.tril(score_matrix, -1))
    prob_draw = np.sum(np.diag(score_matrix))
    prob_away_win = np.sum(np.triu(score_matrix, 1))
    
    # تطبيع الاحتمالات لضمان أن مجموعها يساوي 1.0 بالضبط
    total_prob = prob_home_win + prob_draw + prob_away_win
    if total_prob == 0: return (1/3, 1/3, 1/3) # حالة نادرة جداً
    
    return (prob_home_win / total_prob, prob_draw / total_prob, prob_away_win / total_prob)

def calculate_logloss(predictions: List[Tuple], validation_set: List[Dict]) -> float:
    """
    تحسب Log-Loss لمجموعة من التنبؤات والنتائج الفعلية.
    """
    log_losses = []
    epsilon = 1e-9 # قيمة صغيرة لمنع log(0)
    
    for i, match in enumerate(validation_set):
        actual_result = match.get('score', {}).get('winner')
        probs = predictions[i]
        
        if actual_result == 'HOME_TEAM':
            prob_actual = probs[0]
        elif actual_result == 'DRAW':
            prob_actual = probs[1]
        elif actual_result == 'AWAY_TEAM':
            prob_actual = probs[2]
        else:
            continue # تجاهل المباريات بدون نتيجة

        log_losses.append(np.log(prob_actual + epsilon))
            
    if not log_losses: return 999.0 # قيمة عالية جداً في حالة عدم وجود نتائج
    
    return -np.mean(log_losses)

# ----------------------------------------------------------------------------------
# ✅ --- القسم الثاني: دالة البحث عن أفضل المعايير (القلب النابض للتحسين) ---
# ----------------------------------------------------------------------------------

def find_best_params_for_season(matches: List[Dict[str, Any]], prev_factors: Dict) -> Dict:
    """
    دالة مُصححة بالكامل: تبحث عن أفضل مجموعة معاملات لموسم معين باستخدام تقييم Log-Loss حقيقي.
    """
    matches.sort(key=lambda m: m.get('utcDate', ''))
    split_point = int(len(matches) * 0.8)
    train_set = matches[:split_point]
    validation_set = matches[split_point:]

    if len(validation_set) < 10:
        log(f"مجموعة التحقق صغيرة جدًا ({len(validation_set)} مباراة)، لا يمكن إجراء بحث دقيق.", "WARNING")
        # في هذه الحالة، نختار أول تركيبة كإعداد افتراضي لتجنب الفشل
        keys, values = zip(*config.HYPERPARAM_GRID.items())
        return dict(zip(keys, [v[0] for v in values]))

    best_params = None
    best_score = float('inf') # Log-Loss الأقل هو الأفضل

    keys, values = zip(*config.HYPERPARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    log(f"البحث عن أفضل المعايير عبر {len(param_combinations)} تركيبة...", "INFO")
    
    end_date_train = parse_date_safe(train_set[-1]['utcDate'])

    for params in param_combinations:
        # 1. تدريب نموذج مؤقت على **مجموعة التدريب فقط**
        league_avgs_train = calculate_league_averages(train_set)
        
        factors_A_train, factors_D_train = build_team_factors(
            train_set, league_avgs_train, end_date_train,
            decay_halflife_days=params["TEAM_FACTORS_HALFLIFE_DAYS"],
            prior_strength=params["TEAM_FACTORS_PRIOR_GLOBAL"],
            damping=params["TEAM_FACTORS_DAMPING"],
            prior_attack=prev_factors.get("attack"),
            prior_defense=prev_factors.get("defense"),
            team_prior_weight=params["TEAM_FACTORS_TEAM_PRIOR_WEIGHT"],
        )
        
        temp_rho = fit_dc_rho_mle(
            train_set, factors_A_train, factors_D_train, league_avgs_train,
            decay_halflife_days=params["TEAM_FACTORS_HALFLIFE_DAYS"],
            rho_max=params["DC_RHO_MAX"]
        )
        
        # 2. التنبؤ بنتائج **مجموعة التحقق** باستخدام النموذج المؤقت
        predictions = [
            predict_match_probabilities(match, factors_A_train, factors_D_train, league_avgs_train, temp_rho)
            for match in validation_set
        ]
        
        # 3. حساب Log-Loss (التقييم الحقيقي)
        score = calculate_logloss(predictions, validation_set)
        
        # 4. تحديث أفضل المعايير إذا كانت النتيجة الحالية أفضل
        if score < best_score:
            best_score = score
            best_params = params

    log(f"تم العثور على أفضل المعايير: {best_params} بنتيجة Log-Loss = {best_score:.4f}", "SUCCESS")
    return best_params

# ----------------------------------------------------------------------------------
# ✅ --- القسم الثالث: دوال التدريب الرئيسية (المنسق العام) ---
# ----------------------------------------------------------------------------------

def load_and_group_matches() -> Dict[str, List[Dict[str, Any]]]:
    """
    مهمة مزدوجة: تحميل وتجميع المباريات في قاموس حسب الموسم.
    """
    log("جاري تحميل وتجميع بيانات المباريات...", "INFO")
    all_matches = load_matches(config.DATA_DIR / "matches.json")
    
    matches_by_season = defaultdict(list)
    for match in all_matches:
        match_date = parse_date_safe(match.get('utcDate'))
        if not match_date: continue
        
        start_month = getattr(config, "CURRENT_SEASON_START_MONTH", 7)
        season_start_year = match_date.year if match_date.month >= start_month else match_date.year - 1
        
        comp_code = match.get('competition', {}).get('code', 'UNK')
        season_key = f"{comp_code}_{season_start_year}"
        matches_by_season[season_key].append(match)

    log(f"تم تجميع المباريات في {len(matches_by_season)} موسمًا فريدًا.", "INFO")
    return matches_by_season

def train_all_models(matches_by_season: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    المنسق الرئيسي: يدير عملية التدريب الكاملة لكل المواسم.
    """
    trained_models = {
        "team_factors": {}, "elo_ratings": {}, "league_averages": {}, 
        "rho_values": {}, "best_params": {}
    }

    def parse_key(k: str) -> Tuple[str, int]:
        try:
            comp, yr_str = k.split("_")
            return comp, int(yr_str)
        except (ValueError, IndexError) as e:
            log(f"فشل في تحليل مفتاح الموسم '{k}': {e}. سيتم استخدام ترتيب افتراضي.", "WARNING")
            return k, 0

    # الترتيب الزمني للمواسم أمر حاسم لعمل البراير الهرمي
    sorted_seasons = sorted(matches_by_season.items(), key=lambda kv: parse_key(kv[0]))
    last_factors_by_comp = defaultdict(dict)

    for season_key, matches in sorted_seasons:
        if len(matches) < 50:
            log(f"تجاهل الموسم {season_key} لقلة عدد المباريات ({len(matches)}).", "WARNING")
            continue

        comp_code, _ = parse_key(season_key)
        log(f"--- بدء المعالجة لموسم: {season_key} ---", "HEADER")

        # الخطوة 1: البحث عن أفضل المعايير لهذا الموسم باستخدام مجموعة تحقق
        prev_factors = last_factors_by_comp.get(comp_code, {})
        best_params = find_best_params_for_season(matches, prev_factors)
        trained_models["best_params"][season_key] = best_params

        # الخطوة 2: تدريب النموذج النهائي على **كامل** بيانات الموسم باستخدام أفضل المعايير
        log(f"جاري تدريب النموذج النهائي لـ {season_key}...", "INFO")
        
        season_end_date = max(d for d in (parse_date_safe(m.get('utcDate')) for m in matches) if d)
        
        league_avgs = calculate_league_averages(matches)

        factors_A, factors_D = build_team_factors(
            matches, league_avgs, season_end_date,
            decay_halflife_days=best_params["TEAM_FACTORS_HALFLIFE_DAYS"],
            prior_strength=best_params["TEAM_FACTORS_PRIOR_GLOBAL"],
            damping=best_params["TEAM_FACTORS_DAMPING"],
            prior_attack=prev_factors.get("attack"),
            prior_defense=prev_factors.get("defense"),
            team_prior_weight=best_params["TEAM_FACTORS_TEAM_PRIOR_WEIGHT"],
        )

        elo = build_elo_ratings(matches, k_base=config.ELO_K_BASE, hfa_elo=config.ELO_HFA)

        rho = fit_dc_rho_mle(
            matches, factors_A, factors_D, league_avgs,
            decay_halflife_days=best_params["TEAM_FACTORS_HALFLIFE_DAYS"],
            rho_max=best_params["DC_RHO_MAX"]
        )

        # الخطوة 3: حفظ كل مكونات النموذج المدرب
        trained_models["league_averages"][season_key] = league_avgs
        trained_models["team_factors"][season_key] = {"attack": factors_A, "defense": factors_D}
        trained_models["elo_ratings"][season_key] = elo
        trained_models["rho_values"][season_key] = rho

        # الخطوة 4: تحديث عوامل الموسم السابق لاستخدامها كـ "براير" للموسم التالي
        last_factors_by_comp[comp_code] = {"attack": factors_A, "defense": factors_D}

    return trained_models

def save_models(trained_models: Dict[str, Dict[str, Any]]):
    # ... (هذه الدالة تبقى كما هي، لا تحتاج لتعديل)

def main():
    """ الدالة الرئيسية التي تنسق عملية تدريب النماذج. """
    log("--- بدء عملية تدريب النماذج (Model Trainer) ---", "HEADER")
    try:
        matches_by_season = load_and_group_matches()
        trained_models = train_all_models(matches_by_season)
        # save_models(trained_models) # يمكنك تفعيل الحفظ بعد التأكد من النتائج
    except Exception as e:
        log(f"حدث خطأ غير متوقع أثناء عملية التدريب: {e}", "CRITICAL", exc_info=True)
        return
    log("--- انتهت عملية تدريب النماذج بنجاح ---", "SUCCESS")

if __name__ == "__main__":
    main()
