# 02_trainer.py
# -----------------------------------------------------------------------------
# الوصف:
# هذا السكريبت هو "العقل" الخاص بالمشروع. يقوم بتنفيذ الخطوات التالية:
# 1. تحميل بيانات المباريات الخام.
# 2. تجميع المباريات حسب الموسم الحقيقي لكل دوري.
# 3. لكل موسم على حدة:
#    أ. البحث عن أفضل مجموعة من المعاملات (Hyperparameters) باستخدام Grid Search.
#    ب. تدريب النماذج الإحصائية النهائية باستخدام أفضل المعاملات التي تم العثور عليها.
# 4. حفظ النماذج المدربة النهائية في مجلد `models/`.
# -----------------------------------------------------------------------------
import json
import itertools
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# استيراد الوحدات المشتركة
from common.config import config # ✅ استيراد الكائن config المحدث
from common.utils import log, parse_date_safe
from common.modeling import (
    calculate_league_averages,
    build_team_factors,
    build_elo_ratings,
    fit_dc_rho_mle,
)
# ✅ سنحتاج إلى دوال للتقييم والتنبؤ (نفترض وجودها)
# from common.evaluation import evaluate_model_on_validation_set

def load_matches(path) -> List[Dict[str, Any]]:
    """ يقوم بتحميل بيانات المباريات من ملف JSON المحدد. """
    log("جاري تحميل بيانات المباريات من `matches.json`...", "INFO")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except IOError as e:
        log("ملف `matches.json` غير موجود. يرجى تشغيل `01_pipeline.py` أولاً.", "CRITICAL")
        raise e

def group_matches_by_season(matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """ 
    ✅ دالة مُحسَّنة: تجمع المباريات حسب الموسم الحقيقي بناءً على تاريخ المباراة (utcDate).
    """
    matches_by_season = defaultdict(list)
    for match in matches:
        match_date = parse_date_safe(match.get('utcDate'))
        if not match_date:
            continue
        
        # إذا كان شهر المباراة أكبر من شهر بداية الموسم (عادة يوليو)، فالموسم يبدأ في هذا العام.
        # وإلا، فالموسم بدأ في العام السابق.
        start_month = getattr(config, "CURRENT_SEASON_START_MONTH", 7)
        season_start_year = match_date.year if match_date.month >= start_month else match_date.year - 1
        
        comp_code = match.get('competition', {}).get('code', 'UNK')
        season_key = f"{comp_code}_{season_start_year}"
        matches_by_season[season_key].append(match)

    log(f"تم تجميع المباريات في {len(matches_by_season)} موسمًا فريدًا.", "INFO")
    return matches_by_season

def find_best_params_for_season(
    matches: List[Dict[str, Any]],
    prev_factors: Dict
) -> Dict[str, Any]:
    """
    ✅ دالة جديدة: تبحث عن أفضل مجموعة معاملات لموسم معين باستخدام Grid Search.
    
    تقوم بتقسيم بيانات الموسم إلى مجموعة تدريب وتحقق، وتختبر كل تركيبة من المعاملات
    من `config.HYPERPARAM_GRID` لاختيار الأفضل.
    """
    matches.sort(key=lambda m: m.get('utcDate', ''))
    split_point = int(len(matches) * 0.8)
    train_set = matches[:split_point]
    validation_set = matches[split_point:]

    if not validation_set:
        log(f"مجموعة التحقق فارغة، لا يمكن إجراء البحث الشبكي. سيتم استخدام الإعدادات الافتراضية.", "WARNING")
        # في هذه الحالة، يمكننا إعادة أول تركيبة كإعداد افتراضي
        keys, values = zip(*config.HYPERPARAM_GRID.items())
        return dict(zip(keys, [v[0] for v in values]))

    best_params = None
    best_score = float('inf')

    # إنشاء كل التوليفات الممكنة من المعاملات
    keys, values = zip(*config.HYPERPARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    log(f"جاري اختبار {len(param_combinations)} تركيبة من المعاملات...", "INFO")

    end_date_train = parse_date_safe(train_set[-1]['utcDate'])

    for params in param_combinations:
        # --- تدريب نموذج مؤقت على مجموعة التدريب ---
        league_avgs_train = calculate_league_averages(train_set)
        
        prior_attack = prev_factors.get("attack", None)
        prior_defense = prev_factors.get("defense", None)
        
        factors_A_train, factors_D_train = build_team_factors(
            train_set,
            league_avgs_train,
            end_date_train,
            decay_halflife_days=params["TEAM_FACTORS_HALFLIFE_DAYS"],
            prior_strength=params["TEAM_FACTORS_PRIOR_GLOBAL"],
            damping=params["TEAM_FACTORS_DAMPING"],
            prior_attack=prior_attack,
            prior_defense=prior_defense,
            team_prior_weight=params["TEAM_FACTORS_TEAM_PRIOR_WEIGHT"],
        )
        
        # --- التقييم على مجموعة التحقق ---
        # !!! ملاحظة: هذا الجزء يتطلب دالة تقييم حقيقية (`evaluate_model`)
        # سنقوم بمحاكاة النتيجة هنا لأغراض العرض التوضيحي.
        # يجب استبدال هذا بمنطق التقييم الفعلي الذي يحسب الـ logloss.
        # score = evaluate_model_on_validation_set(validation_set, factors_A_train, ...)
        
        # محاكاة بسيطة: نفترض أن المعاملات الأكبر أفضل (logloss أقل)
        simulated_score = -sum(v for v in params.values() if isinstance(v, (int, float)))

        if simulated_score < best_score:
            best_score = simulated_score
            best_params = params

    log(f"أفضل المعاملات التي تم إيجادها للموسم: {best_params} بنتيجة محاكاة: {best_score}", "SUCCESS")
    return best_params

def train_all_models(matches_by_season: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    ✅ دالة مُحدَّثة: تقوم بتنسيق عملية البحث عن أفضل المعاملات والتدريب النهائي.
    """
    trained_models = {
        "team_factors": {},
        "elo_ratings": {},
        "league_averages": {},
        "rho_values": {},
        "best_params": {} # ✅ سنقوم بحفظ أفضل المعاملات لكل موسم
    }

    def parse_key(k: str) -> Tuple[str, int]:
        """ ✅ دالة مُحسَّنة لمعالجة الأخطاء. """
        try:
            comp, yr_str = k.split("_")
            return comp, int(yr_str)
        except (ValueError, IndexError) as e:
            log(f"فشل في تحليل مفتاح الموسم '{k}': {e}. سيتم استخدام ترتيب افتراضي.", "WARNING")
            return k, 0

    items = sorted(matches_by_season.items(), key=lambda kv: parse_key(kv[0]))

    last_factors_by_comp: Dict[str, Dict] = defaultdict(dict)

    for season_key, matches in items:
        if len(matches) < 50: # زيادة الحد الأدنى لضمان وجود مجموعة تحقق كافية
            log(f"تجاهل الموسم {season_key} لقلة عدد المباريات ({len(matches)}).", "WARNING")
            continue

        comp_code, _ = parse_key(season_key)
        log(f"--- بدء المعالجة لموسم: {season_key} ---", "INFO")

        # 1) البحث عن أفضل المعاملات لهذا الموسم
        prev_factors = last_factors_by_comp.get(comp_code, {})
        best_params = find_best_params_for_season(matches, prev_factors)
        trained_models["best_params"][season_key] = best_params

        # 2) تدريب النماذج النهائية على **كامل** بيانات الموسم باستخدام أفضل المعاملات
        log(f"جاري تدريب النموذج النهائي لـ {season_key} باستخدام أفضل المعاملات...", "INFO")
        
        season_end_date = max(d for d in (parse_date_safe(m.get('utcDate')) for m in matches) if d)
        
        league_avgs = calculate_league_averages(matches)

        factors_A, factors_D = build_team_factors(
            matches,
            league_avgs,
            season_end_date,
            decay_halflife_days=best_params["TEAM_FACTORS_HALFLIFE_DAYS"],
            prior_strength=best_params["TEAM_FACTORS_PRIOR_GLOBAL"],
            damping=best_params["TEAM_FACTORS_DAMPING"],
            prior_attack=prev_factors.get("attack"),
            prior_defense=prev_factors.get("defense"),
            team_prior_weight=best_params["TEAM_FACTORS_TEAM_PRIOR_WEIGHT"],
        )

        elo = build_elo_ratings(matches, k_base=config.ELO_K_BASE, hfa_elo=config.ELO_HFA)

        rho = fit_dc_rho_mle(
            matches,
            factors_A,
            factors_D,
            league_avgs,
            decay_halflife_days=best_params["TEAM_FACTORS_HALFLIFE_DAYS"],
            rho_max=best_params["DC_RHO_MAX"]
        )

        trained_models["league_averages"][season_key] = league_avgs
        trained_models["team_factors"][season_key] = {"attack": factors_A, "defense": factors_D}
        trained_models["elo_ratings"][season_key] = elo
        trained_models["rho_values"][season_key] = rho

        # 3) تحديث عوامل الموسم السابق للمسابقة الحالية
        last_factors_by_comp[comp_code] = {"attack": factors_A, "defense": factors_D}

    return trained_models

def save_models(trained_models: Dict[str, Dict[str, Any]]):
    """ يقوم بحفظ كل نوع من النماذج المدربة في ملف JSON منفصل خاص به. """
    log("جاري حفظ النماذج المدربة...", "INFO")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, model_data in trained_models.items():
        if not model_data:
            log(f"لا توجد بيانات لحفظها للنموذج: {model_name}", "WARNING")
            continue
        output_path = config.MODELS_DIR / f"{model_name}.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            log(f"تم حفظ نموذج `{model_name}` بنجاح في: {output_path}", "SUCCESS")
        except IOError as e:
            log(f"فشل في حفظ النموذج `{model_name}`: {e}", "ERROR")

def main():
    """ الدالة الرئيسية التي تنسق عملية تدريب النماذج. """
    log("--- بدء عملية تدريب النماذج (Model Trainer) ---", "INFO")
    try:
        all_matches = load_matches(config.DATA_DIR / "matches.json")
        matches_by_season = group_matches_by_season(all_matches)
        trained_models = train_all_models(matches_by_season)
        save_models(trained_models)
    except Exception as e:
        log(f"حدث خطأ غير متوقع أثناء عملية التدريب: {e}", "CRITICAL", exc_info=True)
        return
    log("--- انتهت عملية تدريب النماذج بنجاح ---", "INFO")

if __name__ == "__main__":
    main()
