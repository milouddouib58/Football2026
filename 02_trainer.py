# 02_trainer.py
# -----------------------------------------------------------------------------
# الوصف:
# هذا السكريبت هو "العقل" الخاص بالمشروع. يقوم بتنفيذ الخطوات التالية:
# 1. تحميل بيانات المباريات الخام التي تم جمعها بواسطة `01_pipeline.py`.
# 2. تجميع المباريات حسب كل موسم لكل دوري على حدة وترتيبها زمنيًا.
# 3. تدريب النماذج الإحصائية (Team Factors, Elo, Rho) لكل موسم بشكل مستقل،
#    مع وزن زمني وانكماش Gamma وبراير هرمي نحو الموسم السابق.
# 4. حفظ النماذج المدربة في ملفات JSON منفصلة داخل مجلد `models/`
# -----------------------------------------------------------------------------
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe
from common.modeling import (
    calculate_league_averages,
    build_team_factors,
    build_elo_ratings,
    fit_dc_rho_mle,
)

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
    """ يجمع قائمة المباريات في قاموس مفتاحه هو "مفتاح الموسم" (e.g., 'PL_2024'). """
    matches_by_season = defaultdict(list)
    for match in matches:
        season_year = match.get('season', {}).get('startDate', '1900')[:4]
        comp_code = match.get('competition', {}).get('code', 'UNK')
        season_key = f"{comp_code}_{season_year}"
        matches_by_season[season_key].append(match)
    log(f"تم تجميع المباريات في {len(matches_by_season)} موسمًا فريدًا.", "INFO")
    return matches_by_season

def train_all_models(matches_by_season: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    تدريب نماذج كل موسم بترتيب زمني لكل مسابقة:
    - Team Factors مع وزن زمني + Gamma-smoothing + براير هرمي نحو الموسم السابق
    - Elo مستقر بوزن زمني
    - rho عبر بحث شبكي مرجّح زمنياً
    """
    trained_models = {
        "team_factors": {},
        "elo_ratings": {},
        "league_averages": {},
        "rho_values": {}
    }

    # ترتيب المواسم حسب (رمز المسابقة، سنة البداية)
    def parse_key(k: str) -> Tuple[str, int]:
        try:
            comp, yr = k.split("_")
            return comp, int(yr)
        except Exception:
            return k, 0

    items = list(matches_by_season.items())
    items.sort(key=lambda kv: (parse_key(kv[0])[0], parse_key(kv[0])[1]))

    # آخر عوامل لكل مسابقة لاستخدامها كبراير هرمي للموسم التالي
    last_factors_by_comp: Dict[str, Dict[str, Dict[str, float]]] = {}

    for season_key, matches in items:
        if len(matches) < 30:
            log(f"تجاهل الموسم {season_key} لقلة عدد المباريات ({len(matches)} مباراة).", "WARNING")
            continue

        comp_code, season_year = parse_key(season_key)
        log(f"جاري تدريب النماذج لموسم: {season_key}", "INFO")

        end_dates = [d for d in (parse_date_safe(m.get('utcDate')) for m in matches) if d]
        if not end_dates:
            log(f"لا توجد تواريخ صالحة للموسم {season_key}.", "WARNING")
            continue
        season_end_date = max(end_dates)

        # 1) متوسطات الدوري
        league_avgs = calculate_league_averages(matches)

        # 2) عوامل الفرق مع وزن زمني + Gamma-smoothing + براير هرمي
        prev = last_factors_by_comp.get(comp_code, {})
        prior_attack = prev.get("attack", None)
        prior_defense = prev.get("defense", None)

        factors_A, factors_D = build_team_factors(
            matches,
            league_avgs,
            season_end_date,
            decay_halflife_days=getattr(config, "TEAM_FACTORS_HALFLIFE_DAYS", 180),
            prior_strength=getattr(config, "TEAM_FACTORS_PRIOR_GLOBAL", 3.0),
            damping=0.5,
            prior_attack=prior_attack,
            prior_defense=prior_defense,
            team_prior_weight=getattr(config, "TEAM_FACTORS_TEAM_PRIOR_WEIGHT", 0.0),
        )

        # 3) تقييم ELO بوزن زمني
        elo = build_elo_ratings(
            matches,
            start_rating=getattr(config, "ELO_START", 1500.0),
            k_base=getattr(config, "ELO_K_BASE", 24.0),
            hfa_elo=getattr(config, "ELO_HFA", 60.0),
            scale=getattr(config, "ELO_SCALE", 400.0),
            decay_halflife_days=getattr(config, "ELO_HALFLIFE_DAYS", 365),
        )

        # 4) تقدير ρ عبر بحث شبكي مرجّح زمنيًا
        rho = fit_dc_rho_mle(
            matches,
            factors_A,
            factors_D,
            league_avgs,
            decay_halflife_days=getattr(config, "TEAM_FACTORS_HALFLIFE_DAYS", 180),
            rho_min=getattr(config, "DC_RHO_MIN", -0.2),
            rho_max=getattr(config, "DC_RHO_MAX", 0.2),
            rho_step=getattr(config, "DC_RHO_STEP", 0.001),
        )

        trained_models["league_averages"][season_key] = league_avgs
        trained_models["team_factors"][season_key] = {"attack": factors_A, "defense": factors_D}
        trained_models["elo_ratings"][season_key] = elo
        trained_models["rho_values"][season_key] = rho

        # خزّن العوامل كبراير للموسم التالي في نفس المسابقة
        last_factors_by_comp[comp_code] = {"attack": factors_A, "defense": factors_D}

    return trained_models

def save_models(trained_models: Dict[str, Dict[str, Any]]):
    """ يقوم بحفظ كل نوع من النماذج المدربة في ملف JSON منفصل خاص به. """
    log("جاري حفظ النماذج المدربة...", "INFO")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)  # تأكد من وجود مجلد النماذج
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
        log(f"حدث خطأ غير متوقع أثناء عملية التدريب: {e}", "CRITICAL")
        return
    log("--- انتهت عملية تدريب النماذج بنجاح ---", "INFO")

if __name__ == "__main__":
    main()
