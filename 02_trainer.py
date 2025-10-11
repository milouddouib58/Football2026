# -----------------------------------------------------------------------------
# 02_trainer.py (النسخة المحسنة والمنظمة)
# -----------------------------------------------------------------------------
# الوصف:
#   هذا السكريبت هو "العقل" الخاص بالمشروع. يقوم بتنفيذ الخطوات التالية:
#   1. تحميل بيانات المباريات الخام التي تم جمعها بواسطة `01_pipeline.py`.
#   2. تجميع المباريات حسب كل موسم لكل دوري على حدة.
#   3. تدريب النماذج الإحصائية (Team Factors, Elo, Rho) لكل موسم بشكل مستقل.
#   4. حفظ النماذج المدربة في ملفات JSON منفصلة داخل مجلد `models/`
#      لتكون جاهزة للاستخدام في التنبؤ أو التحليل.
# -----------------------------------------------------------------------------

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe
from common.modeling import (
    calculate_league_averages,
    build_team_factors,
    build_elo_ratings,
    fit_dc_rho_mle,
)


def load_matches(path: Path) -> List[Dict[str, Any]]:
    """
    يقوم بتحميل بيانات المباريات من ملف JSON المحدد.

    Args:
        path (Path): المسار إلى ملف matches.json.

    Returns:
        List[Dict[str, Any]]: قائمة بجميع المباريات.
    
    Raises:
        IOError: إذا لم يتم العثور على الملف.
    """
    log("جاري تحميل بيانات المباريات من `matches.json`...", "INFO")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except IOError as e:
        log("ملف `matches.json` غير موجود. يرجى تشغيل `01_pipeline.py` أولاً.", "CRITICAL")
        raise e


def group_matches_by_season(matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    يجمع قائمة المباريات في قاموس مفتاحه هو "مفتاح الموسم" (e.g., 'PL_2024').

    Returns:
        Dict[str, List[Dict[str, Any]]]: قاموس يحتوي على مباريات مجمعة حسب الموسم.
    """
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
    يمر على كل موسم ويقوم بتدريب جميع النماذج الإحصائية المطلوبة له.

    Returns:
        Dict[str, Dict[str, Any]]: قاموس شامل يحتوي على جميع النماذج المدربة.
    """
    # هيكل لتخزين جميع النماذج المدربة
    trained_models = {
        "team_factors": {},
        "elo_ratings": {},
        "league_averages": {},
        "rho_values": {}
    }

    for season_key, matches in matches_by_season.items():
        if len(matches) < 30:  # نتجاهل المواسم التي تحتوي على عدد قليل من المباريات
            log(f"تجاهل الموسم {season_key} لقلة عدد المباريات ({len(matches)} مباراة).", "WARNING")
            continue
            
        log(f"جاري تدريب النماذج لموسم: {season_key}", "INFO")

        # الحصول على تاريخ نهاية الموسم لتحديد الأوزان الزمنية
        end_dates = [d for d in (parse_date_safe(m.get('utcDate')) for m in matches) if d]
        if not end_dates:
            continue
        season_end_date = max(end_dates)

        # تدريب النماذج الأربعة
        league_avgs = calculate_league_averages(matches)
        factors_A, factors_D = build_team_factors(matches, league_avgs, season_end_date)
        elo = build_elo_ratings(matches)
        rho = fit_dc_rho_mle(matches, factors_A, factors_D, league_avgs)

        # تخزين النماذج المدربة
        trained_models["league_averages"][season_key] = league_avgs
        trained_models["team_factors"][season_key] = {"attack": factors_A, "defense": factors_D}
        trained_models["elo_ratings"][season_key] = elo
        trained_models["rho_values"][season_key] = rho

    return trained_models


def save_models(trained_models: Dict[str, Dict[str, Any]]):
    """
    يقوم بحفظ كل نوع من النماذج المدربة في ملف JSON منفصل خاص به.
    """
    log("جاري حفظ النماذج المدربة...", "INFO")
    for model_name, model_data in trained_models.items():
        if not model_data:
            log(f"لا توجد بيانات لحفظها للنموذج: {model_name}", "WARNING")
            continue
            
        output_path = config.MODELS_DIR / f"{model_name}.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            log(f"تم حفظ نموذج `{model_name}` بنجاح في: {output_path}", "SUCCESS")
        except IOError as e:
            log(f"فشل في حفظ النموذج `{model_name}`: {e}", "ERROR")


def main():
    """
    الدالة الرئيسية التي تنسق عملية تدريب النماذج.
    """
    log("--- بدء عملية تدريب النماذج (Model Trainer) ---", "INFO")
    try:
        # الخطوة 1: تحميل البيانات
        all_matches = load_matches(config.DATA_DIR / "matches.json")
        
        # الخطوة 2: تجميع المباريات
        matches_by_season = group_matches_by_season(all_matches)
        
        # الخطوة 3: تدريب جميع النماذج
        trained_models = train_all_models(matches_by_season)
        
        # الخطوة 4: حفظ النماذج
        save_models(trained_models)
        
    except Exception as e:
        log(f"حدث خطأ غير متوقع أثناء عملية التدريب: {e}", "CRITICAL")
        return

    log("--- انتهت عملية تدريب النماذج بنجاح ---", "INFO")


if __name__ == "__main__":
    main()
