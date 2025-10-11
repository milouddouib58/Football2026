# -----------------------------------------------------------------------------
# 04_feature_generator.py
# -----------------------------------------------------------------------------
# الوصف:
#   يقوم هذا السكريبت بإنشاء مجموعة بيانات التدريب (Features) لنموذج تعلم الآلة.
#   يقرأ بيانات المباريات الخام والنماذج الإحصائية التي تم تدريبها مسبقًا
#   (Elo, Team Factors, etc.) ويقوم بتجميعها في ملف CSV واحد منظم.
#   كل صف في هذا الملف يمثل مباراة واحدة مع جميع ميزاتها والنتيجة الفعلية.
# -----------------------------------------------------------------------------

import json
import pandas as pd
from datetime import datetime

from common import config
from common.utils import log, parse_date_safe, parse_score
from common.modeling import calculate_team_form

def run_feature_generator():
    log("--- بدء عملية إنشاء الميزات لتعلم الآلة ---", "INFO")

    # --- 1. تحميل كل البيانات اللازمة ---
    try:
        with open(config.DATA_DIR / "matches.json", 'r', encoding='utf-8') as f:
            all_matches = json.load(f)
        with open(config.MODELS_DIR / "team_factors.json", 'r', encoding='utf-8') as f:
            team_factors = json.load(f)
        with open(config.MODELS_DIR / "elo_ratings.json", 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)
        with open(config.MODELS_DIR / "league_averages.json", 'r', encoding='utf-8') as f:
            league_averages = json.load(f)
    except IOError as e:
        log(f"فشل في تحميل أحد الملفات المطلوبة: {e}. يرجى تشغيل السكريبتات 01 و 02 أولاً.", "CRITICAL")
        return

    # --- 2. إنشاء قائمة من الميزات لكل مباراة ---
    feature_list = []

    for match in all_matches:
        dt = parse_date_safe(match.get("utcDate"))
        hg, ag = parse_score(match)
        if not dt or hg is None:
            continue

        season_year = match.get('season', {}).get('startDate', '1900')[:4]
        comp_code = match.get('competition', {}).get('code', 'UNK')
        season_key = f"{comp_code}_{season_year}"

        h_id = str(match.get("homeTeam", {}).get("id"))
        a_id = str(match.get("awayTeam", {}).get("id"))

        # استرجاع الميزات الإحصائية من النماذج المدربة
        season_factors = team_factors.get(season_key, {})
        season_elo = elo_ratings.get(season_key, {})
        season_avg = league_averages.get(season_key, {})

        if not all([season_factors, season_elo, season_avg]):
            continue # تخطي المباراة إذا لم تكن نماذجها موجودة

        # حساب "الفورمة" لكل فريق قبل المباراة
        home_form = calculate_team_form(all_matches, int(h_id), dt, num_matches=5)
        away_form = calculate_team_form(all_matches, int(a_id), dt, num_matches=5)

        # إنشاء قاموس يحتوي على ميزات هذه المباراة
        features = {
            "match_id": match["id"],
            "date": dt.isoformat(),
            "home_team_id": h_id,
            "away_team_id": a_id,
            
            # ميزات من Team Factors
            "home_attack": season_factors.get("attack", {}).get(h_id, 1.0),
            "away_attack": season_factors.get("attack", {}).get(a_id, 1.0),
            "home_defense": season_factors.get("defense", {}).get(h_id, 1.0),
            "away_defense": season_factors.get("defense", {}).get(a_id, 1.0),
            
            # ميزات Elo
            "home_elo": season_elo.get(h_id, 1500.0),
            "away_elo": season_elo.get(a_id, 1500.0),
            "elo_diff": season_elo.get(h_id, 1500.0) - season_elo.get(a_id, 1500.0),
            
            # ميزات الفورمة
            "home_avg_points": home_form.get("avg_points", 1.0),
            "away_avg_points": away_form.get("avg_points", 1.0),
            
            # الهدف (Target) الذي سيتعلمه النموذج
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "result": 1 if hg > ag else (0 if hg == ag else -1) # 1: فوز المضيف, 0: تعادل, -1: فوز الضيف
        }
        feature_list.append(features)

    if not feature_list:
        log("لم يتم إنشاء أي ميزات. يرجى التحقق من الملفات المدخلة.", "CRITICAL")
        return

    # --- 3. حفظ مجموعة البيانات ---
    df = pd.DataFrame(feature_list)
    output_path = config.DATA_DIR / "ml_dataset.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')

    log(f"تم إنشاء مجموعة بيانات تعلم الآلة بنجاح ({len(df)} مباراة).", "INFO")
    log(f"تم الحفظ في: {output_path}", "INFO")


if __name__ == "__main__":
    config.DATA_DIR.mkdir(exist_ok=True)
    run_feature_generator()
