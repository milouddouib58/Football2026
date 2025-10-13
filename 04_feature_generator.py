# 04_feature_generator.py
# -----------------------------------------------------------------------------
# الوصف:
# يقوم هذا السكريبت بإنشاء مجموعة بيانات التدريب (Features) لنموذج تعلم الآلة.
# ملاحظة مهمة: الاعتماد على Team Factors/Elo الخاصة بالموسم ككل قد يسبب تسرب معلومات.
# يُستحسن مستقبلًا حساب هذه الميزات "زمنيًا" بحيث تعكس القيم قبل كل مباراة.
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

    feature_list = []
    for match in all_matches:
        dt = parse_date_safe(match.get("utcDate"))
        hg, ag = parse_score(match)
        if not dt or hg is None:
            continue

        season_year = match.get('season', {}).get('startDate', '1900')[:4]
        comp_code = match.get('competition', {}).get('code', 'UNK')
        season_key = f"{comp_code}_{season_year}"

        h_id_val = match.get("homeTeam", {}).get("id")
        a_id_val = match.get("awayTeam", {}).get("id")
        if not h_id_val or not a_id_val:
            continue  # IDs غير متاحة
        h_id = str(h_id_val)
        a_id = str(a_id_val)

        # استرجاع الميزات الإحصائية من النماذج المدربة
        season_factors = team_factors.get(season_key, {})
        season_elo = elo_ratings.get(season_key, {})
        season_avg = league_averages.get(season_key, {})
        if not all([season_factors, season_elo, season_avg]):
            continue  # تخطي المباراة إذا لم تكن نماذجها موجودة

        # حساب "الفورمة" لكل فريق قبل المباراة (تحذير: قد تُحسب من كامل التاريخ)
        home_form = calculate_team_form(all_matches, int(h_id), dt, num_matches=5)
        away_form = calculate_team_form(all_matches, int(a_id), dt, num_matches=5)

        features = {
            "match_id": match["id"],
            "date": dt.isoformat(),
            "home_team_id": h_id,
            "away_team_id": a_id,

            # Team Factors
            "home_attack": season_factors.get("attack", {}).get(h_id, 1.0),
            "away_attack": season_factors.get("attack", {}).get(a_id, 1.0),
            "home_defense": season_factors.get("defense", {}).get(h_id, 1.0),
            "away_defense": season_factors.get("defense", {}).get(a_id, 1.0),

            # Elo
            "home_elo": season_elo.get(h_id, 1500.0),
            "away_elo": season_elo.get(a_id, 1500.0),
            "elo_diff": season_elo.get(h_id, 1500.0) - season_elo.get(a_id, 1500.0),

            # Form
            "home_avg_points": home_form.get("avg_points", 1.0),
            "away_avg_points": away_form.get("avg_points", 1.0),

            # الهدف (Target)
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "result": 1 if hg > ag else (0 if hg == ag else -1)  # 1: فوز المضيف, 0: تعادل, -1: فوز الضيف
        }
        feature_list.append(features)

    if not feature_list:
        log("لم يتم إنشاء أي ميزات. يرجى التحقق من الملفات المدخلة.", "CRITICAL")
        return

    df = pd.DataFrame(feature_list)
    output_path = config.DATA_DIR / "ml_dataset.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    log(f"تم إنشاء مجموعة بيانات تعلم الآلة بنجاح ({len(df)} مباراة).", "INFO")
    log(f"تم الحفظ في: {output_path}", "INFO")

if __name__ == "__main__":
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    run_feature_generator()
