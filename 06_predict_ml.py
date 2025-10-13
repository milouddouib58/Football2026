# 06_predict_ml.py
# -----------------------------------------------------------------------------
# الوصف:
# السكريبت النهائي للتنبؤ بنتيجة مباراة قادمة باستخدام نموذج XGBoost المدرب.
# -----------------------------------------------------------------------------
import json
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from common import config
from common.utils import log
from common.modeling import calculate_team_form

def predict_match(home_team_id: int, away_team_id: int, competition_code: str, season_start_year: int):
    """ الدالة الرئيسية للتنبؤ بمباراة واحدة محددة. """
    log(f"--- بدء التنبؤ للمباراة: {home_team_id} vs {away_team_id} ---", "INFO")

    # --- 1. تحميل جميع النماذج والبيانات المطلوبة ---
    try:
        with open(config.DATA_DIR / "matches.json", 'r', encoding='utf-8') as f:
            all_matches = json.load(f)
        with open(config.MODELS_DIR / "team_factors.json", 'r', encoding='utf-8') as f:
            team_factors = json.load(f)
        with open(config.MODELS_DIR / "elo_ratings.json", 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)

        # تحميل نموذج تعلم الآلة
        model = xgb.XGBClassifier()
        model.load_model(config.MODELS_DIR / "xgboost_model.json")
    except (IOError, xgb.core.XGBoostError) as e:
        log(f"فشل في تحميل أحد الملفات المطلوبة: {e}. يرجى تشغيل السكريبتات السابقة.", "CRITICAL")
        return

    # --- 2. بناء الميزات للمباراة المستهدفة ---
    season_key = f"{competition_code}_{season_start_year}"
    h_id_str, a_id_str = str(home_team_id), str(away_team_id)
    prediction_date = datetime.now()  # التاريخ الحالي

    season_factors = team_factors.get(season_key)
    season_elo = elo_ratings.get(season_key)
    if not season_factors or not season_elo:
        log(f"لم يتم العثور على نماذج إحصائية للموسم {season_key}. لا يمكن المتابعة.", "ERROR")
        return

    home_form = calculate_team_form(all_matches, home_team_id, prediction_date, num_matches=5)
    away_form = calculate_team_form(all_matches, away_team_id, prediction_date, num_matches=5)

    features_dict = {
        'home_attack': [season_factors.get("attack", {}).get(h_id_str, 1.0)],
        'away_attack': [season_factors.get("attack", {}).get(a_id_str, 1.0)],
        'home_defense': [season_factors.get("defense", {}).get(h_id_str, 1.0)],
        'away_defense': [season_factors.get("defense", {}).get(a_id_str, 1.0)],
        'home_elo': [season_elo.get(h_id_str, 1500.0)],
        'away_elo': [season_elo.get(a_id_str, 1500.0)],
        'elo_diff': [season_elo.get(h_id_str, 1500.0) - season_elo.get(a_id_str, 1500.0)],
        'home_avg_points': [home_form.get("avg_points", 1.0)],
        'away_avg_points': [away_form.get("avg_points", 1.0)],
    }
    features_df = pd.DataFrame.from_dict(features_dict)
    log("الميزات التي تم إنشاؤها للمباراة:", "DEBUG")
    print(features_df)

    # --- 3. إجراء التنبؤ باستخدام نموذج XGBoost ---
    log("إجراء التنبؤ باستخدام نموذج تعلم الآلة...", "INFO")
    predicted_probabilities = model.predict_proba(features_df)

    # الترتيب: (-1 -> 0), (0 -> 1), (1 -> 2)
    le = LabelEncoder()
    le.fit([-1, 0, 1])
    prob_away = float(predicted_probabilities[0][le.transform([-1])[0]])
    prob_draw = float(predicted_probabilities[0][le.transform([0])[0]])
    prob_home = float(predicted_probabilities[0][le.transform([1])[0]])

    # --- 4. عرض النتائج ---
    print("\n" + "="*40)
    log(f"📊 نتائج التنبؤ للمباراة: {home_team_id} vs {away_team_id}", "RESULT")
    print("="*40)
    print(f" - احتمال فوز الفريق المضيف: {prob_home:.2%}")
    print(f" - احتمال التعادل: {prob_draw:.2%}")
    print(f" - احتمال فوز الفريق الضيف: {prob_away:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # مثال: TARGET_HOME_TEAM_ID = 65  # Manchester City
    TARGET_HOME_TEAM_ID = 65
    TARGET_AWAY_TEAM_ID = 64  # Liverpool
    TARGET_COMP_CODE = "PL"
    TARGET_SEASON_YEAR = 2025

    predict_match(
        home_team_id=TARGET_HOME_TEAM_ID,
        away_team_id=TARGET_AWAY_TEAM_ID,
        competition_code=TARGET_COMP_CODE,
        season_start_year=TARGET_SEASON_YEAR
    )
