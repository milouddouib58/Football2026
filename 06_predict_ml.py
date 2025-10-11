# -----------------------------------------------------------------------------
# 06_predict_ml.py
# -----------------------------------------------------------------------------
# الوصف:
#   السكريبت النهائي للتنبؤ بنتيجة مباراة قادمة. يقوم بالخطوات التالية:
#   1. تحميل جميع النماذج المدربة (الإحصائية ونموذج تعلم الآلة).
#   2. تجميع الميزات (Features) للمباراة المستهدفة كما فعلنا في مرحلة التدريب.
#   3. إدخال هذه الميزات إلى نموذج XGBoost المدرب للحصول على احتمالات النتائج.
#   4. عرض التنبؤ النهائي بطريقة واضحة.
#
# الاستخدام:
#   عدّل المتغيرات في دالة `main` لتحديد المباراة التي تريد التنبؤ بها.
# -----------------------------------------------------------------------------

import json
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from common import config
from common.utils import log, parse_date_safe
from common.modeling import calculate_team_form

def predict_match(home_team_id: int, away_team_id: int, competition_code: str, season_start_year: int):
    """
    الدالة الرئيسية للتنبؤ بمباراة واحدة محددة.
    """
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

    # --- 2. بناء الميزات (Features) للمباراة المستهدفة ---
    season_key = f"{competition_code}_{season_start_year}"
    h_id_str, a_id_str = str(home_team_id), str(away_team_id)
    prediction_date = datetime.now() # تاريخ التنبؤ هو الآن

    # استرجاع الميزات الإحصائية من النماذج
    season_factors = team_factors.get(season_key)
    season_elo = elo_ratings.get(season_key)

    if not season_factors or not season_elo:
        log(f"لم يتم العثور على نماذج إحصائية للموسم {season_key}. لا يمكن المتابعة.", "ERROR")
        return

    # حساب "الفورمة" الحالية لكل فريق
    home_form = calculate_team_form(all_matches, home_team_id, prediction_date, num_matches=5)
    away_form = calculate_team_form(all_matches, away_team_id, prediction_date, num_matches=5)

    # تجميع الميزات في قاموس بنفس الترتيب المستخدم في التدريب
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
    
    # النموذج يتوقع احتمالات لكل فئة (0, 1, 2)
    predicted_probabilities = model.predict_proba(features_df)
    
    # يجب أن نعرف أي فئة تمثل أي نتيجة
    # بناءً على `05_train_ml_model.py`, الترتيب هو: فوز الضيف (-1), تعادل (0), فوز المضيف (1)
    le = LabelEncoder()
    le.fit([-1, 0, 1]) # fit with the original classes to ensure correct order
    
    prob_away = predicted_probabilities[0][le.transform([-1])[0]]
    prob_draw = predicted_probabilities[0][le.transform([0])[0]]
    prob_home = predicted_probabilities[0][le.transform([1])[0]]

    # --- 4. عرض النتائج ---
    print("\n" + "="*40)
    log(f"📊  نتائج التنبؤ للمباراة: {home_team_id} vs {away_team_id}", "RESULT")
    print("="*40)
    print(f"  - احتمال فوز الفريق المضيف: {prob_home:.2%}")
    print(f"  - احتمال التعادل:           {prob_draw:.2%}")
    print(f"  - احتمال فوز الفريق الضيف: {prob_away:.2%}")
    print("="*40 + "\n")


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 🎯  هنا تحدد المباراة التي تريد التنبؤ بها
    # --------------------------------------------------------------------------
    # مثال: التنبؤ بمباراة في الدوري الإنجليزي (PL) لموسم 2025/26
    # لنفترض أن ID فريق مانشستر سيتي هو 65 و ID فريق ليفربول هو 64
    
    TARGET_HOME_TEAM_ID = 65      # مثال: Manchester City
    TARGET_AWAY_TEAM_ID = 64      # مثال: Liverpool
    TARGET_COMP_CODE = "PL"     # رمز المسابقة
    TARGET_SEASON_YEAR = 2025   # سنة بداية الموسم
    
    # --------------------------------------------------------------------------

    predict_match(
        home_team_id=TARGET_HOME_TEAM_ID,
        away_team_id=TARGET_AWAY_TEAM_ID,
        competition_code=TARGET_COMP_CODE,
        season_start_year=TARGET_SEASON_YEAR
    )
