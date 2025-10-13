# -----------------------------------------------------------------------------
# 03_backtester.py (النسخة الكاملة والنهائية)
# -----------------------------------------------------------------------------
# الوصف:
#   يقوم هذا السكريبت بإجراء اختبار تاريخي (Backtesting) للنموذج الإحصائي
#   لتقييم أدائه على البيانات السابقة بشكل موضوعي.
#
# آلية العمل:
#   1. يمر على جميع المباريات بترتيب زمني.
#   2. لكل يوم، يقوم "بإعادة تدريب" النماذج باستخدام كل البيانات المتاحة *قبل* هذا اليوم.
#   3. يستخدم النماذج المدربة للتنبؤ بنتائج مباريات ذلك اليوم.
#   4. يخزن التنبؤات والنتائج الفعلية في ملف CSV لتقييمها.
# -----------------------------------------------------------------------------

import json
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, parse_date_safe, parse_score
from common.modeling import (
    calculate_league_averages,
    build_team_factors,
    build_elo_ratings,
    fit_dc_rho_mle,
    poisson_matrix_dc,
    matrix_to_outcomes
)

def run_backtester():
    log("--- بدء عملية الاختبار التاريخي (Backtester) ---", "INFO")

    # --- 1. تحميل وتنظيم البيانات ---
    try:
        with open(config.DATA_DIR / "matches.json", 'r', encoding='utf-8') as f:
            all_matches = json.load(f)
    except IOError:
        log("ملف matches.json غير موجود. يرجى تشغيل 01_pipeline.py أولاً.", "CRITICAL")
        return

    # فرز جميع المباريات حسب التاريخ (ضروري جداً)
    all_matches.sort(key=lambda m: parse_date_safe(m.get("utcDate", "")))
    
    # تجميع المباريات حسب تاريخ إقامتها لتسهيل المعالجة
    matches_by_date = defaultdict(list)
    for match in all_matches:
        dt = parse_date_safe(match.get("utcDate"))
        if dt:
            matches_by_date[dt.date()].append(match)

    log(f"تم تجميع المباريات في {len(matches_by_date)} يومًا مختلفًا.", "INFO")

    # --- 2. تنفيذ حلقة الاختبار التاريخي ---
    predictions_log = []
    sorted_dates = sorted(matches_by_date.keys())

    for i, current_date in enumerate(sorted_dates):
        log(f"المعالجة لليوم: {current_date} ({i+1}/{len(sorted_dates)})", "DEBUG")

        # إنشاء نقطة قطع زمنية "مدركة للمنطقة الزمنية" (aware) لحل مشكلة المقارنة
        cutoff_datetime = datetime.combine(current_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        # تجميع كل المباريات التاريخية التي حدثت قبل اليوم الحالي
        historical_matches = [
            m for m in all_matches if parse_date_safe(m.get("utcDate")) < cutoff_datetime
        ]
        
        if len(historical_matches) < 50: # نتجاهل الأيام الأولى لضمان وجود بيانات كافية للتدريب
            continue

        # تقسيم المباريات التاريخية حسب الموسم للتدريب
        matches_by_season = defaultdict(list)
        for m in historical_matches:
            season_year = m.get('season', {}).get('startDate', '1900')[:4]
            comp_code = m.get('competition', {}).get('code', 'UNK')
            season_key = f"{comp_code}_{season_year}"
            matches_by_season[season_key].append(m)

        # --- 3. إعادة تدريب النماذج على البيانات التاريخية ---
        models = {
            "team_factors": {}, "elo_ratings": {},
            "league_averages": {}, "rho_values": {}
        }
        
        for season_key, season_matches in matches_by_season.items():
            if len(season_matches) < 20: continue
            
            league_avgs = calculate_league_averages(season_matches)
            factors_A, factors_D = build_team_factors(season_matches, league_avgs, cutoff_datetime)
            elo = build_elo_ratings(season_matches)
            rho = fit_dc_rho_mle(season_matches, factors_A, factors_D, league_avgs)

            models["league_averages"][season_key] = league_avgs
            models["team_factors"][season_key] = {"attack": factors_A, "defense": factors_D}
            models["elo_ratings"][season_key] = elo
            models["rho_values"][season_key] = rho
        
        # --- 4. التنبؤ بمباريات اليوم الحالي ---
        matches_for_today = matches_by_date[current_date]
        for match in matches_for_today:
            season_year = match.get('season', {}).get('startDate', '1900')[:4]
            comp_code = match.get('competition', {}).get('code', 'UNK')
            season_key = f"{comp_code}_{season_year}"
            
            h_id = str(match.get("homeTeam", {}).get("id"))
            a_id = str(match.get("awayTeam", {}).get("id"))
            hg, ag = parse_score(match)

            season_models = {
                "avg": models["league_averages"].get(season_key),
                "factors": models["team_factors"].get(season_key),
                "elo": models["elo_ratings"].get(season_key),
                "rho": models["rho_values"].get(season_key, 0.0)
            }

            if not all(season_models.values()): continue

            avg_home = season_models["avg"]["avg_home_goals"]
            avg_away = season_models["avg"]["avg_away_goals"]
            
            attack_h = season_models["factors"]["attack"].get(h_id, 1.0)
            defense_a = season_models["factors"]["defense"].get(a_id, 1.0)
            attack_a = season_models["factors"]["attack"].get(a_id, 1.0)
            defense_h = season_models["factors"]["defense"].get(h_id, 1.0)

            lambda_h = avg_home * attack_h * defense_a
            lambda_a = avg_away * attack_a * defense_h
            
            prob_matrix = poisson_matrix_dc(lambda_h, lambda_a, season_models["rho"])
            p_home, p_draw, p_away = matrix_to_outcomes(prob_matrix)

            if hg is not None:
                predictions_log.append({
                    "date": current_date.isoformat(), "match_id": match["id"],
                    "home_team_id": h_id, "away_team_id": a_id,
                    "actual_home_goals": hg, "actual_away_goals": ag,
                    "predicted_prob_home": p_home, "predicted_prob_draw": p_draw,
                    "predicted_prob_away": p_away,
                    "elo_home": season_models["elo"].get(h_id, 1500),
                    "elo_away": season_models["elo"].get(a_id, 1500)
                })

    # --- 5. حفظ نتائج الاختبار وتحليلها ---
    if not predictions_log:
        log("لم يتم إنشاء أي تنبؤات. قد تكون البيانات غير كافية.", "WARNING")
        return

    output_path = config.DATA_DIR / "backtest_results.csv"
    df = pd.DataFrame(predictions_log)
    df.to_csv(output_path, index=False, encoding='utf-8')
    log(f"--- انتهى الاختبار التاريخي. تم حفظ النتائج في: {output_path} ---", "INFO")
    
    df['actual_result'] = df.apply(
        lambda row: 'H' if row['actual_home_goals'] > row['actual_away_goals']
        else ('D' if row['actual_home_goals'] == row['actual_away_goals'] else 'A'),
        axis=1
    )
    
    brier_h = ((df['predicted_prob_home'] - (df['actual_result'] == 'H'))**2).mean()
    brier_d = ((df['predicted_prob_draw'] - (df['actual_result'] == 'D'))**2).mean()
    brier_a = ((df['predicted_prob_away'] - (df['actual_result'] == 'A'))**2).mean()
    total_brier = (brier_h + brier_d + brier_a) / 3
    
    log(f"متوسط Brier Score للنتائج: {total_brier:.4f} (كلما قل كان أفضل)", "SUCCESS")

# --- 6. نقطة الدخول لتشغيل السكريبت ---
if __name__ == "__main__":
    # التأكد من وجود مجلد البيانات
    config.DATA_DIR.mkdir(exist_ok=True)
    # تشغيل الدالة الرئيسية للاختبار
    run_backtester()
