# app.py (النسخة النهائية مع دمج الخبير في الواجهة الرسومية)

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# ---------------------------------------------

import streamlit as st

# استيراد الوحدات المشتركة
from common import config
from common.utils import enhanced_team_search
from common.modeling import (
    poisson_matrix_dc, matrix_to_outcomes, top_scorelines,
    calculate_team_form # ضروري لنموذج تعلم الآلة
)

# إعدادات الصفحة
st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")

# --- دوال مساعدة لتحميل البيانات والنماذج مع التخزين المؤقت ---
@st.cache_data
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

@st.cache_data
def load_all_matches() -> List[Dict]:
    """تحميل جميع المباريات."""
    return _load_json(config.DATA_DIR / "matches.json") or []

@st.cache_data
def load_teams_map() -> Optional[Dict]:
    """تحميل خريطة الفرق."""
    return _load_json(config.DATA_DIR / "teams.json")

@st.cache_data
def load_statistical_models() -> Dict[str, dict]:
    """تحميل النماذج الإحصائية."""
    return {
        "averages": _load_json(config.MODELS_DIR / "league_averages.json") or {},
        "factors": _load_json(config.MODELS_DIR / "team_factors.json") or {},
        "elo": _load_json(config.MODELS_DIR / "elo_ratings.json") or {},
        "rho": _load_json(config.MODELS_DIR / "rho_values.json") or {},
    }

@st.cache_data
def load_xgboost_model() -> Optional[xgb.XGBClassifier]:
    """تحميل نموذج تعلم الآلة XGBoost."""
    model_path = config.MODELS_DIR / "xgboost_model.json"
    if not model_path.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

# --- دوال تشغيل السكريبتات من سطر الأوامر ---
def run_cli_script(cmd: List[str]) -> Tuple[bool, str]:
    """دالة عامة لتشغيل أي سكريبت."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', cwd=project_root)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"فشل في تشغيل السكريبت: {e}"

# --- دوال حسابية للتنبؤ ---
def current_season_year(now: datetime) -> int:
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def compute_statistical_prediction(team1_id: int, team2_id: int, comp_code: str, use_elo: bool, topk: int) -> dict:
    """الدالة الأساسية لحساب التوقع الإحصائي."""
    season_year = current_season_year(datetime.now())
    season_key = f"{comp_code}_{season_year}"
    models = load_statistical_models()
    
    if season_key not in models["elo"]:
        season_key = f"{comp_code}_{season_year - 1}"
        if season_key not in models["elo"]:
            raise RuntimeError(f"لا توجد نماذج إحصائية حديثة لمسابقة {comp_code}.")

    # استرجاع كل المعاملات اللازمة من النماذج المحملة
    elo, factors, avgs, rho = (models["elo"].get(season_key, {}), models["factors"].get(season_key, {}),
                               models["averages"].get(season_key, {}), float(models["rho"].get(season_key, 0.0)))
    
    elo_home = float(elo.get(str(team1_id), 1500))
    elo_away = float(elo.get(str(team2_id), 1500))
    home_attack = float(factors.get("attack", {}).get(str(team1_id), 1.0))
    home_defense = float(factors.get("defense", {}).get(str(team1_id), 1.0))
    away_attack = float(factors.get("attack", {}).get(str(team2_id), 1.0))
    away_defense = float(factors.get("defense", {}).get(str(team2_id), 1.0))
    avg_home = float(avgs.get("avg_home_goals", 1.4))
    avg_away = float(avgs.get("avg_away_goals", 1.1))

    lam_home = home_attack * away_defense * avg_home
    lam_away = away_attack * home_defense * avg_away

    if use_elo:
        edge = (elo_home - elo_away) + config.ELO_HFA
        factor = 10 ** (edge / config.ELO_LAMBDA_SCALE)
        lam_home *= factor
        lam_away = max(1e-6, lam_away / factor)

    matrix = poisson_matrix_dc(lam_home, lam_away, rho, max_goals=8)
    p_home, p_draw, p_away = matrix_to_outcomes(matrix)

    result = {
        "meta": {"model_season_used": season_key},
        "model_inputs": {"lambda_home": round(lam_home, 3), "lambda_away": round(lam_away, 3), "rho": round(rho, 3)},
        "probabilities": {"home_win": p_home, "draw": p_draw, "away_win": p_away}
    }
    if topk > 0:
        tops = top_scorelines(matrix, top_k=topk)
        result["top_scorelines"] = [{"home_goals": i, "away_goals": j, "prob": p} for i, j, p in tops]
    return result

def compute_ml_prediction(home_team_id: int, away_team_id: int, competition_code: str) -> Dict:
    """الدالة الأساسية لحساب التوقع باستخدام نموذج تعلم الآلة (الخبير)."""
    model = load_xgboost_model()
    if not model:
        raise RuntimeError("نموذج `xgboost_model.json` غير موجود. يرجى تدريب النموذج (الخطوة 4) أولاً.")

    # تحميل البيانات اللازمة لبناء الميزات
    all_matches = load_all_matches()
    stat_models = load_statistical_models()
    season_year = current_season_year(datetime.now())
    season_key = f"{competition_code}_{season_year}"
    
    if season_key not in stat_models["elo"]:
        season_key = f"{competition_code}_{season_year - 1}"
        if season_key not in stat_models["elo"]:
            raise ValueError(f"لا توجد نماذج إحصائية حديثة لموسم {competition_code} لبناء الميزات.")

    # بناء الميزات التسعة للمباراة الحالية
    season_factors = stat_models["factors"].get(season_key, {})
    season_elo = stat_models["elo"].get(season_key, {})
    
    home_form = calculate_team_form(all_matches, home_team_id, datetime.now(config.TZ), 5)
    away_form = calculate_team_form(all_matches, away_team_id, datetime.now(config.TZ), 5)

    features = {
        'home_attack': season_factors.get("attack", {}).get(str(home_team_id), 1.0),
        'away_attack': season_factors.get("attack", {}).get(str(away_team_id), 1.0),
        'home_defense': season_factors.get("defense", {}).get(str(home_team_id), 1.0),
        'away_defense': season_factors.get("defense", {}).get(str(away_team_id), 1.0),
        'home_elo': season_elo.get(str(home_team_id), 1500.0),
        'away_elo': season_elo.get(str(away_team_id), 1500.0),
        'elo_diff': season_elo.get(str(home_team_id), 1500.0) - season_elo.get(str(away_team_id), 1500.0),
        'home_avg_points': home_form.get("avg_points", 1.0),
        'away_avg_points': away_form.get("avg_points", 1.0),
    }
    
    features_df = pd.DataFrame([features])
    
    # إجراء التنبؤ
    predicted_probabilities = model.predict_proba(features_df)
    
    # تحويل الاحتمالات إلى الشكل المطلوب
    le = LabelEncoder().fit([-1, 0, 1])
    prob_away = predicted_probabilities[0][le.transform([-1])[0]]
    prob_draw = predicted_probabilities[0][le.transform([0])[0]]
    prob_home = predicted_probabilities[0][le.transform([1])[0]]
    
    return {
        "probabilities": {"home_win": prob_home, "draw": prob_draw, "away_win": prob_away},
        "features_used": features
    }

# ==============================================================================
# واجهة المستخدم الرئيسية لـ Streamlit
# ==============================================================================

st.title("⚽ لوحة تحكم متكاملة لتوقع نتائج المباريات")

# --- الشريط الجانبي (Sidebar) ---
with st.sidebar:
    # ... (كود الشريط الجانبي يبقى كما هو) ...

# --- قسم اختيار الفرق ---
st.header("اختيار المباراة")
teams_map = load_teams_map()
if not teams_map:
    st.error("ملف `teams.json` غير موجود. يرجى تشغيل 'تحديث البيانات' أولاً.")
    st.stop()

comp_code = st.selectbox("اختر المسابقة", options=config.TARGET_COMPETITIONS, index=0)
comp_teams = teams_for_comp(teams_map, comp_code)

if not comp_teams:
    st.warning(f"لم يتم العثور على فرق لمسابقة '{comp_code}'.")
else:
    names = [n for n, _ in comp_teams]
    name_to_id = {n: tid for n, tid in comp_teams}
    
    c1, c2 = st.columns(2)
    team1_name = c1.selectbox("الفريق المضيف", options=names, index=0)
    team2_name = c2.selectbox("الفريق الضيف", options=names, index=1 if len(names) > 1 else 0)

    if team1_name == team2_name:
        st.warning("يرجى اختيار فريقين مختلفين.")
    else:
        team1_id = name_to_id[team1_name]
        team2_id = name_to_id[team2_name]

        st.divider()
        
        # --- قسم التنبؤ المزدوج ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("تنبؤ النموذج الإحصائي")
            use_elo_stat = st.checkbox("تفعيل تعديل ELO", value=True, key="stat_elo")
            if st.button("📊 احسب التنبؤ الإحصائي", use_container_width=True):
                try:
                    result = compute_statistical_prediction(team1_id, team2_id, comp_code, use_elo_stat, topk=5)
                    p_home, p_draw, p_away = (result["probabilities"]["home_win"], 
                                              result["probabilities"]["draw"], 
                                              result["probabilities"]["away_win"])
                    st.metric("فوز المضيف", f"{p_home*100:.1f}%")
                    st.metric("تعادل", f"{p_draw*100:.1f}%")
                    st.metric("فوز الضيف", f"{p_away*100:.1f}%")
                    with st.expander("عرض مدخلات النموذج"): st.json(result["model_inputs"])
                except Exception as e:
                    st.error(f"فشل التنبؤ: {e}")

        with c2:
            st.subheader("تنبؤ نموذج تعلم الآلة (الخبير)")
            if st.button("🧠 احسب تنبؤ الخبير (ML)", type="primary", use_container_width=True):
                try:
                    result = compute_ml_prediction(team1_id, team2_id, comp_code)
                    p_home, p_draw, p_away = (result["probabilities"]["home_win"], 
                                              result["probabilities"]["draw"], 
                                              result["probabilities"]["away_win"])
                    st.metric("فوز المضيف", f"{p_home*100:.1f}%")
                    st.metric("تعادل", f"{p_draw*100:.1f}%")
                    st.metric("فوز الضيف", f"{p_away*100:.1f}%")
                    with st.expander("عرض الميزات المستخدمة"): st.json(result["features_used"])
                except Exception as e:
                    st.error(f"فشل التنبؤ: {e}")

