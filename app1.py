# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# app.py (النسخة النهائية المنظمة والمصححة)
# -----------------------------------------------------------------------------
# الوصف:
#   واجهة مستخدم رسومية باستخدام Streamlit لمشروع التنبؤ بنتائج مباريات كرة القدم.
# -----------------------------------------------------------------------------

# --- 1. الإعداد الأولي والاستيراد ---
import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import streamlit as st

# الحل النهائي لمشكلة المسار (Path) لضمان العثور على مجلد 'common'
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# استيراد الوحدات من مجلد 'common' بأمان
try:
    from common import config
    from common.utils import enhanced_team_search
    from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines
except ImportError as e:
    st.error(
        f"**خطأ فادح في الاستيراد:** لم يتم العثور على مجلد `common` أو ملفاته.\n"
        f"تأكد من أن بنية المشروع صحيحة (وجود `common/__init__.py`).\n"
        f"التفاصيل: {e}"
    )
    st.stop()

# --- 2. إعداد الصفحة ---
st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")

# --- 3. دوال تحميل البيانات والنماذج ---
@st.cache_data
def load_data_file(filename: str) -> Optional[dict]:
    """تحميل ملف بيانات (مثل teams.json) مع معالجة الأخطاء."""
    path = config.DATA_DIR / filename
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"خطأ في قراءة ملف JSON: {filename}. قد يكون الملف تالفًا.")
            return None
    return None

@st.cache_data
def load_all_models() -> Dict[str, dict]:
    """تحميل جميع ملفات النماذج المدربة."""
    models = {}
    model_dir = config.MODELS_DIR
    for name in ["league_averages", "team_factors", "elo_ratings", "rho_values"]:
        models[name] = load_data_file(f"../models/{name}.json") or {} # المسار النسبي الصحيح
    return models

# --- 4. المنطق الحسابي للتنبؤ ---
def compute_prediction(home_name: str, away_name: str, comp_code: str, use_elo: bool, topk: int) -> dict:
    teams_map = load_data_file("teams.json")
    if not teams_map:
        raise FileNotFoundError("ملف `teams.json` غير موجود. يرجى تشغيل 'بناء البيانات' أولاً.")

    home_id = enhanced_team_search(home_name, teams_map, comp_code)
    away_id = enhanced_team_search(away_name, teams_map, comp_code)
    if not home_id or not away_id:
        raise ValueError(f"لم يتم العثور على: '{home_name}' أو '{away_name}' في المسابقة المحددة.")

    now = datetime.now()
    season_year = now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1
    season_key = f"{comp_code}_{season_year}"
    
    models = load_all_models()
    if not models["elo_ratings"].get(season_key):
        season_key = f"{comp_code}_{season_year - 1}"
        if not models["elo_ratings"].get(season_key):
            raise FileNotFoundError(f"لا توجد نماذج للمسابقة {comp_code}. يرجى 'تدريب النماذج' أولاً.")

    elo, factors, avgs, rho_data = models["elo_ratings"].get(season_key, {}), models["team_factors"].get(season_key, {}), models["league_averages"].get(season_key, {}), models["rho_values"]
    rho = rho_data.get(season_key, 0.0)

    elo_home, elo_away = float(elo.get(str(home_id), 1500)), float(elo.get(str(away_id), 1500))
    home_attack, home_defense = float(factors.get("attack", {}).get(str(home_id), 1)), float(factors.get("defense", {}).get(str(home_id), 1))
    away_attack, away_defense = float(factors.get("attack", {}).get(str(away_id), 1)), float(factors.get("defense", {}).get(str(away_id), 1))
    avg_home, avg_away = float(avgs.get("avg_home_goals", 1.4)), float(avgs.get("avg_away_goals", 1.1))

    lam_home, lam_away = home_attack * away_defense * avg_home, away_attack * home_defense * avg_away

    if use_elo:
        edge = (elo_home - elo_away) + config.ELO_HFA
        factor = 10 ** (edge / config.ELO_LAMBDA_SCALE)
        lam_home *= factor
        lam_away = max(1e-9, lam_away / factor)

    matrix = poisson_matrix_dc(lam_home, lam_away, rho)
    p_home, p_draw, p_away = matrix_to_outcomes(matrix)
    
    return {
        "match_info": f"{home_name} (المضيف) vs {away_name} (الضيف) - {comp_code}",
        "model_season": season_key,
        "probabilities": {"home": p_home, "draw": p_draw, "away": p_away},
        "top_scores": [{"score": f"{s[0]}-{s[1]}", "prob": s[2]} for s in top_scorelines(matrix, topk)],
    }

# --- 5. أدوات الواجهة وتشغيل السكريبتات ---
def run_cli_command(script_name: str, args: List[str] = []) -> Tuple[bool, str]:
    cmd = [sys.executable, str(APP_ROOT / script_name)] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0, f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return False, f"فشل في تشغيل العملية: {e}"

def get_teams_for_competition(teams_map: Dict, comp_code: str) -> List[str]:
    team_names = []
    for team_data in teams_map.values():
        if comp_code in team_data.get("competitions", []):
            names = team_data.get("names", [])
            if names:
                primary_name = sorted(names, key=lambda n: (int(" " in n), len(n)), reverse=True)[0]
                team_names.append(primary_name)
    return sorted(team_names)

# --- 6. بناء الواجهة الرسومية ---
st.title("⚽ متنبئ نتائج مباريات كرة القدم")
st.caption(f"الإصدار: {config.VERSION}")

with st.sidebar:
    st.header("⚙️ الإعداد والتحكم")
    years = st.number_input("عدد المواسم لجلبها", 1, 10, 3)
    if st.button("🏗️ بناء البيانات (Pipeline)"):
        with st.spinner("جارٍ بناء قاعدة البيانات..."):
            ok, logs = run_cli_command("01_pipeline.py", ["--years", str(years)])
            st.cache_data.clear() # مسح الكاش بعد تحديث البيانات
            st.toast("اكتمل بناء البيانات!", icon="✅" if ok else "❌")
            if not ok:
                with st.expander("عرض سجل الأخطاء"): st.code(logs)

    if st.button("🧠 تدريب النماذج (Trainer)"):
        with st.spinner("جارٍ تدريب النماذج..."):
            ok, logs = run_cli_command("02_trainer.py")
            st.cache_data.clear() # مسح الكاش بعد تحديث النماذج
            st.toast("اكتمل تدريب النماذج!", icon="✅" if ok else "❌")
            if not ok:
                with st.expander("عرض سجل الأخطاء"): st.code(logs)
    
    st.divider()
    st.header("🔧 خيارات التنبؤ")
    use_elo = st.checkbox("تفعيل تعديل ELO", value=True)
    topk = st.slider("عدد أعلى النتائج", 0, 10, 5)

st.header("🔮 التنبؤ بمباراة")
teams_map = load_data_file("teams.json")
if not teams_map:
    st.warning("لم يتم العثور على بيانات الفرق. يرجى تشغيل 'بناء البيانات' أولاً.")
else:
    all_comps = sorted(config.TARGET_COMPETITIONS)
    comp_code = st.selectbox("اختر المسابقة", options=all_comps)
    
    comp_teams = get_teams_for_competition(teams_map, comp_code)
    
    if not comp_teams:
        st.warning(f"لا توجد فرق للمسابقة '{comp_code}'. تأكد من إضافتها لـ TARGET_COMPETITIONS وتشغيل بناء البيانات.")
    else:
        col1, col2 = st.columns(2)
        home_name = col1.selectbox("الفريق المضيف", options=comp_teams)
        away_name = col2.selectbox("الفريق الضيف", options=comp_teams, index=min(1, len(comp_teams)-1))

        if st.button("احسب التنبؤ", type="primary", use_container_width=True):
            if home_name == away_name:
                st.error("يرجى اختيار فريقين مختلفين.")
            else:
                try:
                    with st.spinner("جاري حساب الاحتمالات..."):
                        result = compute_prediction(home_name, away_name, comp_code, use_elo, topk)

                    st.subheader(result["match_info"])
                    st.caption(f"تم استخدام نماذج موسم: {result['model_season']}")
                    
                    probs = result["probabilities"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("فوز المضيف", f"{probs['home']:.1%}")
                    c2.metric("تعادل", f"{probs['draw']:.1%}")
                    c3.metric("فوز الضيف", f"{probs['away']:.1%}")

                    if topk > 0 and result["top_scores"]:
                        st.write("**أعلى النتائج المحتملة:**")
                        df_scores = st.dataframe([{"النتيجة": s["score"], "الاحتمال": f"{s['prob']:.2%}"} for s in result["top_scores"]], use_container_width=True)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التنبؤ: {e}")
