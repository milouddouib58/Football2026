# app.py
# -----------------------------------------------------------------------------
# لوحة Streamlit مع أزرار لإدارة البيانات والنماذج وزر تنبؤ باستخدام Predictor.
# -----------------------------------------------------------------------------

import sys
import os
import json
import subprocess
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# ---------------------------------------------

import streamlit as st
from common import config
from common.utils import enhanced_team_search, log
from common.modeling import calculate_team_form
from predictor import Predictor  # استخدام الكلاس الموحد

# إعدادات الصفحة
st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")


# --- دوال مساعدة لتحميل البيانات مع كاش ---

@st.cache_data
def _load_json(path: Path) -> Optional[dict]:
    """تحميل ملف JSON مع معالجة الأخطاء."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_teams_map() -> Optional[Dict]:
    """تحميل بيانات الفرق من ملف JSON."""
    return _load_json(config.DATA_DIR / "teams.json")

@st.cache_data
def load_all_matches() -> Optional[List]:
    """تحميل جميع المباريات من ملف JSON."""
    return _load_json(config.DATA_DIR / "matches.json")

@st.cache_resource
def load_xgboost_model() -> Optional[xgb.XGBClassifier]:
    """تحميل نموذج XGBoost المدرب."""
    try:
        model = xgb.XGBClassifier()
        model.load_model(config.MODELS_DIR / "xgboost_model.json")
        return model
    except Exception:
        return None

# --- كائن Predictor مع كاش على مستوى الموارد ---

@st.cache_resource
def get_predictor() -> Predictor:
    """إنشاء وتحميل كائن Predictor الإحصائي مع التخزين المؤقت."""
    return Predictor()


# --- دوال الواجهة والمساعدة ---

def run_cli_script(cmd: List[str]) -> Tuple[bool, str]:
    """تشغيل أي سكريبت والتقاط مخرجاته."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"فشل في تشغيل السكريبت: {e}"

def _primary_name(names: List[str]) -> str:
    """اختيار الاسم الأساسي للفريق من قائمة الأسماء المتاحة."""
    names = [n for n in (names or []) if n]
    if not names:
        return "Unknown"
    def score(n: str) -> Tuple[int, int, int]:
        return (int(" " in n), len(n), -int(n.isupper()))
    return sorted(names, key=score, reverse=True)[0]

def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    """الحصول على قائمة الفرق لمسابقة معينة."""
    out = sorted(
        [
            (_primary_name(t.get("names", [])), t.get("id"))
            for t in teams_map.values()
            if comp_code in t.get("competitions", []) and t.get("id")
        ],
        key=lambda x: x[0].lower(),
    )
    return out

def compute_statistical_prediction(team1_name: str, team2_name: str, comp_code: str, use_elo: bool, topk: int):
    """استدعاء Predictor الإحصائي لحساب التنبؤ."""
    pred = get_predictor()
    result = pred.predict(team1_name, team2_name, comp_code, topk=topk, use_elo=use_elo)
    return result

def compute_ml_prediction(home_team_id: int, away_team_id: int, comp_code: str):
    """حساب التنبؤ باستخدام نموذج الخبير (XGBoost)."""
    model = load_xgboost_model()
    team_factors = _load_json(config.MODELS_DIR / "team_factors.json")
    elo_ratings = _load_json(config.MODELS_DIR / "elo_ratings.json")
    all_matches = load_all_matches()

    if not all([model, team_factors, elo_ratings, all_matches]):
        raise FileNotFoundError("أحد النماذج أو ملفات البيانات اللازمة لنموذج الخبير غير موجود.")

    # اختيار آخر موسم متاح للنماذج
    predictor = get_predictor() # نستعير منه دالة اختيار الموسم
    season_key = predictor._select_season_key(comp_code)
    
    h_id_str, a_id_str = str(home_team_id), str(away_team_id)
    
    # معالجة مشكلة المنطقة الزمنية باستخدام UTC
    prediction_date = datetime.now(timezone.utc)

    season_factors = team_factors.get(season_key)
    season_elo = elo_ratings.get(season_key)

    if not season_factors or not season_elo:
        raise ValueError(f"لم يتم العثور على نماذج إحصائية للموسم {season_key} اللازمة للنموذج الخبير.")

    # حساب "فورمة" الفريق حتى تاريخ اليوم
    home_form = calculate_team_form(all_matches, home_team_id, prediction_date, num_matches=5)
    away_form = calculate_team_form(all_matches, away_team_id, prediction_date, num_matches=5)

    # بناء الميزات
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

    # إجراء التنبؤ
    probs = model.predict_proba(features_df)[0]
    
    # فئات النموذج هي [-1, 0, 1] والتي تترجم إلى [away_win, draw, home_win]
    return {"away_win": probs[0], "draw": probs[1], "home_win": probs[2], "model_inputs": features_dict}


# دالة مساعدة لإظهار زر التحميل
def show_download_button(file_path: Path):
    if file_path.exists():
        with open(file_path, "rb") as fp:
            st.download_button(
                label=f"📥 تحميل {file_path.name}",
                data=fp,
                file_name=file_path.name,
                mime="application/octet-stream"
            )

# ==============================================================================
# واجهة المستخدم الرئيسية لـ Streamlit
# ==============================================================================

st.title("⚽ لوحة تحكم متكاملة لتوقع نتائج المباريات")
st.caption("Dixon–Coles + Team Factors + ELO + XGBoost Model")

# --- الشريط الجانبي (Sidebar) ---
with st.sidebar:
    st.header("إدارة البيانات والنماذج")
    st.info("يجب تشغيل العمليات بالترتيب لضمان عمل التطبيق بشكل صحيح.")

    # --- قسم النماذج الإحصائية الأساسية ---
    with st.expander("1. بناء البيانات والنماذج الإحصائية", expanded=True):
        years = st.number_input("عدد المواسم المطلوب جلبها", min_value=1, max_value=20, value=3, step=1)
        
        if st.button("تشغيل بناء البيانات (01)"):
            with st.spinner("⏳ جارٍ بناء قاعدة البيانات..."):
                ok, logs = run_cli_script([sys.executable, "01_pipeline.py", "--years", str(years)])
                st.cache_data.clear()
            st.success("✅ اكتملت عملية بناء البيانات.") if ok else st.error("❌ فشل بناء البيانات.")
            with st.expander("عرض سجلات التنفيذ"): st.code(logs)
            if ok:
                show_download_button(config.DATA_DIR / "matches.json")
                show_download_button(config.DATA_DIR / "teams.json")

        if st.button("تدريب النماذج الإحصائية (02)"):
            with st.spinner("⏳ جارٍ تدريب النماذج الإحصائية..."):
                ok, logs = run_cli_script([sys.executable, "02_trainer.py"])
                st.cache_data.clear()
            st.success("✅ اكتمل تدريب النماذج الإحصائية.") if ok else st.error("❌ فشل تدريب النماذج.")
            with st.expander("عرض سجلات التنفيذ"): st.code(logs)
            if ok:
                show_download_button(config.MODELS_DIR / "team_factors.json")
                show_download_button(config.MODELS_DIR / "elo_ratings.json")
                show_download_button(config.MODELS_DIR / "league_averages.json")

    # --- قسم نماذج تعلم الآلة ---
    with st.expander("2. بناء نماذج تعلم الآلة (متقدم)"):
        if st.button("إنشاء ميزات التدريب (04)"):
            with st.spinner("⏳ جارٍ إنشاء ملف الميزات..."):
                ok, logs = run_cli_script([sys.executable, "04_feature_generator.py"])
            st.success("✅ تم إنشاء ملف الميزات.") if ok else st.error("❌ فشل إنشاء الميزات.")
            with st.expander("عرض سجلات التنفيذ"): st.code(logs)
            if ok:
                show_download_button(config.DATA_DIR / "ml_dataset.csv")

        if st.button("تدريب نموذج ML (05_المُعلّم)"):
            with st.spinner("⏳ جارٍ تدريب نموذج XGBoost..."):
                ok, logs = run_cli_script([sys.executable, "05_train_ml_model.py"])
                st.cache_data.clear()
            st.success("✅ اكتمل تدريب نموذج تعلم الآلة.") if ok else st.error("❌ فشل التدريب.")
            with st.expander("عرض سجلات التنفيذ"): st.code(logs)
            if ok:
                show_download_button(config.MODELS_DIR / "xgboost_model.json")

    st.divider()
    if st.button("🔄 تحديث الكاش بالكامل"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("تم مسح الكاش بنجاح!")

# --- قسم التنبؤ الرئيسي ---
st.header("🔮 اختر مباراة وتنبأ بالنتيجة")

teams_map = load_teams_map()
if not teams_map:
    st.error("لم يتم العثور على ملف بيانات الفرق `teams.json`. يرجى تشغيل 'بناء البيانات' من الشريط الجانبي أولاً.")
    st.stop()

comp_code = st.selectbox("اختر المسابقة", options=config.TARGET_COMPETITIONS, index=0)
comp_teams = teams_for_comp(teams_map, comp_code)

if not comp_teams:
    st.warning(f"لم يتم العثور على فرق لمسابقة '{comp_code}'. قد تحتاج لتشغيل بناء البيانات.")
    st.stop()

names = [n for n, _ in comp_teams]
name_to_id = {n: tid for n, tid in comp_teams}

c1, c2 = st.columns(2)
with c1:
    team1_name = st.selectbox("الفريق المضيف", options=names, index=0)
with c2:
    team2_name = st.selectbox("الفريق الضيف", options=names, index=1 if len(names) > 1 else 0)

if team1_name == team2_name:
    st.warning("يرجى اختيار فريقين مختلفين.")
    st.stop()

home_team_id = name_to_id.get(team1_name)
away_team_id = name_to_id.get(team2_name)

# --- Tabs للتنبؤ ---
stat_tab, ml_tab = st.tabs(["📊 التنبؤ الإحصائي", "🧠 التنبؤ بالخبير (ML)"])

with stat_tab:
    st.subheader("إعدادات النموذج الإحصائي")
    use_elo = st.checkbox("تفعيل تعديل ELO في λ", value=True, key="stat_elo")
    topk = st.slider("أظهر أعلى K من النتائج المحتملة", min_value=0, max_value=10, value=5, key="stat_topk")

    if st.button("احسب التنبؤ الإحصائي", type="primary"):
        try:
            result = compute_statistical_prediction(team1_name, team2_name, comp_code, use_elo=use_elo, topk=topk)
            st.markdown(f"#### {result['match']}")
            p_home, p_draw, p_away = result["probabilities"]["home_win"], result["probabilities"]["draw"], result["probabilities"]["away_win"]

            col1, col2, col3 = st.columns(3)
            col1.metric("فوز المضيف", f"{p_home*100:.1f}%")
            col2.metric("تعادل", f"{p_draw*100:.1f}%")
            col3.metric("فوز الضيف", f"{p_away*100:.1f}%")

            if topk and "top_scorelines" in result:
                st.write(f"**أعلى {topk} نتائج محتملة:**")
                rows = [{"النتيجة": f"{s['home_goals']} - {s['away_goals']}", "الاحتمال": f"{s['prob']*100:.2f}%"} for s in result["top_scorelines"]]
                st.table(rows)
            with st.expander("عرض مدخلات النموذج الإحصائي"): st.json(result["model_inputs"])
        except Exception as e:
            st.error(f"فشل التنبؤ الإحصائي: {e}")

with ml_tab:
    st.info("يستخدم هذا النموذج مخرجات النماذج الإحصائية كميزات لتدريب نموذج XGBoost أكثر قوة.")
    if st.button("🧠 احسب التنبؤ (نموذج الخبير)", type="primary"):
        if not load_xgboost_model():
             st.error("نموذج الخبير (XGBoost) غير موجود. يرجى تدريبه أولاً من الشريط الجانبي (الخطوة 05).")
        else:
            try:
                with st.spinner("...جارٍ حساب تنبؤ الخبير"):
                    result = compute_ml_prediction(home_team_id, away_team_id, comp_code)
                st.markdown(f"#### {team1_name} (المضيف) ضد {team2_name} (الضيف)")
                p_home, p_draw, p_away = result["home_win"], result["draw"], result["away_win"]

                col1, col2, col3 = st.columns(3)
                col1.metric("فوز المضيف", f"{p_home*100:.1f}%")
                col2.metric("تعادل", f"{p_draw*100:.1f}%")
                col3.metric("فوز الضيف", f"{p_away*100:.1f}%")
                
                with st.expander("عرض الميزات المستخدمة في التنبؤ"): 
                    st.json(result["model_inputs"])

            except Exception as e:
                st.error(f"فشل تنبؤ الخبير: {e}")
