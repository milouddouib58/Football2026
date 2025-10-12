# app.py (النسخة النهائية مع جميع الأزرار)

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
# هذا يجب أن يكون في الأعلى قبل استيراد أي وحدة من 'common'
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# ---------------------------------------------

import streamlit as st

# الآن ستعمل الاستيرادات بشكل صحيح
from common import config
from common.utils import enhanced_team_search, log
from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines

# إعدادات الصفحة
st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")

# --- دوال مساعدة لتحميل البيانات مع كاش ---
@st.cache_data
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_teams_map() -> Optional[Dict]:
    return _load_json(config.DATA_DIR / "teams.json")

@st.cache_data
def load_models() -> Dict[str, dict]:
    return {
        "averages": _load_json(config.MODELS_DIR / "league_averages.json") or {},
        "factors": _load_json(config.MODELS_DIR / "team_factors.json") or {},
        "elo": _load_json(config.MODELS_DIR / "elo_ratings.json") or {},
        "rho": _load_json(config.MODELS_DIR / "rho_values.json") or {},
    }
# -----------------------------------------

# --- دوال لتشغيل السكريبتات من سطر الأوامر ---
def run_cli_script(cmd: List[str]) -> Tuple[bool, str]:
    """دالة عامة لتشغيل أي سكريبت والتقاط مخرجاته."""
    try:
        # استخدام check=False لالتقاط الأخطاء يدويًا وعرضها بشكل أفضل
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
        ok = (result.returncode == 0)
        # دمج المخرجات القياسية ومخرجات الخطأ لعرض كل شيء
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"فشل في تشغيل السكريبت: {e}"

# --- دوال الواجهة الرسومية ---
def current_season_year(now: datetime) -> int:
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def _primary_name(names: List[str]) -> str:
    names = [n for n in (names or []) if n]
    if not names: return "Unknown"
    def score(n: str) -> Tuple[int, int, int]:
        return (int(" " in n), len(n), -int(n.isupper()))
    return sorted(names, key=score, reverse=True)[0]

def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    out = sorted(
        [(_primary_name(t.get("names", [])), t.get("id"))
         for t in teams_map.values() if comp_code in t.get("competitions", []) and t.get("id")],
        key=lambda x: x[0].lower()
    )
    return out
# -----------------------------------------

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
    st.subheader("1. النماذج الإحصائية (أساسي)")
    years = st.number_input("عدد المواسم المطلوب جلبها", min_value=1, max_value=20, value=3, step=1)
    if st.button("تشغيل بناء البيانات (01_pipeline)"):
        with st.spinner("⏳ جارٍ بناء قاعدة البيانات... هذه العملية قد تستغرق عدة دقائق."):
            ok, logs = run_cli_script([sys.executable, "01_pipeline.py", "--years", str(years)])
        st.cache_data.clear()
        st.success("✅ اكتملت عملية بناء البيانات.") if ok else st.error("❌ فشل بناء البيانات.")
        with st.expander("عرض سجلات التنفيذ"): st.code(logs)

    if st.button("تدريب النماذج الإحصائية (02_trainer)"):
        with st.spinner("⏳ جارٍ تدريب النماذج الإحصائية (Elo, Factors)..."):
            ok, logs = run_cli_script([sys.executable, "02_trainer.py"])
        st.cache_data.clear()
        st.success("✅ اكتمل تدريب النماذج الإحصائية.") if ok else st.error("❌ فشل تدريب النماذج.")
        with st.expander("عرض سجلات التنفيذ"): st.code(logs)
    
    st.divider()

    # --- بداية الإضافة: قسم نماذج تعلم الآلة ---
    st.subheader("2. نماذج تعلم الآلة (متقدم)")
    
    if st.button("إجراء الاختبار التاريخي (03_backtester)"):
        with st.spinner("⏳ جارٍ إجراء الاختبار التاريخي لتقييم النموذج..."):
            ok, logs = run_cli_script([sys.executable, "03_backtester.py"])
        st.success("✅ اكتمل الاختبار التاريخي.") if ok else st.error("❌ فشل الاختبار.")
        with st.expander("عرض سجلات التنفيذ"): st.code(logs)

    if st.button("إنشاء ميزات التدريب (04_features)"):
        with st.spinner("⏳ جارٍ إنشاء ملف الميزات لنموذج تعلم الآلة..."):
            ok, logs = run_cli_script([sys.executable, "04_feature_generator.py"])
        st.success("✅ تم إنشاء ملف الميزات بنجاح.") if ok else st.error("❌ فشل إنشاء الميزات.")
        with st.expander("عرض سجلات التنفيذ"): st.code(logs)

    if st.button("تدريب نموذج ML (05_المُعلّم)"):
        st.warning("تأكد من إنشاء ملف الميزات أولاً. هذه العملية قد تستغرق بعض الوقت.")
        with st.spinner("⏳ جارٍ تدريب نموذج XGBoost..."):
            ok, logs = run_cli_script([sys.executable, "05_train_ml_model.py"])
        st.cache_data.clear() # مسح الكاش بعد تدريب نموذج جديد
        st.success("✅ اكتمل تدريب نموذج تعلم الآلة.") if ok else st.error("❌ فشل التدريب.")
        with st.expander("عرض سجلات التنفيذ"): st.code(logs)

    if st.button("تشغيل توقع ML (06_الخبير)"):
        st.info("سيتم تشغيل التوقع للمباراة المحددة داخل ملف `06_predict_ml.py`.")
        with st.spinner("⏳ جارٍ تشغيل الخبير للتنبؤ..."):
            ok, logs = run_cli_script([sys.executable, "06_predict_ml.py"])
        st.success("✅ تم تشغيل الخبير بنجاح.") if ok else st.error("❌ فشل تشغيل الخبير.")
        with st.expander("عرض سجلات التنفيذ"): st.code(logs)
    # --- نهاية الإضافة ---

    st.divider()
    st.header("إعدادات التنبؤ")
    use_elo = st.checkbox("تفعيل تعديل ELO في λ", value=True)
    topk = st.slider("أظهر أعلى K من النتائج المحتملة", min_value=0, max_value=10, value=5)
    if st.button("🔄 تحديث الكاش"):
        st.cache_data.clear()
        st.success("تم مسح الكاش بنجاح!")
# ---------------------------------------------

# --- قسم التنبؤ الرئيسي ---
st.header("تنبؤ المباراة (النموذج الإحصائي)")

# تحميل البيانات
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

if st.button("🔮 احسب التنبؤ الآن", type="primary"):
    try:
        result, _ = compute_prediction(team1_name, team2_name, comp_code, use_elo=use_elo, topk=topk)
        
        st.markdown(f"### {result['match']}")
        st.caption(f"المسابقة: {result['competition']} | الموسم المستخدم للنموذج: {result['meta']['model_season_used']}")
        
        p_home = result["probabilities"]["home_win"]
        p_draw = result["probabilities"]["draw"]
        p_away = result["probabilities"]["away_win"]

        col1, col2, col3 = st.columns(3)
        col1.metric("فوز المضيف", f"{p_home*100:.1f}%")
        col2.metric("تعادل", f"{p_draw*100:.1f}%")
        col3.metric("فوز الضيف", f"{p_away*100:.1f}%")

        if topk and "top_scorelines" in result:
            st.subheader(f"أعلى {topk} نتائج محتملة")
            rows = [{"النتيجة": f"{s['home_goals']} - {s['away_goals']}", "الاحتمال": f"{s['prob']*100:.2f}%"} for s in result["top_scorelines"]]
            st.table(rows)

        with st.expander("عرض مدخلات النموذج"):
            st.json(result["model_inputs"])

    except Exception as e:
        st.error(f"فشل التنبؤ: {e}")
