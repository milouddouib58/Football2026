# app.py
# -----------------------------------------------------------------------------
# لوحة Streamlit لإدارة خط أنابيب البيانات والنماذج، وتشغيل التنبؤ باستخدام
# كلاس Predictor الموحّد مع كاش محسّن وتشخيصات مبسّطة.
# -----------------------------------------------------------------------------

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ---------------------------------------------
import streamlit as st
from common import config
from common.utils import log
from predictor import Predictor

# -----------------------------------------------------------------------------
# إعدادات الصفحة
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="⚽ Football Predictor",
    page_icon="⚽",
    layout="wide"
)

# -----------------------------------------------------------------------------
# كاش وقراءة الملفات
# -----------------------------------------------------------------------------
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

@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()

def safe_clear_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# أدوات مساعدة
# -----------------------------------------------------------------------------
def run_cli_script(cmd: List[str]) -> Tuple[bool, str]:
    """تشغيل أي سكريبت والتقاط مخرجاته."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"فشل في تشغيل السكريبت: {e}"

def current_season_year(now: datetime) -> int:
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def _primary_name(names: List[str]) -> str:
    names = [n for n in (names or []) if n]
    if not names:
        return "Unknown"

    def score(n: str) -> Tuple[int, int, int]:
        # تفضيل الأسماء التي تحتوي فراغ (أقرب للاسم الكامل)، الأطول، وغير الكبيرة بالكامل
        return (int(" " in n), len(n), -int(n.isupper()))

    return sorted(names, key=score, reverse=True)[0]

def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    out = sorted(
        [(_primary_name(t.get("names", [])), t.get("id")) for t in teams_map.values()
         if comp_code in t.get("competitions", []) and t.get("id")],
        key=lambda x: x[0].lower()
    )
    return out

def compute_prediction(team1_name: str, team2_name: str, comp_code: str, use_elo: bool, topk: int):
    pred = get_predictor()
    result = pred.predict(team1_name, team2_name, comp_code, topk=topk, use_elo=use_elo)
    return result

def model_file_info() -> List[Tuple[str, Path, bool, Optional[float], Optional[int]]]:
    files = [
        ("matches.json", config.DATA_DIR / "matches.json"),
        ("teams.json", config.DATA_DIR / "teams.json"),
        ("league_averages.json", config.MODELS_DIR / "league_averages.json"),
        ("team_factors.json", config.MODELS_DIR / "team_factors.json"),
        ("elo_ratings.json", config.MODELS_DIR / "elo_ratings.json"),
        ("rho_values.json", config.MODELS_DIR / "rho_values.json"),
        ("xgboost_model.json", config.MODELS_DIR / "xgboost_model.json"),
    ]
    out = []
    for name, p in files:
        try:
            exists = p.exists()
            mtime = p.stat().st_mtime if exists else None
            size = p.stat().st_size if exists else None
        except Exception:
            exists, mtime, size = False, None, None
        out.append((name, p, exists, mtime, size))
    return out

# -----------------------------------------------------------------------------
# واجهة المستخدم
# -----------------------------------------------------------------------------
st.title("⚽ لوحة تحكم متكاملة لتوقع نتائج المباريات")
st.caption("Dixon–Coles + Team Factors + ELO + XGBoost | الإصدار: " + str(getattr(config, "VERSION", "N/A")))

# --- الشريط الجانبي (Sidebar) ---
with st.sidebar:
    st.header("إدارة البيانات والنماذج")
    st.info("يُفضّل تشغيل العمليات بالترتيب لضمان عمل التطبيق بشكل صحيح.")

    st.subheader("1. النماذج الإحصائية (أساسي)")
    years = st.number_input("عدد المواسم المطلوب جلبها", min_value=1, max_value=20, value=3, step=1)

    if st.button("تشغيل بناء البيانات (01_pipeline)"):
        with st.spinner("⏳ جارٍ بناء قاعدة البيانات... هذه العملية قد تستغرق عدة دقائق."):
            ok, logs = run_cli_script([sys.executable, "01_pipeline.py", "--years", str(years)])
        safe_clear_cache()
        st.success("✅ اكتملت عملية بناء البيانات.") if ok else st.error("❌ فشل بناء البيانات.")
        with st.expander("عرض سجلات التنفيذ"):
            st.code(logs)

    if st.button("تدريب النماذج الإحصائية (02_trainer)"):
        with st.spinner("⏳ جارٍ تدريب النماذج الإحصائية (Elo, Factors, Rho)..."):
            ok, logs = run_cli_script([sys.executable, "02_trainer.py"])
        safe_clear_cache()
        st.success("✅ اكتمل تدريب النماذج الإحصائية.") if ok else st.error("❌ فشل تدريب النماذج.")
        with st.expander("عرض سجلات التنفيذ"):
            st.code(logs)

    st.divider()

    st.subheader("2. نماذج تعلم الآلة (متقدم)")
    if st.button("إجراء الاختبار التاريخي (03_backtester)"):
        with st.spinner("⏳ جارٍ إجراء الاختبار التاريخي لتقييم النموذج..."):
            ok, logs = run_cli_script([sys.executable, "03_backtester.py"])
        st.success("✅ اكتمل الاختبار التاريخي.") if ok else st.error("❌ فشل الاختبار.")
        with st.expander("عرض سجلات التنفيذ"):
            st.code(logs)

    if st.button("إنشاء ميزات التدريب (04_feature_generator)"):
        with st.spinner("⏳ جارٍ إنشاء ملف الميزات لنموذج تعلم الآلة..."):
            ok, logs = run_cli_script([sys.executable, "04_feature_generator.py"])
        st.success("✅ تم إنشاء ملف الميزات بنجاح.") if ok else st.error("❌ فشل إنشاء الميزات.")
        with st.expander("عرض سجلات التنفيذ"):
            st.code(logs)

    if st.button("تدريب نموذج ML (05_train_ml_model)"):
        st.warning("تأكد من إنشاء ملف الميزات أولاً. هذه العملية قد تستغرق بعض الوقت.")
        with st.spinner("⏳ جارٍ تدريب نموذج XGBoost..."):
            ok, logs = run_cli_script([sys.executable, "05_train_ml_model.py"])
        safe_clear_cache()
        st.success("✅ اكتمل تدريب نموذج تعلم الآلة.") if ok else st.error("❌ فشل التدريب.")
        with st.expander("عرض سجلات التنفيذ"):
            st.code(logs)

    if st.button("تشغيل توقع ML (06_predict_ml)"):
        st.info("سيتم تشغيل التوقع للمباراة المحددة داخل ملف 06_predict_ml.py.")
        with st.spinner("⏳ جارٍ تشغيل خبير تعلم الآلة للتنبؤ..."):
            ok, logs = run_cli_script([sys.executable, "06_predict_ml.py"])
        st.success("✅ تم تشغيل الخبير بنجاح.") if ok else st.error("❌ فشل تشغيل الخبير.")
        with st.expander("عرض سجلات التنفيذ"):
            st.code(logs)

    st.divider()
    st.header("إعدادات التنبؤ")
    use_elo = st.checkbox("تفعيل تعديل ELO في λ", value=True)
    topk = st.slider("أظهر أعلى K من النتائج المحتملة", min_value=0, max_value=10, value=5)

    if st.button("🔄 تحديث الكاش"):
        safe_clear_cache()
        st.success("تم مسح الكاش بنجاح!")

# -----------------------------------------------------------------------------
# تشخيص سريع لحالة الملفات
# -----------------------------------------------------------------------------
with st.expander("🩺 تشخيص الحالة (الملفات والجاهزية)", expanded=False):
    info = model_file_info()
    cols = st.columns(3)
    for i, (name, p, exists, mtime, size) in enumerate(info):
        with cols[i % 3]:
            if exists:
                ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "-"
                st.success(f"{name}\n{p}")
                st.caption(f"آخر تحديث: {ts} | حجم: {size or 0} بايت")
            else:
                st.error(f"{name} غير متوفر")
                st.caption(str(p))

# -----------------------------------------------------------------------------
# قسم التنبؤ الرئيسي
# -----------------------------------------------------------------------------
st.header("تنبؤ المباراة (النموذج الإحصائي)")
teams_map = load_teams_map()
if not teams_map:
    st.error("لم يتم العثور على ملف بيانات الفرق teams.json. يرجى تشغيل 'بناء البيانات' من الشريط الجانبي أولاً.")
    st.stop()

models = load_models()
missing_models = [k for k, v in models.items() if not v]
if missing_models:
    st.warning(f"بعض ملفات النماذج غير متوفرة: {', '.join(missing_models)}. قد تحتاج لتشغيل تدريب النماذج الإحصائية.")

# اختيار المسابقة والفرق
comp_code = st.selectbox("اختر المسابقة", options=getattr(config, "TARGET_COMPETITIONS", []), index=0 if getattr(config, "TARGET_COMPETITIONS", []) else 0)

comp_teams = teams_for_comp(teams_map, comp_code) if comp_code else []
if not comp_teams:
    st.warning(f"لم يتم العثور على فرق لمسابقة '{comp_code}'. قد تحتاج لتشغيل بناء البيانات.")
    st.stop()

names = [n for n, _ in comp_teams]

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
        result = compute_prediction(team1_name, team2_name, comp_code, use_elo=use_elo, topk=topk)

        st.markdown(f"### {result.get('match', f'{team1_name} vs {team2_name}')}")
        meta = result.get("meta", {})
        st.caption(f"المسابقة: {result.get('competition', comp_code)} | الموسم المستخدم للنموذج: {meta.get('model_season_used', 'N/A')}")

        p_home = float(result["probabilities"]["home_win"])
        p_draw = float(result["probabilities"]["draw"])
        p_away = float(result["probabilities"]["away_win"])

        col1, col2, col3 = st.columns(3)
        col1.metric("فوز المضيف", f"{p_home*100:.1f}%")
        col2.metric("تعادل", f"{p_draw*100:.1f}%")
        col3.metric("فوز الضيف", f"{p_away*100:.1f}%")

        # أعلى K من النتائج المحتملة
        if topk and "top_scorelines" in result:
            st.subheader(f"أعلى {topk} نتائج محتملة")
            rows = [
                {
                    "النتيجة": f"{s['home_goals']} - {s['away_goals']}",
                    "الاحتمال": f"{s['prob']*100:.2f}%"
                }
                for s in result["top_scorelines"]
            ]
            st.table(rows)

        # مدخلات النموذج وتفاصيل
        with st.expander("عرض مدخلات النموذج"):
            st.json(result.get("model_inputs", {}))

        with st.expander("تفاصيل الفرق (IDs)"):
            st.json(result.get("teams_found", {}))

        with st.expander("النتيجة الكاملة (JSON)"):
            st.code(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        st.error(f"فشل التنبؤ: {e}")
