# app.py (النسخة النهائية مع جميع الأزرار وزر التحميل)

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
# هذا يجب أن يكون في الأعلى قبل استيراد أي وحدة من 'common'
try:
    # الطريقة القياسية التي تعمل في معظم البيئات
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    # حل بديل لبيئات مثل Streamlit Cloud حيث __file__ غير معرف
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# ---------------------------------------------

import streamlit as st

# الآن ستعمل الاستيرادات بشكل صحيح
from common import config
from common.utils import enhanced_team_search
from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines

# إعدادات الصفحة
st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")

# --- دوال مساعدة لتحميل البيانات مع كاش ---
@st.cache_data
def _load_json(path: Path) -> Optional[dict]:
    """يقوم بتحميل ملف JSON مع معالجة الأخطاء."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

@st.cache_data
def load_teams_map() -> Optional[Dict]:
    """تحميل خريطة الفرق من ملف teams.json."""
    return _load_json(config.DATA_DIR / "teams.json")

@st.cache_data
def load_models() -> Dict[str, dict]:
    """تحميل جميع النماذج الإحصائية."""
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
        # cwd=project_root يضمن تشغيل السكريبت من المجلد الصحيح
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, encoding='utf-8', cwd=project_root
        )
        ok = (result.returncode == 0)
        # دمج المخرجات القياسية ومخرجات الخطأ لعرض كل شيء
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"فشل في تشغيل السكريبت: {e}"

# --- دوال الواجهة الرسومية ---
def current_season_year(now: datetime) -> int:
    """يحسب سنة بداية الموسم الحالي."""
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def _primary_name(names: List[str]) -> str:
    """يختار أفضل اسم قابل للعرض من قائمة أسماء الفريق."""
    names = [n for n in (names or []) if n]
    if not names: return "Unknown"
    def score(n: str) -> Tuple[int, int, int]:
        return (int(" " in n), len(n), -int(n.isupper()))
    return sorted(names, key=score, reverse=True)[0]

def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    """يسترجع قائمة الفرق لمسابقة معينة."""
    out = sorted(
        [(_primary_name(t.get("names", [])), t.get("id"))
         for t in teams_map.values() if comp_code in t.get("competitions", []) and t.get("id")],
        key=lambda x: x[0].lower()
    )
    return out

def compute_prediction(team1_name: str, team2_name: str, comp_code: str, use_elo: bool, topk: int) -> dict:
    """الدالة الأساسية لحساب التوقع الإحصائي."""
    teams_map = load_teams_map()
    if not teams_map:
        raise RuntimeError("ملف `teams.json` غير موجود. يرجى تشغيل (01_pipeline) أولاً.")

    home_id = enhanced_team_search(team1_name, teams_map, comp_code)
    away_id = enhanced_team_search(team2_name, teams_map, comp_code)
    if not home_id or not away_id:
        raise ValueError(f"لم يتم العثور على أحد الفريقين: '{team1_name}', '{team2_name}'")

    season_year = current_season_year(datetime.now())
    season_key = f"{comp_code}_{season_year}"
    
    models = load_models()
    if season_key not in models["elo"]:
        season_key = f"{comp_code}_{season_year - 1}"
        if season_key not in models["elo"]:
            raise RuntimeError(f"لا توجد نماذج حديثة لمسابقة {comp_code}. يرجى تدريب النماذج أولاً.")

    elo = models["elo"].get(season_key, {})
    elo_home = float(elo.get(str(home_id), 1500))
    elo_away = float(elo.get(str(away_id), 1500))

    factors = models["factors"].get(season_key, {})
    home_attack = float(factors.get("attack", {}).get(str(home_id), 1.0))
    home_defense = float(factors.get("defense", {}).get(str(home_id), 1.0))
    away_attack = float(factors.get("attack", {}).get(str(away_id), 1.0))
    away_defense = float(factors.get("defense", {}).get(str(away_id), 1.0))

    avgs = models["averages"].get(season_key, {})
    avg_home = float(avgs.get("avg_home_goals", 1.4))
    avg_away = float(avgs.get("avg_away_goals", 1.1))
    rho = float(models["rho"].get(season_key, 0.0))

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
        "meta": {"version": config.VERSION, "model_season_used": season_key},
        "match": f"{team1_name} (مضيف) ضد {team2_name} (ضيف)",
        "competition": comp_code,
        "model_inputs": {
            "lambda_home": round(lam_home, 3), "lambda_away": round(lam_away, 3),
            "rho": round(rho, 3), "use_elo_adjust": use_elo
        },
        "probabilities": {"home_win": p_home, "draw": p_draw, "away_win": p_away}
    }

    if topk > 0:
        tops = top_scorelines(matrix, top_k=topk)
        result["top_scorelines"] = [{"home_goals": i, "away_goals": j, "prob": p} for i, j, p in tops]
    return result

# ==============================================================================
# واجهة المستخدم الرئيسية لـ Streamlit
# ==============================================================================

st.title("⚽ لوحة تحكم متكاملة لتوقع نتائج المباريات")
st.caption("النموذج الإحصائي (Dixon-Coles + ELO) ونموذج تعلم الآلة (XGBoost)")

# --- الشريط الجانبي (Sidebar) ---
with st.sidebar:
    st.header("⚙️ إدارة المشروع")
    st.info("يجب تشغيل العمليات بالترتيب لضمان عمل التطبيق بشكل صحيح.")

    with st.expander("المرحلة 1: البيانات والنماذج الإحصائية", expanded=True):
        years = st.number_input("عدد المواسم لجلب البيانات", 1, 20, 3)
        if st.button("1. تحديث البيانات (Pipeline)"):
            with st.spinner("⏳ جارٍ جلب البيانات... قد تستغرق هذه العملية عدة دقائق."):
                ok, logs = run_cli_script([sys.executable, "01_pipeline.py", "--years", str(years)])
            st.cache_data.clear()
            st.success("✅ اكتملت عملية بناء البيانات.") if ok else st.error("❌ فشل بناء البيانات.")
            st.code(logs, language='bash')

        if st.button("2. تدريب النماذج الإحصائية (Trainer)"):
            with st.spinner("⏳ جارٍ تدريب النماذج الإحصائية (Elo, Factors)..."):
                ok, logs = run_cli_script([sys.executable, "02_trainer.py"])
            st.cache_data.clear()
            st.success("✅ اكتمل تدريب النماذج الإحصائية.") if ok else st.error("❌ فشل تدريب النماذج.")
            st.code(logs, language='bash')

    with st.expander("المرحلة 2: نماذج تعلم الآلة (ML)"):
        if st.button("3. إنشاء ميزات التدريب (Features)"):
            with st.spinner("⏳ جارٍ إنشاء ملف الميزات لنموذج تعلم الآلة..."):
                ok, logs = run_cli_script([sys.executable, "04_feature_generator.py"])
            st.success("✅ تم إنشاء ملف الميزات بنجاح.") if ok else st.error("❌ فشل إنشاء الميزات.")
            st.code(logs, language='bash')

        if st.button("4. تدريب نموذج ML (المُعلّم)"):
            st.warning("تأكد من إنشاء ملف الميزات (الخطوة 3) أولاً.")
            with st.spinner("⏳ جارٍ تدريب نموذج XGBoost... هذه العملية قد تستغرق بعض الوقت."):
                ok, logs = run_cli_script([sys.executable, "05_train_ml_model.py"])
            st.cache_data.clear()
            model_path = config.MODELS_DIR / "xgboost_model.json"
            if ok and model_path.exists():
                st.success("✅ اكتمل تدريب نموذج تعلم الآلة وتم إنشاء الملف بنجاح.")
            elif ok:
                st.error("❌ السكريبت اكتمل، لكنه لم يقم بإنشاء ملف النموذج! تحقق من السجلات.")
            else:
                st.error("❌ فشل التدريب.")
            st.code(logs, language='bash')
    
    with st.expander("المرحلة 3: أدوات إضافية"):
        model_path_check = config.MODELS_DIR / "xgboost_model.json"
        if model_path_check.exists():
            with open(model_path_check, "rb") as fp:
                st.download_button(
                    label="📥 تحميل نموذج ML (xgboost_model.json)",
                    data=fp,
                    file_name="xgboost_model.json",
                    mime="application/json"
                )
        else:
            st.info("ملف نموذج تعلم الآلة غير موجود. قم بتدريب النموذج (الخطوة 4) أولاً.")
        
        if st.button("تشغيل توقع ML (الخبير)"):
            st.info("سيتم تشغيل التوقع للمباراة المحددة داخل ملف `06_predict_ml.py`.")
            with st.spinner("⏳ جارٍ تشغيل الخبير للتنبؤ..."):
                ok, logs = run_cli_script([sys.executable, "06_predict_ml.py"])
            st.success("✅ تم تشغيل الخبير بنجاح.") if ok else st.error("❌ فشل تشغيل الخبير.")
            st.code(logs, language='bash')
        
        if st.button("إجراء الاختبار التاريخي (Backtester)"):
            with st.spinner("⏳ جارٍ إجراء الاختبار التاريخي لتقييم النموذج الإحصائي..."):
                ok, logs = run_cli_script([sys.executable, "03_backtester.py"])
            st.success("✅ اكتمل الاختبار التاريخي.") if ok else st.error("❌ فشل الاختبار.")
            st.code(logs, language='bash')

    st.divider()
    st.header("⚙️ إعدادات التنبؤ")
    use_elo = st.checkbox("تفعيل تعديل ELO", value=True)
    topk = st.slider("عدد النتائج المحتملة", 0, 10, 5)
    if st.button("🔄 تحديث الكاش"):
        st.cache_data.clear()
        st.rerun()

# --- قسم التنبؤ الرئيسي ---
st.header("🔮 تنبؤ المباراة (النموذج الإحصائي)")

teams_map = load_teams_map()
if not teams_map:
    st.error("لم يتم العثور على ملف بيانات الفرق `teams.json`. يرجى تشغيل 'تحديث البيانات' من الشريط الجانبي أولاً.")
    st.stop()

comp_code = st.selectbox("اختر المسابقة", options=config.TARGET_COMPETITIONS, index=0)
comp_teams = teams_for_comp(teams_map, comp_code)

if not comp_teams:
    st.warning(f"لم يتم العثور على فرق لمسابقة '{comp_code}'. قد تحتاج لتشغيل تحديث البيانات.")
else:
    names = [n for n, _ in comp_teams]
    name_to_id = {n: tid for n, tid in comp_teams}

    c1, c2 = st.columns(2)
    with c1:
        team1_name = st.selectbox("الفريق المضيف", options=names, index=0)
    with c2:
        team2_name = st.selectbox("الفريق الضيف", options=names, index=1 if len(names) > 1 else 0)

    if team1_name == team2_name:
        st.warning("يرجى اختيار فريقين مختلفين.")
    elif st.button("📊 احسب التنبؤ الآن", type="primary", use_container_width=True):
        try:
            result = compute_prediction(team1_name, team2_name, comp_code, use_elo=use_elo, topk=topk)
            st.markdown(f"### {result['match']}")
            st.caption(f"المسابقة: {result['competition']} | الموسم المستخدم: {result['meta']['model_season_used']}")
            
            p_home = result["probabilities"]["home_win"]
            p_draw = result["probabilities"]["draw"]
            p_away = result["probabilities"]["away_win"]

            col1, col2, col3 = st.columns(3)
            col1.metric("فوز المضيف", f"{p_home*100:.1f}%")
            col2.metric("تعادل", f"{p_draw*100:.1f}%")
            col3.metric("فوز الضيف", f"{p_away*100:.1f}%")

            if topk > 0 and "top_scorelines" in result:
                st.subheader(f"أعلى {topk} نتائج محتملة")
                rows = [{"النتيجة": f"{s['home_goals']} - {s['away_goals']}", "الاحتمال": f"{s['prob']*100:.2f}%"} for s in result["top_scorelines"]]
                st.table(rows)

            with st.expander("عرض مدخلات النموذج"):
                st.json(result["model_inputs"])

        except Exception as e:
            st.error(f"فشل التنبؤ: {e}")
