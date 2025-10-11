# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# app.py (النسخة المنظمة والنهائية)
# -----------------------------------------------------------------------------
# الوصف:
#   واجهة مستخدم رسومية باستخدام Streamlit لمشروع التنبؤ بنتائج مباريات كرة القدم.
#   تسمح الواجهة للمستخدم بما يلي:
#   - تشغيل عمليات بناء البيانات وتدريب النماذج مباشرة من المتصفح.
#   - اختيار دوري وفريقين للتنبؤ بنتيجة المباراة بينهما.
#   - عرض احتمالات الفوز/التعادل/الخسارة وأبرز النتائج المتوقعة.
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
# يجب أن يكون هذا الجزء في بداية الملف تمامًا
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# الآن، نستورد الوحدات من مجلد 'common' بأمان
try:
    from common import config
    from common.utils import enhanced_team_search
    from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines
except ImportError as e:
    st.error(
        f"**خطأ فادح في الاستيراد:** لم يتم العثور على مجلد `common` وملفاته.\n"
        f"تأكد من أن بنية المشروع صحيحة وأن ملف `__init__.py` موجود داخل `common`.\n"
        f"التفاصيل: {e}"
    )
    st.stop()


# --- 2. إعداد الصفحة ---

st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")


# --- 3. دوال تحميل البيانات والنماذج (مع التخزين المؤقت) ---

@st.cache_data
def _load_json(path: Path) -> Optional[dict]:
    """دالة مساعدة لتحميل ملف JSON مع معالجة الأخطاء."""
    if path and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def load_teams_map() -> Optional[Dict]:
    """تحميل قاموس الفرق من ملف teams.json."""
    return _load_json(config.DATA_DIR / "teams.json")

@st.cache_data
def load_models() -> Dict[str, dict]:
    """تحميل جميع النماذج الإحصائية المدربة."""
    return {
        "averages": _load_json(config.MODELS_DIR / "league_averages.json") or {},
        "factors": _load_json(config.MODELS_DIR / "team_factors.json") or {},
        "elo": _load_json(config.MODELS_DIR / "elo_ratings.json") or {},
        "rho": _load_json(config.MODELS_DIR / "rho_values.json") or {},
    }


# --- 4. دوال المنطق الأساسي والحسابات ---

def run_cli_command(script_name: str, args: List[str] = []) -> Tuple[bool, str]:
    """تشغيل سكريبت خارجي كعملية منفصلة وإرجاع النتيجة."""
    cmd = [sys.executable, str(APP_ROOT / script_name)] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"فشل في تشغيل العملية: {e}"

def get_current_season_year() -> int:
    """تحديد سنة بداية الموسم الحالي."""
    now = datetime.now()
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def compute_prediction(
    home_name: str, away_name: str, comp_code: str, use_elo: bool, topk: int
) -> dict:
    """الدالة الرئيسية لحساب التنبؤ لمباراة محددة."""
    teams_map = load_teams_map()
    if not teams_map:
        raise RuntimeError("ملف `teams.json` غير موجود. يرجى تشغيل 'بناء البيانات' أولاً.")

    home_id = enhanced_team_search(home_name, teams_map, comp_code)
    away_id = enhanced_team_search(away_name, teams_map, comp_code)

    if not home_id or not away_id:
        raise ValueError(f"لم يتم العثور على أحد الفريقين: '{home_name}', '{away_name}'")

    season_year = get_current_season_year()
    season_key = f"{comp_code}_{season_year}"
    last_season_key = f"{comp_code}_{season_year - 1}"

    models = load_models()
    # استخدام نماذج الموسم الماضي كبديل إذا لم تكن نماذج الموسم الحالي جاهزة
    if season_key not in models["elo"]:
        season_key = last_season_key
        if season_key not in models["elo"]:
            raise RuntimeError(f"لا توجد نماذج حديثة للمسابقة {comp_code}. يرجى 'تدريب النماذج' أولاً.")

    # استخلاص المعاملات من النماذج المحملة
    elo = models["elo"].get(season_key, {})
    factors = models["factors"].get(season_key, {})
    avgs = models["averages"].get(season_key, {})
    rho = models["rho"].get(season_key, 0.0)

    elo_home = float(elo.get(str(home_id), 1500.0))
    elo_away = float(elo.get(str(away_id), 1500.0))

    home_attack = float(factors.get("attack", {}).get(str(home_id), 1.0))
    home_defense = float(factors.get("defense", {}).get(str(home_id), 1.0))
    away_attack = float(factors.get("attack", {}).get(str(away_id), 1.0))
    away_defense = float(factors.get("defense", {}).get(str(away_id), 1.0))

    avg_home = float(avgs.get("avg_home_goals", 1.4))
    avg_away = float(avgs.get("avg_away_goals", 1.1))

    # حساب الأهداف المتوقعة (Lambda)
    lam_home = home_attack * away_defense * avg_home
    lam_away = away_attack * home_defense * avg_away

    # تطبيق تعديل ELO اختياريًا
    if use_elo:
        edge = (elo_home - elo_away) + config.ELO_HFA
        factor = 10 ** (edge / config.ELO_LAMBDA_SCALE)
        lam_home *= factor
        lam_away = max(1e-9, lam_away / factor)

    # حساب مصفوفة الاحتمالات والنتائج
    matrix = poisson_matrix_dc(lam_home, lam_away, rho, max_goals=8)
    p_home, p_draw, p_away = matrix_to_outcomes(matrix)
    top_scores = top_scorelines(matrix, top_k=topk)

    return {
        "meta": {"model_season_used": season_key},
        "match": f"{home_name} (المضيف) ضد {away_name} (الضيف)",
        "competition": comp_code,
        "model_inputs": {
            "lambda_home": round(lam_home, 3),
            "lambda_away": round(lam_away, 3),
            "rho": round(rho, 3),
            "use_elo_adjust": use_elo,
        },
        "probabilities": {"home_win": p_home, "draw": p_draw, "away_win": p_away},
        "top_scorelines": [{"home_goals": i, "away_goals": j, "prob": p} for i, j, p in top_scores],
    }


# --- 5. دوال مساعدة لواجهة المستخدم ---

def get_teams_for_competition(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    """استخراج قائمة الفرق لمسابقة معينة."""
    out = []
    for t in teams_map.values():
        if comp_code in t.get("competitions", []):
            names = [n for n in t.get("names", []) if n]
            primary_name = sorted(names, key=lambda n: (int(" " in n), len(n)), reverse=True)[0]
            out.append((primary_name, t.get("id")))
    return sorted(out, key=lambda x: x[0].lower())


# --- 6. بناء الواجهة الرسومية ---

st.title("⚽ متنبئ نتائج مباريات كرة القدم")
st.caption(f"نموذج Dixon–Coles + عوامل الفرق + تصنيف ELO | الإصدار: {config.VERSION}")

# -- الشريط الجانبي --
with st.sidebar:
    st.header("إدارة البيانات والنماذج")
    st.info("للاستخدام لأول مرة، قم بتشغيل 'بناء البيانات' ثم 'تدريب النماذج'.")

    years = st.number_input("عدد المواسم المطلوب جلبها", min_value=1, max_value=10, value=3)
    if st.button("🏗️ بناء البيانات (Pipeline)"):
        with st.spinner("جارٍ بناء قاعدة البيانات... قد يستغرق هذا وقتًا طويلاً."):
            ok, logs = run_cli_command("01_pipeline.py", ["--years", str(years)])
        st.cache_data.clear()
        if ok: st.success("تم بناء البيانات بنجاح!")
        else: st.error("فشل بناء البيانات.")
        with st.expander("عرض السجلات"): st.code(logs)

    if st.button("🧠 تدريب النماذج (Trainer)"):
        with st.spinner("جارٍ تدريب النماذج..."):
            ok, logs = run_cli_command("02_trainer.py")
        st.cache_data.clear()
        if ok: st.success("تم تدريب النماذج بنجاح!")
        else: st.error("فشل تدريب النماذج.")
        with st.expander("عرض السجلات"): st.code(logs)

    st.divider()
    st.header("خيارات التنبؤ")
    use_elo = st.checkbox("تفعيل تعديل ELO على الأهداف", value=True)
    topk = st.slider("عدد أعلى النتائج المتوقعة", min_value=0, max_value=10, value=5)
    if st.button("🔄 تحديث الكاش"):
        st.cache_data.clear()
        st.rerun()

# -- الصفحة الرئيسية --
st.subheader("اختر مباراة للتنبؤ بها")

teams_map = load_teams_map()
if not teams_map:
    st.warning("لم يتم العثور على بيانات الفرق. يرجى تشغيل 'بناء البيانات' من الشريط الجانبي أولاً.")
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        comp_code = st.selectbox("اختر المسابقة", options=config.TARGET_COMPETITIONS)

    comp_teams = get_teams_for_competition(teams_map, comp_code)

    if not comp_teams:
        st.error(f"لم يتم العثور على فرق للمسابقة '{comp_code}'. تأكد من تضمينها في `TARGET_COMPETITIONS` وتشغيل بناء البيانات.")
    else:
        team_names = [name for name, _ in comp_teams]
        name_to_id = {name: tid for name, tid in comp_teams}

        col1, col2 = st.columns(2)
        with col1:
            home_name = st.selectbox("الفريق المضيف", options=team_names, index=0)
        with col2:
            away_name = st.selectbox("الفريق الضيف", options=team_names, index=1 if len(team_names) > 1 else 0)

        if st.button("🔮 احسب التنبؤ الآن", type="primary", use_container_width=True):
            if home_name == away_name:
                st.error("يرجى اختيار فريقين مختلفين.")
            else:
                try:
                    with st.spinner("جاري حساب الاحتمالات..."):
                        result = compute_prediction(home_name, away_name, comp_code, use_elo, topk)

                    st.markdown(f"#### {result['match']}")
                    st.caption(f"المسابقة: {result['competition']} | الموسم المستخدم للنموذج: {result['meta']['model_season_used']}")

                    p_home = result["probabilities"]["home_win"]
                    p_draw = result["probabilities"]["draw"]
                    p_away = result["probabilities"]["away_win"]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("فوز المضيف", f"{p_home:.1%}")
                    c2.metric("تعادل", f"{p_draw:.1%}")
                    c3.metric("فوز الضيف", f"{p_away:.1%}")

                    if topk > 0 and result["top_scorelines"]:
                        st.subheader(f"أعلى {topk} نتائج محتملة")
                        score_data = {
                            "النتيجة": [f"{s['home_goals']} - {s['away_goals']}" for s in result["top_scorelines"]],
                            "الاحتمال": [f"{s['prob']:.2%}" for s in result["top_scorelines"]],
                        }
                        st.dataframe(score_data, use_container_width=True)

                    with st.expander("عرض مدخلات النموذج التفصيلية"):
                        st.json(result["model_inputs"])

                except Exception as e:
                    st.error(f"حدث خطأ أثناء التنبؤ: {e}")
