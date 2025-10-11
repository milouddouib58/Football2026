# app.py
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st

from common import config
from common.utils import enhanced_team_search, log
from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines

# إعدادات الصفحة
st.set_page_config(page_title="⚽ Football Predictor (Dixon–Coles + ELO)", page_icon="⚽", layout="wide")

# كاش بسيط للتحميل
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

def current_season_year(now: datetime) -> int:
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def _primary_name(names: List[str]) -> str:
    # اختيار اسم "جيّد" للعرض من قائمة الأسماء المتاحة للفريق
    names = [n for n in (names or []) if n]
    if not names:
        return "Unknown"
    def score(n: str) -> Tuple[int, int, int]:
        # أعلى وزن للاسم الأطول والذي يحتوي على مسافة (الاسم الكامل عادة)
        return (int(" " in n), len(n), -int(n.isupper()))
    return sorted(names, key=score, reverse=True)[0]

def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    out = []
    for t in teams_map.values():
        comps = t.get("competitions", [])
        if comp_code in comps:
            out.append((_primary_name(t.get("names", [])), t.get("id")))
    out = [(name, tid) for name, tid in out if tid]
    out.sort(key=lambda x: x[0].lower())
    return out

def run_pipeline_cli(years: int) -> Tuple[bool, str]:
    # لأن 01_pipeline.py لا يمكن استيراده، نشغله كعملية
    cmd = [sys.executable, "01_pipeline.py", "--years", str(years)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"Failed to run pipeline: {e}"

def run_trainer_cli() -> Tuple[bool, str]:
    cmd = [sys.executable, "02_trainer.py"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"Failed to run trainer: {e}"

def compute_prediction(
    team1_name: str,
    team2_name: str,
    comp_code: str,
    use_elo: bool,
    topk: int
) -> Tuple[dict, List[List[float]]]:

    teams_map = load_teams_map()
    if not teams_map:
        raise RuntimeError("Teams map not found. Please run the data pipeline first (01_pipeline.py).")

    # البحث عن الفرق
    home_id = enhanced_team_search(team1_name, teams_map, comp_code)
    away_id = enhanced_team_search(team2_name, teams_map, comp_code)
    if not home_id or not away_id:
        raise ValueError(f"Could not find one or both teams: '{team1_name}', '{team2_name}'")

    # تحديد الموسم
    season_year = current_season_year(datetime.now())
    season_key = f"{comp_code}_{season_year}"
    last_season_key = f"{comp_code}_{season_year - 1}"

    models = load_models()
    if season_key not in models["elo"]:
        season_key = last_season_key
        if season_key not in models["elo"]:
            raise RuntimeError(f"No recent model available for competition {comp_code}. Train models first.")

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

    # حساب لامبدا
    lam_home = home_attack * away_defense * avg_home
    lam_away = away_attack * home_defense * avg_away

    # تعديل ELO اختياريًا
    if use_elo:
        edge = (elo_home - elo_away) + config.ELO_HFA
        factor = 10 ** (edge / config.ELO_LAMBDA_SCALE)
        lam_home = lam_home * factor
        lam_away = max(1e-6, lam_away / factor)

    matrix = poisson_matrix_dc(lam_home, lam_away, rho, max_goals=8)
    p_home, p_draw, p_away = matrix_to_outcomes(matrix)

    result = {
        "meta": {"version": config.VERSION, "model_season_used": season_key},
        "match": f"{team1_name} (Home) vs {team2_name} (Away)",
        "competition": comp_code,
        "teams_found": {
            "home": {"name": team1_name, "id": home_id},
            "away": {"name": team2_name, "id": away_id}
        },
        "model_inputs": {
            "lambda_home": round(lam_home, 3),
            "lambda_away": round(lam_away, 3),
            "rho": round(rho, 3),
            "use_elo_adjust": use_elo
        },
        "probabilities": {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away
        }
    }

    if topk and topk > 0:
        tops = top_scorelines(matrix, top_k=topk)
        result["top_scorelines"] = [
            {"home_goals": i, "away_goals": j, "prob": p} for i, j, p in tops
        ]

    return result, matrix

# واجهة المستخدم
st.title("⚽ Football Predictor — Streamlit UI")
st.caption("Dixon–Coles + Team Factors + ELO (Arabic-enabled team search)")

# الشريط الجانبي: إدارة البيانات والنماذج
with st.sidebar:
    st.header("إدارة البيانات والنماذج")
    st.markdown("- تأكد من وضع مفتاح API في ملف .env")
    st.markdown("- أول مرة: شغّل بناء البيانات ثم درّب النماذج")

    years = st.number_input("عدد السنوات المطلوب جلبها لكل مسابقة", min_value=1, max_value=20, value=5, step=1)
    if st.button("تشغيل بناء البيانات (Pipeline)"):
        with st.spinner("جارٍ بناء قاعدة البيانات... قد يستغرق وقتًا"):
            ok, logs = run_pipeline_cli(years)
        st.cache_data.clear()
        if ok:
            st.success("تم بناء البيانات بنجاح")
        else:
            st.error("فشل بناء البيانات")
        with st.expander("عرض السجلات"):
            st.code(logs)

    if st.button("تدريب النماذج (Trainer)"):
        with st.spinner("جارٍ تدريب النماذج..."):
            ok, logs = run_trainer_cli()
        st.cache_data.clear()
        if ok:
            st.success("تم تدريب النماذج بنجاح")
        else:
            st.error("فشل تدريب النماذج")
        with st.expander("عرض السجلات"):
            st.code(logs)

    st.divider()
    use_elo = st.checkbox("تفعيل تعديل ELO في λ", value=False)
    topk = st.slider("أظهر أعلى K من النتائج المحتملة (scorelines)", min_value=0, max_value=10, value=5)
    if st.button("تحديث الكاش"):
        st.cache_data.clear()
        st.success("تم مسح الكاش")

# القسم الرئيسي: التنبؤ
st.subheader("التنبؤ بالمباراة")

# اختيار الدوري
comp_code = st.selectbox("اختر المسابقة", options=config.TARGET_COMPETITIONS, index=0)

# وضع الإدخال: قائمة أو كتابة أسماء
mode = st.radio("طريقة إدخال الفرق", options=["اختيار من القائمة", "كتابة الأسماء"], horizontal=True)

team1_name = ""
team2_name = ""
selected_home_id = None
selected_away_id = None

teams_map = load_teams_map()
if mode == "اختيار من القائمة":
    if not teams_map:
        st.warning("لا توجد بيانات فرق محلية. شغّل بناء البيانات أولًا.")
    else:
        comp_teams = teams_for_comp(teams_map, comp_code)
        if not comp_teams:
            st.warning("لم يتم العثور على فرق لهذه المسابقة. تأكد من تشغيل بناء البيانات للمسابقات المستهدفة.")
        else:
            names = [n for n, _ in comp_teams]
            name_to_id = {n: tid for n, tid in comp_teams}
            c1, c2 = st.columns(2)
            with c1:
                team1_name = st.selectbox("الفريق المضيف", options=names, index=0 if names else None)
                selected_home_id = name_to_id.get(team1_name)
            with c2:
                team2_name = st.selectbox("الفريق الضيف", options=names, index=1 if len(names) > 1 else 0)
                selected_away_id = name_to_id.get(team2_name)
            if selected_home_id == selected_away_id:
                st.info("اختر فريقين مختلفين من فضلك.")
else:
    c1, c2 = st.columns(2)
    with c1:
        team1_name = st.text_input("اسم الفريق المضيف", value="Manchester City", help="يمكن إدخال اسم عربي مثل 'ريال مدريد'")
    with c2:
        team2_name = st.text_input("اسم الفريق الضيف", value="Arsenal", help="يمكن إدخال اسم عربي مثل 'برشلونة'")

# زر التنبؤ
if st.button("احسب التنبؤ الآن"):
    try:
        if not team1_name or not team2_name:
            st.warning("يرجى تحديد/كتابة اسمي الفريقين.")
            st.stop()
        result, matrix = compute_prediction(team1_name, team2_name, comp_code, use_elo=use_elo, topk=topk)

        # عرض النتائج
        st.markdown(f"### {result['match']} — {result['competition']}  |  الموسم المستخدم: {result['meta']['model_season_used']}")
        p_home = result["probabilities"]["home_win"]
        p_draw = result["probabilities"]["draw"]
        p_away = result["probabilities"]["away_win"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("فوز المضيف", f"{p_home*100:.1f}%")
        with col2:
            st.metric("تعادل", f"{p_draw*100:.1f}%")
        with col3:
            st.metric("فوز الضيف", f"{p_away*100:.1f}%")

        with st.expander("مدخلات النموذج"):
            st.json(result["model_inputs"])

        if topk and "top_scorelines" in result:
            st.subheader(f"أعلى {topk} نتائج محتملة")
            rows = []
            for s in result["top_scorelines"]:
                rows.append({
                    "النتيجة": f"{s['home_goals']} - {s['away_goals']}",
                    "الاحتمال": f"{s['prob']*100:.2f}%"
                })
            st.table(rows)

    except Exception as e:
        st.error(f"فشل التنبؤ: {e}")