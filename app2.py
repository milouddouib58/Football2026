# app.py
# -----------------------------------------------------------------------------
# لوحة Streamlit لإدارة خط أنابيب البيانات والنماذج، مع أزرار تحميل الملفات
# بعد كل عملية، بالإضافة إلى واجهة تنبؤ إحصائي وتنبؤ بنموذج تعلّم الآلة.
# + إصلاح مقارنة التواريخ (naive vs aware) عبر توحيدها إلى Naive-UTC.
# -----------------------------------------------------------------------------

import sys
import os
import io
import json
import zipfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ---------------------------------------------
import streamlit as st
import pandas as pd

# قد لا يتوفر XGBoost في كل بيئة تشغيل، نتعامل بأمان
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from common import config
from common.utils import log
from predictor import Predictor

# -----------------------------------------------------------------------------
# توحيد التواريخ إلى Naive-UTC لمنع أخطاء المقارنة
# -----------------------------------------------------------------------------
def to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            return dt  # Naive بالفعل
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return dt

# محاولة استيراد calculate_team_form وإن لم تتوفر نستخدم بديل بسيط
try:
    from common.modeling import calculate_team_form as _calculate_team_form_base
    def calculate_team_form(all_matches, team_id: int, ref_date: datetime, num_matches: int = 5):
        # نطبّع ref_date قبل تمريره لاحتمال المقارنات الداخلية
        ref_date = to_naive_utc(ref_date)
        return _calculate_team_form_base(all_matches, team_id, ref_date, num_matches=num_matches)
except Exception:
    from common.utils import parse_date_safe, parse_score
    def calculate_team_form(all_matches, team_id: int, ref_date: datetime, num_matches: int = 5):
        # Fallback: توحيد التواريخ إلى Naive-UTC قبل المقارنة
        ref_date = to_naive_utc(ref_date)
        rows = []
        for m in all_matches:
            dt = parse_date_safe(m.get("utcDate"))
            dt = to_naive_utc(dt)
            if not dt or dt >= ref_date:
                continue
            h = m.get("homeTeam", {}).get("id")
            a = m.get("awayTeam", {}).get("id")
            if not h or not a:
                continue
            if int(h) != team_id and int(a) != team_id:
                continue
            hg, ag = parse_score(m)
            if hg is None:
                continue
            if int(h) == team_id:
                pts = 3 if hg > ag else (1 if hg == ag else 0)
            else:
                pts = 3 if ag > hg else (1 if hg == ag else 0)
            rows.append((dt, pts))
        rows.sort(key=lambda x: x[0], reverse=True)
        last = rows[:num_matches]
        if not last:
            return {"avg_points": 1.0}
        avg_pts = sum(p for _, p in last) / len(last)
        return {"avg_points": avg_pts}

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

@st.cache_data
def load_matches_data() -> Optional[List[Dict]]:
    return _load_json(config.DATA_DIR / "matches.json")

@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()

@st.cache_resource
def load_xgb_model():
    if xgb is None:
        return None
    model_path = config.MODELS_DIR / "xgboost_model.json"
    if not model_path.exists():
        return None
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model
    except Exception:
        return None

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

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".json":
        return "application/json"
    if ext == ".csv":
        return "text/csv"
    if ext == ".zip":
        return "application/zip"
    if ext == ".txt":
        return "text/plain"
    return "application/octet-stream"

def offer_file_download(path: Path, label: Optional[str] = None, key: Optional[str] = None):
    if not path.exists():
        st.button(f"❌ {path.name} غير متوفر", disabled=True, key=f"{key or str(path)}_na")
        return
    try:
        with open(path, "rb") as f:
            data = f.read()
        st.download_button(
            label=label or f"⬇️ تحميل {path.name}",
            data=data,
            file_name=path.name,
            mime=guess_mime(path),
            key=key or f"dl_{path.name}"
        )
    except Exception as e:
        st.error(f"تعذّر تحميل {path.name}: {e}")

def zip_bytes(paths: List[Path], arc_prefix: Optional[str] = None) -> Optional[bytes]:
    files = [p for p in paths if p and p.exists()]
    if not files:
        return None
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                arcname = f"{arc_prefix}/{p.name}" if arc_prefix else p.name
                zf.writestr(arcname, data)
            except Exception:
                continue
    return mem.getvalue()

def offer_zip_download(paths: List[Path], zip_name: str, label: Optional[str] = None, key: Optional[str] = None):
    data = zip_bytes(paths)
    if not data:
        st.button(f"❌ لا توجد ملفات متاحة لحزمها ({zip_name})", disabled=True, key=f"{key or zip_name}_na")
        return
    st.download_button(
        label=label or f"⬇️ تحميل حزمة {zip_name}",
        data=data,
        file_name=zip_name,
        mime="application/zip",
        key=key or f"zip_{zip_name}"
    )

def model_file_info() -> List[Tuple[str, Path, bool, Optional[float], Optional[int]]]:
    files = [
        ("matches.json", config.DATA_DIR / "matches.json"),
        ("teams.json", config.DATA_DIR / "teams.json"),
        ("league_averages.json", config.MODELS_DIR / "league_averages.json"),
        ("team_factors.json", config.MODELS_DIR / "team_factors.json"),
        ("elo_ratings.json", config.MODELS_DIR / "elo_ratings.json"),
        ("rho_values.json", config.MODELS_DIR / "rho_values.json"),
        ("ml_dataset.csv", config.DATA_DIR / "ml_dataset.csv"),
        ("xgboost_model.json", config.MODELS_DIR / "xgboost_model.json"),
        ("backtest_results.json", config.DATA_DIR / "backtest_results.json"),
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

    st.subheader("1) بناء البيانات")
    years = st.number_input("عدد المواسم المطلوب جلبها", min_value=1, max_value=20, value=3, step=1)

    pipeline_logs = None
    if st.button("تشغيل بناء البيانات (01_pipeline)"):
        with st.spinner("⏳ جارٍ بناء قاعدة البيانات... هذه العملية قد تستغرق عدة دقائق."):
            ok, pipeline_logs = run_cli_script([sys.executable, "01_pipeline.py", "--years", str(years)])
        safe_clear_cache()
        st.success("✅ اكتملت عملية بناء البيانات.") if ok else st.error("❌ فشل بناء البيانات.")

    with st.expander("تنزيل ملفات البيانات (بعد إتمام 01_pipeline)", expanded=False):
        data_matches = config.DATA_DIR / "matches.json"
        data_teams = config.DATA_DIR / "teams.json"
        col1, col2 = st.columns(2)
        with col1:
            offer_file_download(data_matches, "⬇️ تحميل matches.json")
        with col2:
            offer_file_download(data_teams, "⬇️ تحميل teams.json")
        st.divider()
        offer_zip_download([data_matches, data_teams], "data_bundle.zip", "⬇️ تحميل حزمة بيانات (matches + teams)")

        if pipeline_logs:
            st.download_button("⬇️ تحميل سجلات العملية", pipeline_logs, file_name="pipeline_logs.txt", mime="text/plain", key="dl_pipeline_logs")

    st.subheader("2) تدريب النماذج الإحصائية")
    trainer_logs = None
    if st.button("تدريب النماذج الإحصائية (02_trainer)"):
        with st.spinner("⏳ جارٍ تدريب النماذج الإحصائية (Elo, Factors, Rho)..."):
            ok, trainer_logs = run_cli_script([sys.executable, "02_trainer.py"])
        safe_clear_cache()
        st.success("✅ اكتمل تدريب النماذج الإحصائية.") if ok else st.error("❌ فشل تدريب النماذج.")

    with st.expander("تنزيل ملفات النماذج الإحصائية (بعد إتمام 02_trainer)", expanded=False):
        f_avg = config.MODELS_DIR / "league_averages.json"
        f_fac = config.MODELS_DIR / "team_factors.json"
        f_elo = config.MODELS_DIR / "elo_ratings.json"
        f_rho = config.MODELS_DIR / "rho_values.json"
        c1, c2 = st.columns(2)
        with c1:
            offer_file_download(f_avg, "⬇️ league_averages.json")
            offer_file_download(f_elo, "⬇️ elo_ratings.json")
        with c2:
            offer_file_download(f_fac, "⬇️ team_factors.json")
            offer_file_download(f_rho, "⬇️ rho_values.json")
        st.divider()
        offer_zip_download([f_avg, f_fac, f_elo, f_rho], "stat_models.zip", "⬇️ تحميل حزمة النماذج الإحصائية")

        if trainer_logs:
            st.download_button("⬇️ تحميل سجلات العملية", trainer_logs, file_name="trainer_logs.txt", mime="text/plain", key="dl_trainer_logs")

    st.subheader("3) الاختبار التاريخي (اختياري)")
    backtester_logs = None
    if st.button("إجراء الاختبار التاريخي (03_backtester)"):
        with st.spinner("⏳ جارٍ إجراء الاختبار التاريخي لتقييم النموذج..."):
            ok, backtester_logs = run_cli_script([sys.executable, "03_backtester.py", "--save"])
        st.success("✅ اكتمل الاختبار التاريخي.") if ok else st.error("❌ فشل الاختبار.")

    with st.expander("تنزيل نتائج الاختبار التاريخي (بعد تشغيل 03_backtester)", expanded=False):
        f_bt = config.DATA_DIR / "backtest_results.json"
        offer_file_download(f_bt, "⬇️ backtest_results.json")
        if backtester_logs:
            st.download_button("⬇️ تحميل سجلات العملية", backtester_logs, file_name="backtester_logs.txt", mime="text/plain", key="dl_backtester_logs")

    st.subheader("4) إنشاء ميزات تعلم الآلة")
    features_logs = None
    if st.button("إنشاء ميزات التدريب (04_feature_generator)"):
        with st.spinner("⏳ جارٍ إنشاء ملف الميزات لنموذج تعلم الآلة..."):
            ok, features_logs = run_cli_script([sys.executable, "04_feature_generator.py"])
        st.success("✅ تم إنشاء ملف الميزات بنجاح.") if ok else st.error("❌ فشل إنشاء الميزات.")

    with st.expander("تنزيل ميزات تعلم الآلة (بعد 04_feature_generator)", expanded=False):
        f_ds = config.DATA_DIR / "ml_dataset.csv"
        offer_file_download(f_ds, "⬇️ ml_dataset.csv")
        if features_logs:
            st.download_button("⬇️ تحميل سجلات العملية", features_logs, file_name="features_logs.txt", mime="text/plain", key="dl_features_logs")

    st.subheader("5) تدريب نموذج تعلّم الآلة")
    ml_train_logs = None
    if st.button("تدريب نموذج ML (05_train_ml_model)"):
        st.warning("تأكد من إنشاء ملف الميزات أولاً. هذه العملية قد تستغرق بعض الوقت.")
        with st.spinner("⏳ جارٍ تدريب نموذج XGBoost..."):
            ok, ml_train_logs = run_cli_script([sys.executable, "05_train_ml_model.py"])
        safe_clear_cache()
        st.success("✅ اكتمل تدريب نموذج تعلم الآلة.") if ok else st.error("❌ فشل التدريب.")

    with st.expander("تنزيل نموذج تعلم الآلة (بعد 05_train_ml_model)", expanded=False):
        f_xgb = config.MODELS_DIR / "xgboost_model.json"
        offer_file_download(f_xgb, "⬇️ xgboost_model.json")
        if ml_train_logs:
            st.download_button("⬇️ تحميل سجلات العملية", ml_train_logs, file_name="ml_train_logs.txt", mime="text/plain", key="dl_ml_train_logs")

    st.subheader("6) تشغيل سكربت تنبؤ ML (اختياري)")
    ml_pred_logs = None
    if st.button("تشغيل توقع ML (06_predict_ml)"):
        st.info("سيتم تشغيل التوقع للمباراة المحددة داخل ملف 06_predict_ml.py.")
        with st.spinner("⏳ جارٍ تشغيل خبير تعلم الآلة للتنبؤ..."):
            ok, ml_pred_logs = run_cli_script([sys.executable, "06_predict_ml.py"])
        st.success("✅ تم تشغيل الخبير بنجاح.") if ok else st.error("❌ فشل تشغيل الخبير.")

    if ml_pred_logs:
        with st.expander("سجلات 06_predict_ml"):
            st.code(ml_pred_logs)
            st.download_button("⬇️ تحميل سجلات العملية", ml_pred_logs, file_name="ml_predict_logs.txt", mime="text/plain", key="dl_ml_predict_logs")

    st.divider()
    st.header("إعدادات التنبؤ")
    use_elo = st.checkbox("تفعيل تعديل ELO في λ (للنموذج الإحصائي)", value=True)
    topk = st.slider("أظهر أعلى K من النتائج المحتملة (إحصائي)", min_value=0, max_value=10, value=5)

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
# قسم التنبؤ (النموذج الإحصائي)
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

comp_options = getattr(config, "TARGET_COMPETITIONS", [])
comp_code = st.selectbox("اختر المسابقة", options=comp_options, index=0 if comp_options else 0)

comp_teams = teams_for_comp(teams_map, comp_code) if comp_code else []
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
else:
    if st.button("🔮 احسب التنبؤ الآن (إحصائي)", type="primary"):
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

            # مدخلات ونتيجة كاملة
            with st.expander("عرض مدخلات النموذج"):
                st.json(result.get("model_inputs", {}))
            with st.expander("تفاصيل الفرق (IDs)"):
                st.json(result.get("teams_found", {}))
            with st.expander("النتيجة الكاملة (JSON)"):
                res_json = json.dumps(result, ensure_ascii=False, indent=2)
                st.code(res_json)
                st.download_button("⬇️ تحميل نتيجة التنبؤ (إحصائي)", res_json, file_name="stat_prediction.json", mime="application/json")

        except Exception as e:
            st.error(f"فشل التنبؤ: {e}")

# -----------------------------------------------------------------------------
# قسم التنبؤ (نموذج تعلّم الآلة)
# -----------------------------------------------------------------------------
st.header("تنبؤ المباراة (نموذج تعلّم الآلة)")
with st.container():
    # إعدادات الموسم
    default_year = current_season_year(datetime.now())
    season_year = st.number_input("سنة بداية الموسم", min_value=2000, max_value=2100, value=default_year, step=1)
    model = load_xgb_model()
    if model is None:
        st.warning("نموذج XGBoost غير متوفر. يرجى تدريب نموذج تعلم الآلة أولاً (05_train_ml_model).")
    else:
        # بناء الميزات والتنبؤ
        def compute_ml_prediction(home_name: str, away_name: str, comp_code: str, season_year: int):
            all_matches = load_matches_data()
            if not all_matches:
                raise RuntimeError("matches.json غير متوفر.")
            models = load_models()
            team_factors = models["factors"]
            elo_ratings = models["elo"]

            season_key = f"{comp_code}_{season_year}"
            season_factors = team_factors.get(season_key)
            season_elo = elo_ratings.get(season_key)
            if not season_factors or not season_elo:
                raise RuntimeError(f"لم يتم العثور على نماذج إحصائية للموسم {season_key}.")

            h_id = name_to_id.get(home_name)
            a_id = name_to_id.get(away_name)
            if not h_id or not a_id:
                raise RuntimeError("تعذّر تحديد IDs للفرق المختارة.")

            h_id_str, a_id_str = str(h_id), str(a_id)
            # نستخدم aware-UTC ثم نطبّع إلى Naive-UTC داخل دالة الفورمة
            prediction_date = datetime.now(timezone.utc)

            home_form = calculate_team_form(all_matches, int(h_id), prediction_date, num_matches=5)
            away_form = calculate_team_form(all_matches, int(a_id), prediction_date, num_matches=5)

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

            # توقع الاحتمالات
            predicted_probabilities = model.predict_proba(features_df)

            # ترميز الفئات: (-1 -> 0), (0 -> 1), (1 -> 2)
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit([-1, 0, 1])

            prob_away = float(predicted_probabilities[0][le.transform([-1])[0]])
            prob_draw = float(predicted_probabilities[0][le.transform([0])[0]])
            prob_home = float(predicted_probabilities[0][le.transform([1])[0]])

            result = {
                "meta": {
                    "version": getattr(config, "VERSION", "N/A"),
                    "season_key": season_key,
                    "model": "XGBoostClassifier",
                },
                "match": f"{home_name} (Home) vs {away_name} (Away)",
                "competition": comp_code,
                "teams": {
                    "home": {"name": home_name, "id": h_id},
                    "away": {"name": away_name, "id": a_id},
                },
                "probabilities": {
                    "home_win": prob_home,
                    "draw": prob_draw,
                    "away_win": prob_away,
                },
                "features_used": {k: v[0] for k, v in features_dict.items()},
            }
            return result, features_df

        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            st.write("المسابقة:", comp_code)
        with col_ml2:
            st.write("الموسم:", season_year)
        with col_ml3:
            st.write("الفرق:", f"{team1_name} vs {team2_name}")

        if st.button("🤖 احسب التنبؤ الآن (تعلم الآلة)", type="secondary"):
            try:
                ml_result, ml_features_df = compute_ml_prediction(team1_name, team2_name, comp_code, season_year)

                st.markdown(f"### {ml_result.get('match', '')}")
                st.caption(f"المسابقة: {ml_result.get('competition', comp_code)} | الموسم: {ml_result['meta'].get('season_key')}")

                p_home = float(ml_result["probabilities"]["home_win"])
                p_draw = float(ml_result["probabilities"]["draw"])
                p_away = float(ml_result["probabilities"]["away_win"])

                c1, c2, c3 = st.columns(3)
                c1.metric("فوز المضيف (ML)", f"{p_home*100:.1f}%")
                c2.metric("تعادل (ML)", f"{p_draw*100:.1f}%")
                c3.metric("فوز الضيف (ML)", f"{p_away*100:.1f}%")

                with st.expander("عرض الميزات المُدخلة (ML)"):
                    st.dataframe(ml_features_df)

                with st.expander("النتيجة الكاملة (JSON)"):
                    ml_json = json.dumps(ml_result, ensure_ascii=False, indent=2)
                    st.code(ml_json)
                    st.download_button("⬇️ تحميل نتيجة التنبؤ (ML)", ml_json, file_name="ml_prediction.json", mime="application/json")

            except Exception as e:
                st.error(f"فشل التنبؤ (ML): {e}")
