# app.py
# -----------------------------------------------------------------------------
# لوحة Streamlit لإدارة خط أنابيب البيانات والنماذج، مع أزرار تحميل الملفات
# بعد كل عملية، بالإضافة إلى واجهة تنبؤ إحصائي وتنبؤ بنموذج تعلّم الآلة.
# + إصلاح مقارنة التواريخ (naive vs aware) عبر توحيدها إلى Naive-UTC.
# + إصلاح LabelEncoder واستبداله بـ model.classes_
# + نقل compute_ml_prediction خارج الشرط
# + إضافة timeout لـ subprocess.run
# + توحيد تسمية المتغيرات لتجنب التظليل
# + حماية استيراد sklearn
# + تحسينات عامة في الأمان والوضوح
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

# قد لا يتوفر sklearn في كل بيئة تشغيل، نتعامل بأمان
try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # pragma: no cover
    LabelEncoder = None

from common import config
from common.utils import log
from predictor import Predictor


# -----------------------------------------------------------------------------
# توحيد التواريخ إلى Naive-UTC لمنع أخطاء المقارنة
# -----------------------------------------------------------------------------
def to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    تحويل أي كائن datetime (سواء كان aware أو naive) إلى naive-UTC.
    إذا كان None يعيد None.
    إذا كان naive بالفعل يُرجعه كما هو.
    إذا كان aware يُحوّله إلى UTC ثم يُزيل معلومات المنطقة الزمنية.
    """
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            return dt  # Naive بالفعل
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return dt


# -----------------------------------------------------------------------------
# محاولة استيراد calculate_team_form وإن لم تتوفر نستخدم بديل بسيط
# -----------------------------------------------------------------------------
try:
    from common.modeling import calculate_team_form as _calculate_team_form_base

    def calculate_team_form(
        all_matches: List[Dict], team_id: int, ref_date: datetime, num_matches: int = 5
    ) -> Dict:
        """
        غلاف حول calculate_team_form الأصلية مع توحيد ref_date إلى Naive-UTC
        قبل تمريره لاحتمال المقارنات الداخلية.
        """
        ref_date = to_naive_utc(ref_date)
        return _calculate_team_form_base(
            all_matches, team_id, ref_date, num_matches=num_matches
        )
except Exception:
    from common.utils import parse_date_safe, parse_score

    def calculate_team_form(
        all_matches: List[Dict], team_id: int, ref_date: datetime, num_matches: int = 5
    ) -> Dict:
        """
        بديل بسيط لحساب فورمة الفريق في حال عدم توفر الدالة الأصلية.
        يبحث عن آخر num_matches مباراة للفريق قبل ref_date ويحسب متوسط النقاط.
        يتم توحيد التواريخ إلى Naive-UTC قبل المقارنة.
        """
        ref_date = to_naive_utc(ref_date)
        rows = []
        for m in all_matches:
            # قراءة تاريخ المباراة وتحويله إلى Naive-UTC
            dt = parse_date_safe(m.get("utcDate"))
            dt = to_naive_utc(dt)
            # تخطي المباريات التي ليس لها تاريخ أو التي لم تحدث بعد
            if not dt or dt >= ref_date:
                continue
            # قراءة معرّفات الفريقين
            h = m.get("homeTeam", {}).get("id")
            a = m.get("awayTeam", {}).get("id")
            if not h or not a:
                continue
            # تخطي المباريات التي لا تخص الفريق المطلوب
            if int(h) != team_id and int(a) != team_id:
                continue
            # قراءة النتيجة
            hg, ag = parse_score(m)
            if hg is None:
                continue
            # حساب النقاط بناءً على ما إذا كان الفريق مضيفاً أو ضيفاً
            if int(h) == team_id:
                pts = 3 if hg > ag else (1 if hg == ag else 0)
            else:
                pts = 3 if ag > hg else (1 if hg == ag else 0)
            rows.append((dt, pts))

        # ترتيب حسب التاريخ (الأحدث أولاً) وأخذ آخر num_matches مباراة
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
    """قراءة ملف JSON مع التعامل الآمن مع الأخطاء."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_teams_map() -> Optional[Dict]:
    """تحميل خريطة الفرق من ملف teams.json."""
    return _load_json(config.DATA_DIR / "teams.json")


@st.cache_data
def load_models() -> Dict[str, dict]:
    """تحميل جميع ملفات النماذج الإحصائية."""
    return {
        "averages": _load_json(config.MODELS_DIR / "league_averages.json") or {},
        "factors": _load_json(config.MODELS_DIR / "team_factors.json") or {},
        "elo": _load_json(config.MODELS_DIR / "elo_ratings.json") or {},
        "rho": _load_json(config.MODELS_DIR / "rho_values.json") or {},
    }


@st.cache_data
def load_matches_data() -> Optional[List[Dict]]:
    """تحميل بيانات المباريات من ملف matches.json."""
    return _load_json(config.DATA_DIR / "matches.json")


@st.cache_resource
def get_predictor() -> Predictor:
    """إنشاء أو استرجاع كائن المتنبئ الإحصائي."""
    return Predictor()


@st.cache_resource
def load_xgb_model():
    """
    تحميل نموذج XGBoost المُدرّب من ملف xgboost_model.json.
    يعيد None إذا لم يكن xgboost مثبتاً أو الملف غير موجود.
    """
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
    """مسح جميع أنواع الكاش بأمان."""
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
def run_cli_script(cmd: List[str], timeout_seconds: int = 600) -> Tuple[bool, str]:
    """
    تشغيل أي سكريبت خارجي والتقاط مخرجاته.
    يتم تحديد مهلة زمنية (timeout) لمنع التعليق.
    يتم تحديد مجلد العمل (cwd) كمجلد المشروع الرئيسي.

    المعاملات:
        cmd: قائمة الأمر والمعاملات (مثلاً [sys.executable, "01_pipeline.py", "--years", "3"])
        timeout_seconds: الحد الأقصى لوقت التشغيل بالثواني (افتراضياً 600 ثانية = 10 دقائق)

    العائد: tuple يحتوي على (نجاح العملية, نص المخرجات)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            cwd=project_root,
        )
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except subprocess.TimeoutExpired:
        return False, f"انتهت المهلة الزمنية ({timeout_seconds} ثانية) قبل اكتمال العملية."
    except Exception as e:
        return False, f"فشل في تشغيل السكريبت: {e}"


def current_season_year(now: datetime) -> int:
    """
    تحديد سنة بداية الموسم الحالي بناءً على الشهر الحالي.
    إذا كنا في شهر >= CURRENT_SEASON_START_MONTH فالموسم يبدأ هذا العام،
    وإلا فالموسم يبدأ العام الماضي.
    """
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1


def _primary_name(names: List[str]) -> str:
    """
    اختيار الاسم الأساسي للفريق من قائمة أسماء بديلة.
    قواعد الأفضلية:
        1. الأسماء التي تحتوي على مسافة (أسماء كاملة) تُفضّل على الاختصارات
        2. الأسماء الأطول تُفضّل
        3. الأسماء التي ليست كلها أحرف كبيرة تُفضّل (تجنّب الاختصارات مثل "FCB")

    المعاملات:
        names: قائمة بأسماء بديلة للفريق

    العائد: الاسم الأنسب، أو "Unknown" إذا كانت القائمة فارغة
    """
    names = [n for n in (names or []) if n]
    if not names:
        return "Unknown"

    def score(n: str) -> Tuple[int, int, int]:
        has_space = int(" " in n)          # الأسماء الكاملة (تحتوي مسافة) أولاً
        length = len(n)                    # الأطول أفضل
        not_allcaps = -int(n.isupper())    # تجنّب الاختصارات الكاملة بالأحرف الكبيرة
        return (has_space, length, not_allcaps)

    return sorted(names, key=score, reverse=True)[0]


def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    """
    استخراج قائمة الفرق المشاركة في مسابقة معيّنة.

    المعاملات:
        teams_map: خريطة الفرق (من teams.json)
        comp_code: رمز المسابقة (مثلاً "PL", "CL")

    العائد: قائمة من (اسم_الفريق, معرّف_الفريق) مرتبة أبجدياً
    """
    out = sorted(
        [
            (_primary_name(t.get("names", [])), t.get("id"))
            for t in teams_map.values()
            if comp_code in t.get("competitions", []) and t.get("id")
        ],
        key=lambda x: x[0].lower()
    )
    return out


def compute_prediction(
    team1_name: str,
    team2_name: str,
    comp_code: str,
    use_elo: bool,
    topk: int
) -> Dict:
    """
    حساب التنبؤ الإحصائي (Dixon-Coles) لمباراة معيّنة.

    المعاملات:
        team1_name: اسم الفريق المضيف
        team2_name: اسم الفريق الضيف
        comp_code: رمز المسابقة
        use_elo: تفعيل تعديل ELO في λ
        topk: عدد أعلى النتائج المحتملة للعرض

    العائد: قاموس يحتوي على الاحتمالات والنتائج المحتملة ومدخلات النموذج
    """
    pred = get_predictor()
    result = pred.predict(
        team1_name, team2_name, comp_code, topk=topk, use_elo=use_elo
    )
    return result


def compute_ml_prediction(
    model,
    home_name: str,
    away_name: str,
    comp_code: str,
    season_year: int,
    name_to_id: Dict[str, int],
    all_matches: Optional[List[Dict]],
    models_data: Dict[str, dict]
) -> Tuple[Dict, pd.DataFrame]:
    """
    حساب تنبؤ نموذج تعلم الآلة (XGBoost) لمباراة معيّنة.

    المعاملات:
        model: نموذج XGBoost المُدرّب
        home_name: اسم الفريق المضيف
        away_name: اسم الفريق الضيف
        comp_code: رمز المسابقة
        season_year: سنة بداية الموسم
        name_to_id: قاموس ربط أسماء الفرق بمعرّفاتها
        all_matches: قائمة جميع المباريات من matches.json
        models_data: قاموس يحتوي على بيانات النماذج الإحصائية

    العائد: tuple يحتوي على (قاموس النتيجة, DataFrame الميزات)

    الاستثناءات:
        RuntimeError: في حال عدم توفر النموذج أو البيانات أو الموسم المطلوب
    """
    # التحقق من توفر النموذج
    if model is None:
        raise RuntimeError("نموذج XGBoost غير متوفر. يرجى تدريبه أولاً (05_train_ml_model).")

    # التحقق من توفر بيانات المباريات
    if not all_matches:
        raise RuntimeError("ملف matches.json غير متوفر. يرجى تشغيل بناء البيانات أولاً (01_pipeline).")

    # استخراج بيانات النماذج الإحصائية
    team_factors = models_data.get("factors", {})
    elo_ratings = models_data.get("elo", {})

    # بناء مفتاح الموسم
    season_key = f"{comp_code}_{season_year}"

    # استخراج بيانات الموسم المحدد
    season_factors = team_factors.get(season_key)
    season_elo = elo_ratings.get(season_key)

    if not season_factors:
        raise RuntimeError(
            f"لم يتم العثور على عوامل الفرق (team_factors) للموسم '{season_key}'. "
            f"تأكد من تدريب النماذج الإحصائية لهذا الموسم."
        )

    if not season_elo:
        raise RuntimeError(
            f"لم يتم العثور على تقييمات ELO للموسم '{season_key}'. "
            f"تأكد من تدريب النماذج الإحصائية لهذا الموسم."
        )

    # تحديد معرّفات الفرق
    h_id = name_to_id.get(home_name)
    a_id = name_to_id.get(away_name)

    if not h_id:
        raise RuntimeError(f"تعذّر تحديد معرّف الفريق المضيف '{home_name}'.")
    if not a_id:
        raise RuntimeError(f"تعذّر تحديد معرّف الفريق الضيف '{away_name}'.")

    h_id_str = str(h_id)
    a_id_str = str(a_id)

    # تاريخ التنبؤ (الآن) — نستخدم aware-UTC ثم نطبّع داخل calculate_team_form
    prediction_date = datetime.now(timezone.utc)

    # حساب فورمة الفريقين (آخر 5 مباريات)
    home_form = calculate_team_form(all_matches, int(h_id), prediction_date, num_matches=5)
    away_form = calculate_team_form(all_matches, int(a_id), prediction_date, num_matches=5)

    # استخراج قيم ELO
    home_elo_value = season_elo.get(h_id_str, 1500.0)
    away_elo_value = season_elo.get(a_id_str, 1500.0)

    # استخراج عوامل الهجوم والدفاع
    home_attack_value = season_factors.get("attack", {}).get(h_id_str, 1.0)
    away_attack_value = season_factors.get("attack", {}).get(a_id_str, 1.0)
    home_defense_value = season_factors.get("defense", {}).get(h_id_str, 1.0)
    away_defense_value = season_factors.get("defense", {}).get(a_id_str, 1.0)

    # حساب فرق ELO
    elo_diff_value = home_elo_value - away_elo_value

    # حساب متوسط النقاط (الفورمة)
    home_avg_points_value = home_form.get("avg_points", 1.0)
    away_avg_points_value = away_form.get("avg_points", 1.0)

    # بناء قاموس الميزات
    features_dict = {
        'home_attack': [home_attack_value],
        'away_attack': [away_attack_value],
        'home_defense': [home_defense_value],
        'away_defense': [away_defense_value],
        'home_elo': [home_elo_value],
        'away_elo': [away_elo_value],
        'elo_diff': [elo_diff_value],
        'home_avg_points': [home_avg_points_value],
        'away_avg_points': [away_avg_points_value],
    }

    # تحويل الميزات إلى DataFrame
    features_df = pd.DataFrame.from_dict(features_dict)

    # توقع الاحتمالات باستخدام النموذج
    predicted_probabilities = model.predict_proba(features_df)

    # التحقق من صحة النتائج
    if predicted_probabilities is None or len(predicted_probabilities) == 0:
        raise RuntimeError("فشل النموذج في إنتاج احتمالات. تحقق من توافق الميزات مع النموذج المُدرّب.")

    # استخراج الاحتمالات باستخدام model.classes_ بدلاً من LabelEncoder
    # model.classes_ يحتوي على الفئات بالترتيب الذي تدرّب عليه النموذج
    # الفئات المتوقعة: -1 (فوز ضيف), 0 (تعادل), 1 (فوز مضيف)
    classes_list = list(model.classes_)

    # التحقق من وجود جميع الفئات المتوقعة
    if -1 not in classes_list or 0 not in classes_list or 1 not in classes_list:
        raise RuntimeError(
            f"الفئات في النموذج ({classes_list}) لا تتطابق مع الفئات المتوقعة [-1, 0, 1]. "
            f"قد يحتاج النموذج إلى إعادة تدريب."
        )

    # استخراج فهرس كل فئة من model.classes_
    away_win_index = classes_list.index(-1)
    draw_index = classes_list.index(0)
    home_win_index = classes_list.index(1)

    # استخراج الاحتمالات
    prob_away = float(predicted_probabilities[0][away_win_index])
    prob_draw = float(predicted_probabilities[0][draw_index])
    prob_home = float(predicted_probabilities[0][home_win_index])

    # بناء قاموس النتيجة
    result = {
        "meta": {
            "version": getattr(config, "VERSION", "N/A"),
            "season_key": season_key,
            "model": "XGBClassifier",
            "model_classes": [int(c) for c in classes_list],
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
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
        "form": {
            "home": home_form,
            "away": away_form,
        },
    }

    return result, features_df


def guess_mime(path: Path) -> str:
    """
    تخمين نوع MIME بناءً على امتداد الملف.

    المعاملات:
        path: مسار الملف

    العائد: نص يمثّل نوع MIME
    """
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


def offer_file_download(
    path: Path,
    label: Optional[str] = None,
    key: Optional[str] = None
):
    """
    عرض زر تحميل ملف في واجهة Streamlit.
    إذا كان الملف غير موجود، يُعرض زر معطّل.

    المعاملات:
        path: مسار الملف المطلوب تحميله
        label: نص الزر (اختياري)
        key: مفتاح فريد للزر (اختياري)
    """
    if not path.exists():
        st.button(
            f"❌ {path.name} غير متوفر",
            disabled=True,
            key=f"{key or str(path)}_na"
        )
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
    """
    ضغط مجموعة من الملفات في أرشيف ZIP في الذاكرة.

    المعاملات:
        paths: قائمة مسارات الملفات المطلوب ضغطها
        arc_prefix: بادئة المجلد داخل الأرشيف (اختياري)

    العائد: بيانات ملف ZIP كـ bytes، أو None إذا لم تكن هناك ملفات صالحة
    """
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


def offer_zip_download(
    paths: List[Path],
    zip_name: str,
    label: Optional[str] = None,
    key: Optional[str] = None
):
    """
    عرض زر تحميل حزمة ZIP من مجموعة ملفات في واجهة Streamlit.

    المعاملات:
        paths: قائمة مسارات الملفات المطلوب ضغطها
        zip_name: اسم ملف ZIP الناتج
        label: نص الزر (اختياري)
        key: مفتاح فريد للزر (اختياري)
    """
    data = zip_bytes(paths)
    if not data:
        st.button(
            f"❌ لا توجد ملفات متاحة لحزمها ({zip_name})",
            disabled=True,
            key=f"{key or zip_name}_na"
        )
        return

    st.download_button(
        label=label or f"⬇️ تحميل حزمة {zip_name}",
        data=data,
        file_name=zip_name,
        mime="application/zip",
        key=key or f"zip_{zip_name}"
    )


def model_file_info() -> List[Tuple[str, Path, bool, Optional[float], Optional[int]]]:
    """
    جمع معلومات عن حالة جميع الملفات المهمة (بيانات + نماذج).

    العائد: قائمة من tuples تحتوي:
        (اسم_الملف, المسار, موجود?, وقت_التعديل, الحجم)
    """
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


# =============================================================================
# واجهة المستخدم
# =============================================================================

# --- العنوان الرئيسي ---
st.title("⚽ لوحة تحكم متكاملة لتوقع نتائج المباريات")
st.caption(
    "Dixon–Coles + Team Factors + ELO + XGBoost | الإصدار: "
    + str(getattr(config, "VERSION", "N/A"))
)


# =============================================================================
# الشريط الجانبي (Sidebar)
# =============================================================================
with st.sidebar:
    st.header("إدارة البيانات والنماذج")
    st.info("يُفضّل تشغيل العمليات بالترتيب لضمان عمل التطبيق بشكل صحيح.")

    # =========================================================================
    # 1) بناء البيانات
    # =========================================================================
    st.subheader("1) بناء البيانات")
    years = st.number_input(
        "عدد المواسم المطلوب جلبها",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )
    pipeline_logs = None
    if st.button("تشغيل بناء البيانات (01_pipeline)"):
        with st.spinner("⏳ جارٍ بناء قاعدة البيانات... هذه العملية قد تستغرق عدة دقائق."):
            ok, pipeline_logs = run_cli_script(
                [sys.executable, "01_pipeline.py", "--years", str(years)],
                timeout_seconds=900  # 15 دقيقة للعمليات الطويلة
            )
        safe_clear_cache()
        if ok:
            st.success("✅ اكتملت عملية بناء البيانات.")
        else:
            st.error("❌ فشل بناء البيانات.")

    with st.expander("تنزيل ملفات البيانات (بعد إتمام 01_pipeline)", expanded=False):
        data_matches = config.DATA_DIR / "matches.json"
        data_teams = config.DATA_DIR / "teams.json"
        col1, col2 = st.columns(2)
        with col1:
            offer_file_download(data_matches, "⬇️ تحميل matches.json", key="dl_data_matches")
        with col2:
            offer_file_download(data_teams, "⬇️ تحميل teams.json", key="dl_data_teams")
        st.divider()
        offer_zip_download(
            [data_matches, data_teams],
            "data_bundle.zip",
            "⬇️ تحميل حزمة بيانات (matches + teams)",
            key="zip_data_bundle"
        )

    if pipeline_logs:
        st.download_button(
            "⬇️ تحميل سجلات العملية",
            pipeline_logs,
            file_name="pipeline_logs.txt",
            mime="text/plain",
            key="dl_pipeline_logs"
        )

    # =========================================================================
    # 2) تدريب النماذج الإحصائية
    # =========================================================================
    st.subheader("2) تدريب النماذج الإحصائية")
    trainer_logs = None
    if st.button("تدريب النماذج الإحصائية (02_trainer)"):
        with st.spinner("⏳ جارٍ تدريب النماذج الإحصائية (Elo, Factors, Rho)..."):
            ok, trainer_logs = run_cli_script(
                [sys.executable, "02_trainer.py"],
                timeout_seconds=600
            )
        safe_clear_cache()
        if ok:
            st.success("✅ اكتمل تدريب النماذج الإحصائية.")
        else:
            st.error("❌ فشل تدريب النماذج.")

    with st.expander("تنزيل ملفات النماذج الإحصائية (بعد إتمام 02_trainer)", expanded=False):
        f_avg = config.MODELS_DIR / "league_averages.json"
        f_fac = config.MODELS_DIR / "team_factors.json"
        f_elo = config.MODELS_DIR / "elo_ratings.json"
        f_rho = config.MODELS_DIR / "rho_values.json"
        c1, c2 = st.columns(2)
        with c1:
            offer_file_download(f_avg, "⬇️ league_averages.json", key="dl_f_avg")
            offer_file_download(f_elo, "⬇️ elo_ratings.json", key="dl_f_elo")
        with c2:
            offer_file_download(f_fac, "⬇️ team_factors.json", key="dl_f_fac")
            offer_file_download(f_rho, "⬇️ rho_values.json", key="dl_f_rho")
        st.divider()
        offer_zip_download(
            [f_avg, f_fac, f_elo, f_rho],
            "stat_models.zip",
            "⬇️ تحميل حزمة النماذج الإحصائية",
            key="zip_stat_models"
        )

    if trainer_logs:
        st.download_button(
            "⬇️ تحميل سجلات العملية",
            trainer_logs,
            file_name="trainer_logs.txt",
            mime="text/plain",
            key="dl_trainer_logs"
        )

    # =========================================================================
    # 3) الاختبار التاريخي (اختياري)
    # =========================================================================
    st.subheader("3) الاختبار التاريخي (اختياري)")
    backtester_logs = None
    if st.button("إجراء الاختبار التاريخي (03_backtester)"):
        with st.spinner("⏳ جارٍ إجراء الاختبار التاريخي لتقييم النموذج..."):
            ok, backtester_logs = run_cli_script(
                [sys.executable, "03_backtester.py", "--save"],
                timeout_seconds=600
            )
        if ok:
            st.success("✅ اكتمل الاختبار التاريخي.")
        else:
            st.error("❌ فشل الاختبار.")

    with st.expander("تنزيل نتائج الاختبار التاريخي (بعد تشغيل 03_backtester)", expanded=False):
        f_bt = config.DATA_DIR / "backtest_results.json"
        offer_file_download(f_bt, "⬇️ backtest_results.json", key="dl_f_bt")

    if backtester_logs:
        st.download_button(
            "⬇️ تحميل سجلات العملية",
            backtester_logs,
            file_name="backtester_logs.txt",
            mime="text/plain",
            key="dl_backtester_logs"
        )

    # =========================================================================
    # 4) إنشاء ميزات تعلم الآلة
    # =========================================================================
    st.subheader("4) إنشاء ميزات تعلم الآلة")
    features_logs = None
    if st.button("إنشاء ميزات التدريب (04_feature_generator)"):
        with st.spinner("⏳ جارٍ إنشاء ملف الميزات لنموذج تعلم الآلة..."):
            ok, features_logs = run_cli_script(
                [sys.executable, "04_feature_generator.py"],
                timeout_seconds=600
            )
        if ok:
            st.success("✅ تم إنشاء ملف الميزات بنجاح.")
        else:
            st.error("❌ فشل إنشاء الميزات.")

    with st.expander("تنزيل ميزات تعلم الآلة (بعد 04_feature_generator)", expanded=False):
        f_ds = config.DATA_DIR / "ml_dataset.csv"
        offer_file_download(f_ds, "⬇️ ml_dataset.csv", key="dl_f_ds")

    if features_logs:
        st.download_button(
            "⬇️ تحميل سجلات العملية",
            features_logs,
            file_name="features_logs.txt",
            mime="text/plain",
            key="dl_features_logs"
        )

    # =========================================================================
    # 5) تدريب نموذج تعلّم الآلة
    # =========================================================================
    st.subheader("5) تدريب نموذج تعلّم الآلة")
    ml_train_logs = None
    if st.button("تدريب نموذج ML (05_train_ml_model)"):
        st.warning("تأكد من إنشاء ملف الميزات أولاً. هذه العملية قد تستغرق بعض الوقت.")
        with st.spinner("⏳ جارٍ تدريب نموذج XGBoost..."):
            ok, ml_train_logs = run_cli_script(
                [sys.executable, "05_train_ml_model.py"],
                timeout_seconds=900
            )
        safe_clear_cache()
        if ok:
            st.success("✅ اكتمل تدريب نموذج تعلم الآلة.")
        else:
            st.error("❌ فشل التدريب.")

    with st.expander("تنزيل نموذج تعلم الآلة (بعد 05_train_ml_model)", expanded=False):
        f_xgb = config.MODELS_DIR / "xgboost_model.json"
        offer_file_download(f_xgb, "⬇️ xgboost_model.json", key="dl_f_xgb")

    if ml_train_logs:
        st.download_button(
            "⬇️ تحميل سجلات العملية",
            ml_train_logs,
            file_name="ml_train_logs.txt",
            mime="text/plain",
            key="dl_ml_train_logs"
        )

    # =========================================================================
    # 6) تشغيل سكربت تنبؤ ML (اختياري)
    # =========================================================================
    st.subheader("6) تشغيل سكربت تنبؤ ML (اختياري)")
    ml_pred_logs = None
    if st.button("تشغيل توقع ML (06_predict_ml)"):
        st.info("سيتم تشغيل التوقع للمباراة المحددة داخل ملف 06_predict_ml.py.")
        with st.spinner("⏳ جارٍ تشغيل خبير تعلم الآلة للتنبؤ..."):
            ok, ml_pred_logs = run_cli_script(
                [sys.executable, "06_predict_ml.py"],
                timeout_seconds=300
            )
        if ok:
            st.success("✅ تم تشغيل الخبير بنجاح.")
        else:
            st.error("❌ فشل تشغيل الخبير.")

    if ml_pred_logs:
        with st.expander("سجلات 06_predict_ml"):
            st.code(ml_pred_logs)
        st.download_button(
            "⬇️ تحميل سجلات العملية",
            ml_pred_logs,
            file_name="ml_predict_logs.txt",
            mime="text/plain",
            key="dl_ml_predict_logs"
        )

    # =========================================================================
    # إعدادات التنبؤ
    # =========================================================================
    st.divider()
    st.header("إعدادات التنبؤ")
    use_elo = st.checkbox(
        "تفعيل تعديل ELO في λ (للنموذج الإحصائي)",
        value=True
    )
    topk = st.slider(
        "أظهر أعلى K من النتائج المحتملة (إحصائي)",
        min_value=0,
        max_value=10,
        value=5
    )
    if st.button("🔄 تحديث الكاش"):
        safe_clear_cache()
        st.success("تم مسح الكاش بنجاح!")


# =============================================================================
# تشخيص سريع لحالة الملفات
# =============================================================================
with st.expander("🩺 تشخيص الحالة (الملفات والجاهزية)", expanded=False):
    info = model_file_info()
    cols = st.columns(3)
    for i, (name, p, exists, mtime, size) in enumerate(info):
        with cols[i % 3]:
            if exists:
                ts = (
                    datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    if mtime else "-"
                )
                st.success(f"{name}\n{p}")
                st.caption(f"آخر تحديث: {ts} | حجم: {size or 0} بايت")
            else:
                st.error(f"{name} غير متوفر")
                st.caption(str(p))


# =============================================================================
# قسم التنبؤ (النموذج الإحصائي)
# =============================================================================
st.header("تنبؤ المباراة (النموذج الإحصائي)")

# تحميل خريطة الفرق
teams_map = load_teams_map()
if not teams_map:
    st.error(
        "لم يتم العثور على ملف بيانات الفرق teams.json. "
        "يرجى تشغيل 'بناء البيانات' من الشريط الجانبي أولاً."
    )
    st.stop()

# تحميل النماذج الإحصائية والتحقق من اكتمالها
stat_models = load_models()
missing_models = [k for k, v in stat_models.items() if not v]
if missing_models:
    st.warning(
        f"بعض ملفات النماذج غير متوفرة: {', '.join(missing_models)}. "
        f"قد تحتاج لتشغيل تدريب النماذج الإحصائية."
    )

# اختيار المسابقة
comp_options = getattr(config, "TARGET_COMPETITIONS", [])
comp_code = st.selectbox(
    "اختر المسابقة",
    options=comp_options,
    index=0 if comp_options else 0
)

# استخراج الفرق المشاركة في المسابقة المختارة
comp_teams = teams_for_comp(teams_map, comp_code) if comp_code else []
if not comp_teams:
    st.warning(
        f"لم يتم العثور على فرق لمسابقة '{comp_code}'. "
        f"قد تحتاج لتشغيل بناء البيانات."
    )
    st.stop()

# إعداد قوائم الأسماء والربط بالمعرّفات
names = [n for n, _ in comp_teams]
name_to_id = {n: tid for n, tid in comp_teams}

# اختيار الفريقين
c1, c2 = st.columns(2)
with c1:
    team1_name = st.selectbox("الفريق المضيف", options=names, index=0)
with c2:
    team2_name = st.selectbox(
        "الفريق الضيف",
        options=names,
        index=1 if len(names) > 1 else 0
    )

# التحقق من اختلاف الفريقين
if team1_name == team2_name:
    st.warning("يرجى اختيار فريقين مختلفين.")
else:
    if st.button("🔮 احسب التنبؤ الآن (إحصائي)", type="primary"):
        try:
            result = compute_prediction(
                team1_name,
                team2_name,
                comp_code,
                use_elo=use_elo,
                topk=topk
            )

            # عرض عنوان المباراة
            st.markdown(
                f"### {result.get('match', f'{team1_name} vs {team2_name}')}"
            )
            meta = result.get("meta", {})
            st.caption(
                f"المسابقة: {result.get('competition', comp_code)} | "
                f"الموسم المستخدم للنموذج: {meta.get('model_season_used', 'N/A')}"
            )

            # استخراج الاحتمالات
            p_home = float(result["probabilities"]["home_win"])
            p_draw = float(result["probabilities"]["draw"])
            p_away = float(result["probabilities"]["away_win"])

            # عرض الاحتمالات في أعمدة
            col1, col2, col3 = st.columns(3)
            col1.metric("فوز المضيف", f"{p_home * 100:.1f}%")
            col2.metric("تعادل", f"{p_draw * 100:.1f}%")
            col3.metric("فوز الضيف", f"{p_away * 100:.1f}%")

            # عرض أعلى K من النتائج المحتملة
            if topk and "top_scorelines" in result:
                st.subheader(f"أعلى {topk} نتائج محتملة")
                rows = [
                    {
                        "النتيجة": f"{s['home_goals']} - {s['away_goals']}",
                        "الاحتمال": f"{s['prob'] * 100:.2f}%"
                    }
                    for s in result["top_scorelines"]
                ]
                st.table(rows)

            # عرض مدخلات النموذج
            with st.expander("عرض مدخلات النموذج"):
                st.json(result.get("model_inputs", {}))

            # عرض تفاصيل الفرق
            with st.expander("تفاصيل الفرق (IDs)"):
                st.json(result.get("teams_found", {}))

            # عرض النتيجة الكاملة وزر التحميل
            with st.expander("النتيجة الكاملة (JSON)"):
                res_json = json.dumps(result, ensure_ascii=False, indent=2)
                st.code(res_json)
                st.download_button(
                    "⬇️ تحميل نتيجة التنبؤ (إحصائي)",
                    res_json,
                    file_name="stat_prediction.json",
                    mime="application/json",
                    key="dl_stat_pred_result"
                )

        except Exception as e:
            st.error(f"فشل التنبؤ: {e}")


# =============================================================================
# قسم التنبؤ (نموذج تعلّم الآلة)
# =============================================================================
st.header("تنبؤ المباراة (نموذج تعلّم الآلة)")

with st.container():
    # إعدادات الموسم
    default_year = current_season_year(datetime.now())
    season_year = st.number_input(
        "سنة بداية الموسم",
        min_value=2000,
        max_value=2100,
        value=default_year,
        step=1
    )

    # تحميل نموذج XGBoost
    xgb_model = load_xgb_model()
    if xgb_model is None:
        st.warning(
            "نموذج XGBoost غير متوفر. "
            "يرجى تدريب نموذج تعلم الآلة أولاً (05_train_ml_model)."
        )
    else:
        # عرض معلومات المباراة المختارة
        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            st.write("المسابقة:", comp_code)
        with col_ml2:
            st.write("الموسم:", season_year)
        with col_ml3:
            st.write("الفرق:", f"{team1_name} vs {team2_name}")

        # التحقق من اختلاف الفريقين قبل السماح بالتنبؤ
        if team1_name == team2_name:
            st.warning("يرجى اختيار فريقين مختلفين للتنبؤ بنموذج تعلم الآلة.")
        else:
            if st.button("🤖 احسب التنبؤ الآن (تعلم الآلة)", type="secondary"):
                try:
                    # تحميل بيانات المباريات
                    all_matches_data = load_matches_data()
                    # تحميل بيانات النماذج الإحصائية
                    ml_models_data = load_models()

                    # حساب التنبؤ
                    ml_result, ml_features_df = compute_ml_prediction(
                        model=xgb_model,
                        home_name=team1_name,
                        away_name=team2_name,
                        comp_code=comp_code,
                        season_year=season_year,
                        name_to_id=name_to_id,
                        all_matches=all_matches_data,
                        models_data=ml_models_data,
                    )

                    # عرض عنوان المباراة
                    st.markdown(f"### {ml_result.get('match', '')}")
                    st.caption(
                        f"المسابقة: {ml_result.get('competition', comp_code)} | "
                        f"الموسم: {ml_result['meta'].get('season_key')}"
                    )

                    # استخراج الاحتمالات
                    p_home = float(ml_result["probabilities"]["home_win"])
                    p_draw = float(ml_result["probabilities"]["draw"])
                    p_away = float(ml_result["probabilities"]["away_win"])

                    # عرض الاحتمالات في أعمدة
                    c1, c2, c3 = st.columns(3)
                    c1.metric("فوز المضيف (ML)", f"{p_home * 100:.1f}%")
                    c2.metric("تعادل (ML)", f"{p_draw * 100:.1f}%")
                    c3.metric("فوز الضيف (ML)", f"{p_away * 100:.1f}%")

                    # عرض شريط مرئي للاحتمالات
                    st.subheader("توزيع الاحتمالات")
                    prob_df = pd.DataFrame({
                        "النتيجة": ["فوز المضيف", "تعادل", "فوز الضيف"],
                        "الاحتمال": [p_home, p_draw, p_away]
                    })
                    st.bar_chart(
                        prob_df.set_index("النتيجة"),
                        use_container_width=True
                    )

                    # عرض الميزات المُدخلة
                    with st.expander("عرض الميزات المُدخلة (ML)"):
                        st.dataframe(ml_features_df, use_container_width=True)

                    # عرض تفاصيل الفورمة
                    with st.expander("فورمة الفريقين (آخر 5 مباريات)"):
                        form_data = ml_result.get("form", {})
                        form_col1, form_col2 = st.columns(2)
                        with form_col1:
                            st.write(f"**{team1_name} (مضيف):**")
                            st.json(form_data.get("home", {}))
                        with form_col2:
                            st.write(f"**{team2_name} (ضيف):**")
                            st.json(form_data.get("away", {}))

                    # عرض النتيجة الكاملة وزر التحميل
                    with st.expander("النتيجة الكاملة (JSON)"):
                        ml_json = json.dumps(ml_result, ensure_ascii=False, indent=2)
                        st.code(ml_json)
                        st.download_button(
                            "⬇️ تحميل نتيجة التنبؤ (ML)",
                            ml_json,
                            file_name="ml_prediction.json",
                            mime="application/json",
                            key="dl_ml_pred_result"
                        )

                except RuntimeError as e:
                    st.error(f"فشل التنبؤ (ML): {e}")
                except ValueError as e:
                    st.error(
                        f"خطأ في القيم المُدخلة (ML): {e}. "
                        f"تأكد من توافق الميزات مع النموذج المُدرّب."
                    )
                except Exception as e:
                    st.error(f"خطأ غير متوقع في التنبؤ (ML): {e}")


# =============================================================================
# تذييل الصفحة
# =============================================================================
st.divider()
st.caption(
    "⚽ Football Predictor — "
    "Dixon–Coles + Team Factors + ELO + XGBoost | "
    f"الإصدار: {getattr(config, 'VERSION', 'N/A')} | "
    f"آخر تحديث للصفحة: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
  )
