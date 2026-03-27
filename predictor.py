# predictor.py
# -----------------------------------------------------------------------------
# كلاس Predictor موحّد للتنبؤ بنتائج المباريات باستخدام نموذج Dixon-Coles.
#
# يقوم بـ:
# 1. تحميل النماذج الإحصائية المدرّبة (Team Factors, ELO, League Averages, Rho)
# 2. البحث عن الفرق بالاسم أو المعرّف
# 3. حساب معدّلات الأهداف المتوقعة (λ) مع إمكانية تعديل ELO
# 4. حساب مصفوفة الاحتمالات (Dixon-Coles + Poisson)
# 5. إرجاع احتمالات (فوز المضيف، تعادل، فوز الضيف) وأفضل النتائج المحتملة
#
# التحسينات:
# - معالجة أخطاء شاملة عند تحميل الملفات (لا يتوقف التطبيق إذا كان ملف مفقوداً)
# - إصلاح type hint لـ _adjust_lambdas_with_elo (Tuple بدلاً من tuple القديم)
# - التحقق من صحة البيانات المحمّلة
# - دعم إعادة تحميل النماذج بدون إعادة إنشاء الكائن
# - بحث أكثر ذكاءً عن الموسم المناسب (يبحث في أكثر من موسمين)
# - تسجيل أوضح مع رسائل خطأ مفصّلة
# - حماية من القيم غير الصالحة (NaN, Inf, سالبة)
# - إضافة بيانات وصفية أغنى في النتيجة
# - دعم التنبؤ بالمعرّف مباشرة بدلاً من الاسم فقط
# - إضافة التحقق من توافق الميزات
# - توثيق شامل لجميع الدوال
# -----------------------------------------------------------------------------

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# استيراد الوحدات المشتركة
from common import config
from common.utils import log, enhanced_team_search

# استيراد دوال النمذجة
from common.modeling import (
    poisson_matrix_dc,
    matrix_to_outcomes,
    top_scorelines,
    suggest_goal_cutoff,
)


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# القيمة الافتراضية لتقييم ELO للفرق غير المعروفة
DEFAULT_ELO = 1500.0

# القيم الافتراضية لمتوسطات الدوري
DEFAULT_AVG_HOME_GOALS = 1.40
DEFAULT_AVG_AWAY_GOALS = 1.10

# القيمة الافتراضية لعوامل الفرق
DEFAULT_FACTOR = 1.0

# القيمة الافتراضية لمعامل الارتباط rho
DEFAULT_RHO = 0.0

# الحد الأقصى لعدد المواسم السابقة للبحث عن نموذج متاح
MAX_SEASON_LOOKBACK = 5

# الحد الأدنى لقيمة lambda (لمنع القسمة على صفر)
MIN_LAMBDA = 1e-6


# -----------------------------------------------------------------------------
# دوال مساعدة
# -----------------------------------------------------------------------------

def current_season_year(now: Optional[datetime] = None) -> int:
    """
    تحديد سنة بداية الموسم الكروي الحالي.

    يستخدم config.CURRENT_SEASON_START_MONTH لتحديد شهر بداية الموسم.
    مثلاً: إذا كان شهر البداية هو 7 (يوليو) وكنا في يونيو 2025،
    فالموسم الحالي بدأ في 2024.

    المعاملات:
        now: التاريخ الحالي (اختياري، يُستخدم datetime.now() إن لم يُحدد)

    العائد:
        سنة بداية الموسم الحالي
    """
    if now is None:
        now = datetime.now()

    season_start_month = getattr(config, "CURRENT_SEASON_START_MONTH", 7)

    if now.month >= season_start_month:
        return now.year
    else:
        return now.year - 1


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    تحويل قيمة إلى float بأمان مع التعامل مع القيم غير الصالحة.

    المعاملات:
        value: القيمة المراد تحويلها
        default: القيمة الافتراضية في حالة الفشل

    العائد:
        القيمة كـ float، أو القيمة الافتراضية
    """
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def validate_probability(prob: float) -> float:
    """
    التحقق من أن القيمة احتمال صالح (بين 0 و 1).

    المعاملات:
        prob: القيمة المراد التحقق منها

    العائد:
        القيمة المُصححة (بين 0.0 و 1.0)
    """
    if math.isnan(prob) or math.isinf(prob):
        return 0.0
    return max(0.0, min(1.0, prob))


# -----------------------------------------------------------------------------
# كلاس Predictor
# -----------------------------------------------------------------------------

class Predictor:
    """
    المتنبئ الموحّد لنتائج المباريات باستخدام نموذج Dixon-Coles الإحصائي.

    يقوم بتحميل النماذج المدرّبة مسبقاً ويوفّر واجهة بسيطة للتنبؤ
    بنتائج المباريات عبر أسماء الفرق أو معرّفاتها.

    المُكوّنات المُحمّلة:
    - league_averages.json: متوسطات أهداف المضيف والضيف لكل موسم
    - team_factors.json: عوامل الهجوم والدفاع لكل فريق في كل موسم
    - elo_ratings.json: تقييمات ELO لكل فريق في كل موسم
    - rho_values.json: معامل الارتباط Dixon-Coles لكل موسم
    - teams.json: خريطة الفرق (أسماء ومعرّفات)

    مثال الاستخدام:
        predictor = Predictor()
        result = predictor.predict("Manchester City", "Liverpool", "PL", topk=5, use_elo=True)
        print(result["probabilities"])
    """

    def __init__(self, auto_load: bool = True):
        """
        إنشاء كائن المتنبئ.

        المعاملات:
            auto_load: إذا True، يتم تحميل النماذج تلقائياً عند الإنشاء.
                       إذا False، يجب استدعاء reload() يدوياً.
        """
        self.models: Dict[str, Dict] = {}
        self.teams_map: Dict = {}
        self._loaded: bool = False
        self._load_errors: List[str] = []

        if auto_load:
            self.reload()

    # =========================================================================
    # تحميل البيانات
    # =========================================================================

    def _load_json_safe(self, path: Path, description: str) -> Optional[Dict]:
        """
        تحميل ملف JSON بأمان بدون إيقاف التطبيق.

        المعاملات:
            path: مسار الملف
            description: وصف الملف (للرسائل)

        العائد:
            محتوى الملف كقاموس، أو None في حالة الفشل
        """
        try:
            if not path.exists():
                error_msg = f"ملف {description} غير موجود: {path}"
                log(error_msg, "WARNING")
                self._load_errors.append(error_msg)
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, (dict, list)):
                error_msg = (
                    f"تنسيق ملف {description} غير متوقع: "
                    f"المتوقع dict/list، الموجود {type(data).__name__}"
                )
                log(error_msg, "WARNING")
                self._load_errors.append(error_msg)
                return None

            log(f"تم تحميل {description}: {path.name}", "INFO")
            return data

        except json.JSONDecodeError as e:
            error_msg = f"خطأ في تحليل ملف {description}: {e}"
            log(error_msg, "WARNING")
            self._load_errors.append(error_msg)
            return None

        except Exception as e:
            error_msg = f"خطأ أثناء تحميل {description}: {e}"
            log(error_msg, "WARNING")
            self._load_errors.append(error_msg)
            return None

    def _load_models(self) -> Dict[str, Dict]:
        """
        تحميل جميع ملفات النماذج الإحصائية.

        لا يتوقف إذا كان أحد الملفات مفقوداً — يستخدم قاموساً فارغاً بدلاً منه.

        العائد:
            قاموس يحتوي على جميع النماذج المحمّلة
        """
        log("جارٍ تحميل النماذج الإحصائية المدرّبة...", "INFO")

        models = {
            "averages": (
                self._load_json_safe(
                    config.MODELS_DIR / "league_averages.json",
                    "متوسطات الدوري"
                )
                or {}
            ),
            "factors": (
                self._load_json_safe(
                    config.MODELS_DIR / "team_factors.json",
                    "عوامل الفرق"
                )
                or {}
            ),
            "elo": (
                self._load_json_safe(
                    config.MODELS_DIR / "elo_ratings.json",
                    "تقييمات ELO"
                )
                or {}
            ),
            "rho": (
                self._load_json_safe(
                    config.MODELS_DIR / "rho_values.json",
                    "قيم rho"
                )
                or {}
            ),
        }

        # ملخص التحميل
        for model_name, model_data in models.items():
            season_count = len(model_data) if isinstance(model_data, dict) else 0
            log(f"  {model_name}: {season_count} موسم", "INFO")

        return models

    def _load_teams_map(self) -> Dict:
        """
        تحميل خريطة الفرق من ملف teams.json.

        العائد:
            قاموس الفرق، أو قاموس فارغ في حالة الفشل
        """
        log("جارٍ تحميل خريطة الفرق...", "INFO")

        teams_map = self._load_json_safe(
            config.DATA_DIR / "teams.json",
            "خريطة الفرق"
        )

        if teams_map is None:
            return {}

        if isinstance(teams_map, dict):
            log(f"  عدد الفرق: {len(teams_map)}", "INFO")
            return teams_map

        # في حالة كان الملف قائمة بدلاً من قاموس
        log(
            "تنسيق خريطة الفرق غير متوقع (ليس قاموساً). "
            "سيتم تحويله.",
            "WARNING"
        )
        return {}

    def reload(self):
        """
        إعادة تحميل جميع النماذج وخريطة الفرق.
        مفيد بعد إعادة تدريب النماذج بدون إعادة تشغيل التطبيق.
        """
        log("=" * 50, "INFO")
        log("إعادة تحميل النماذج وخريطة الفرق", "INFO")
        log("=" * 50, "INFO")

        self._load_errors = []
        self.models = self._load_models()
        self.teams_map = self._load_teams_map()

        # تحديد حالة التحميل
        required_models = ["averages", "factors", "elo", "rho"]
        all_loaded = all(
            bool(self.models.get(m)) for m in required_models
        )

        self._loaded = all_loaded and bool(self.teams_map)

        if self._loaded:
            log("✅ تم تحميل جميع النماذج بنجاح.", "INFO")
        else:
            missing = [
                m for m in required_models if not self.models.get(m)
            ]
            if missing:
                log(
                    f"⚠ نماذج مفقودة: {missing}. "
                    f"بعض وظائف التنبؤ قد لا تعمل.",
                    "WARNING"
                )
            if not self.teams_map:
                log(
                    "⚠ خريطة الفرق مفقودة. البحث بالاسم لن يعمل.",
                    "WARNING"
                )

    @property
    def is_loaded(self) -> bool:
        """هل تم تحميل جميع النماذج بنجاح."""
        return self._loaded

    @property
    def load_errors(self) -> List[str]:
        """قائمة أخطاء التحميل (إن وُجدت)."""
        return self._load_errors.copy()

    @property
    def available_seasons(self) -> List[str]:
        """قائمة المواسم المتاحة في النماذج المحمّلة."""
        seasons: set = set()

        for model_name in ("averages", "factors", "elo", "rho"):
            model_data = self.models.get(model_name, {})
            if isinstance(model_data, dict):
                seasons.update(model_data.keys())

        return sorted(seasons)

    @property
    def complete_seasons(self) -> List[str]:
        """
        قائمة المواسم التي تتوفر فيها جميع مكونات النموذج
        (averages + factors + elo + rho).
        """
        required_models = ("averages", "factors", "elo", "rho")
        all_keys: List[set] = []

        for model_name in required_models:
            model_data = self.models.get(model_name, {})
            if isinstance(model_data, dict):
                all_keys.append(set(model_data.keys()))
            else:
                all_keys.append(set())

        if not all_keys:
            return []

        # المواسم الموجودة في جميع النماذج
        common_keys = all_keys[0]
        for keys in all_keys[1:]:
            common_keys = common_keys.intersection(keys)

        return sorted(common_keys)

    # =========================================================================
    # البحث عن الفرق
    # =========================================================================

    def find_team_id(
        self,
        team_identifier: Union[str, int],
        comp_code: Optional[str] = None,
    ) -> Optional[int]:
        """
        البحث عن معرّف فريق بالاسم أو التحقق من صلاحية المعرّف.

        إذا كان المدخل عدداً صحيحاً، يتم إرجاعه مباشرة.
        إذا كان نصاً، يتم البحث في خريطة الفرق.

        المعاملات:
            team_identifier: اسم الفريق (str) أو معرّفه (int)
            comp_code: رمز المسابقة (اختياري — لتضييق البحث)

        العائد:
            معرّف الفريق، أو None إذا لم يُوجد
        """
        # إذا كان عدداً صحيحاً، نُرجعه مباشرة
        if isinstance(team_identifier, int):
            return team_identifier

        # إذا كان نصاً يمثّل عدداً
        if isinstance(team_identifier, str):
            try:
                return int(team_identifier)
            except ValueError:
                pass

        # البحث بالاسم
        if not self.teams_map:
            log(
                f"خريطة الفرق غير محمّلة. لا يمكن البحث عن: '{team_identifier}'",
                "WARNING"
            )
            return None

        team_id = enhanced_team_search(
            str(team_identifier),
            self.teams_map,
            comp_code or ""
        )

        return team_id

    def get_team_name(self, team_id: int) -> str:
        """
        الحصول على اسم الفريق من معرّفه.

        المعاملات:
            team_id: معرّف الفريق

        العائد:
            اسم الفريق، أو "Team {id}" إذا لم يُوجد
        """
        if not self.teams_map:
            return f"Team {team_id}"

        team_id_str = str(team_id)

        for team_key, team_data in self.teams_map.items():
            if not isinstance(team_data, dict):
                continue

            if str(team_data.get("id")) == team_id_str:
                names = team_data.get("names", [])
                if isinstance(names, list):
                    valid_names = [n for n in names if n and isinstance(n, str)]
                    if valid_names:
                        # تفضيل الأسماء الكاملة (تحتوي مسافة) والأطول
                        def name_score(n: str) -> Tuple[int, int]:
                            return (int(" " in n), len(n))

                        return max(valid_names, key=name_score)

                return f"Team {team_id}"

        return f"Team {team_id}"

    # =========================================================================
    # اختيار الموسم
    # =========================================================================

    def _select_season_key(
        self,
        comp_code: str,
        preferred_year: Optional[int] = None,
    ) -> str:
        """
        اختيار مفتاح الموسم الأنسب للتنبؤ.

        يحاول الموسم الحالي أولاً، ثم يبحث في المواسم السابقة.
        يتحقق من توفر جميع مكونات النموذج (averages, factors, elo, rho).

        المعاملات:
            comp_code: رمز المسابقة
            preferred_year: سنة بداية الموسم المفضّل (اختياري)

        العائد:
            مفتاح الموسم المختار

        الاستثناءات:
            ValueError: إذا لم يتم العثور على نموذج كامل لأي موسم
        """
        required_models = ("elo", "factors", "averages", "rho")

        # تحديد سنة البداية
        if preferred_year is not None:
            start_year = preferred_year
        else:
            start_year = current_season_year()

        # البحث في المواسم (من الأحدث إلى الأقدم)
        for offset in range(MAX_SEASON_LOOKBACK):
            year = start_year - offset
            candidate_key = f"{comp_code}_{year}"

            # التحقق من توفر جميع المكونات
            all_available = True
            for model_name in required_models:
                model_data = self.models.get(model_name, {})
                if not isinstance(model_data, dict):
                    all_available = False
                    break
                if candidate_key not in model_data:
                    all_available = False
                    break

            if all_available:
                if offset > 0:
                    log(
                        f"لم يتوفر نموذج للموسم الحالي ({comp_code}_{start_year}). "
                        f"تم استخدام: {candidate_key}",
                        "WARNING"
                    )
                return candidate_key

        # لم يتم العثور على أي موسم كامل
        available = self.complete_seasons
        comp_seasons = [s for s in available if s.startswith(f"{comp_code}_")]

        if comp_seasons:
            # استخدام أحدث موسم متاح لهذه المسابقة
            latest = comp_seasons[-1]
            log(
                f"لم يتوفر نموذج حديث لـ {comp_code}. "
                f"استخدام أحدث موسم متاح: {latest}",
                "WARNING"
            )
            return latest

        raise ValueError(
            f"لا يوجد نموذج إحصائي كامل متاح لمسابقة '{comp_code}'. "
            f"المواسم الكاملة المتاحة: {available or 'لا يوجد'}. "
            f"يرجى تشغيل 02_trainer.py أولاً."
        )

    # =========================================================================
    # حساب λ وتعديل ELO
    # =========================================================================

    def _compute_base_lambdas(
        self,
        home_id: int,
        away_id: int,
        season_key: str,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        حساب معدّلات الأهداف الأساسية (λ) باستخدام عوامل الفرق ومتوسطات الدوري.

        λ_home = A_home * D_away * avg_home_goals
        λ_away = A_away * D_home * avg_away_goals

        المعاملات:
            home_id: معرّف الفريق المضيف
            away_id: معرّف الفريق الضيف
            season_key: مفتاح الموسم

        العائد:
            tuple يحتوي على:
            - lambda_home: معدّل أهداف المضيف
            - lambda_away: معدّل أهداف الضيف
            - details: قاموس تفاصيل الحساب (للتوثيق)
        """
        h_id_str = str(home_id)
        a_id_str = str(away_id)

        # --- متوسطات الدوري ---
        avgs = self.models.get("averages", {}).get(season_key, {})
        avg_home = safe_float(avgs.get("avg_home_goals", DEFAULT_AVG_HOME_GOALS), DEFAULT_AVG_HOME_GOALS)
        avg_away = safe_float(avgs.get("avg_away_goals", DEFAULT_AVG_AWAY_GOALS), DEFAULT_AVG_AWAY_GOALS)

        # --- عوامل الفرق ---
        factors = self.models.get("factors", {}).get(season_key, {})
        attack_factors = factors.get("attack", {})
        defense_factors = factors.get("defense", {})

        home_attack = safe_float(attack_factors.get(h_id_str, DEFAULT_FACTOR), DEFAULT_FACTOR)
        home_defense = safe_float(defense_factors.get(h_id_str, DEFAULT_FACTOR), DEFAULT_FACTOR)
        away_attack = safe_float(attack_factors.get(a_id_str, DEFAULT_FACTOR), DEFAULT_FACTOR)
        away_defense = safe_float(defense_factors.get(a_id_str, DEFAULT_FACTOR), DEFAULT_FACTOR)

        # --- حساب λ ---
        lam_home = home_attack * away_defense * avg_home
        lam_away = away_attack * home_defense * avg_away

        # ضمان قيم موجبة
        lam_home = max(lam_home, MIN_LAMBDA)
        lam_away = max(lam_away, MIN_LAMBDA)

        # تفاصيل الحساب
        details = {
            "avg_home_goals": round(avg_home, 4),
            "avg_away_goals": round(avg_away, 4),
            "home_attack": round(home_attack, 4),
            "home_defense": round(home_defense, 4),
            "away_attack": round(away_attack, 4),
            "away_defense": round(away_defense, 4),
            "lambda_home_base": round(lam_home, 4),
            "lambda_away_base": round(lam_away, 4),
            "home_in_factors": h_id_str in attack_factors,
            "away_in_factors": a_id_str in attack_factors,
        }

        return lam_home, lam_away, details

    def _get_elo_ratings(
        self,
        home_id: int,
        away_id: int,
        season_key: str,
    ) -> Tuple[float, float]:
        """
        استرجاع تقييمات ELO للفريقين.

        المعاملات:
            home_id: معرّف المضيف
            away_id: معرّف الضيف
            season_key: مفتاح الموسم

        العائد:
            tuple يحتوي على (elo_home, elo_away)
        """
        elo_data = self.models.get("elo", {}).get(season_key, {})

        elo_home = safe_float(
            elo_data.get(str(home_id), DEFAULT_ELO),
            DEFAULT_ELO
        )
        elo_away = safe_float(
            elo_data.get(str(away_id), DEFAULT_ELO),
            DEFAULT_ELO
        )

        return elo_home, elo_away

    def _get_rho(self, season_key: str) -> float:
        """
        استرجاع معامل الارتباط (rho) لموسم معيّن.

        المعاملات:
            season_key: مفتاح الموسم

        العائد:
            قيمة rho
        """
        rho_data = self.models.get("rho", {})

        if isinstance(rho_data, dict):
            rho_value = rho_data.get(season_key, DEFAULT_RHO)
        else:
            rho_value = DEFAULT_RHO

        return safe_float(rho_value, DEFAULT_RHO)

    def _adjust_lambdas_with_elo(
        self,
        lam_home: float,
        lam_away: float,
        elo_home: float,
        elo_away: float,
    ) -> Tuple[float, float]:
        """
        تعديل معدّلات الأهداف (λ) باستخدام فرق تقييمات ELO.

        يحسب عامل التعديل بناءً على:
        - فرق ELO بين الفريقين
        - أفضلية الأرض (Home Field Advantage)
        - مقياس التحويل (ELO_LAMBDA_SCALE)

        المعاملات:
            lam_home: معدّل أهداف المضيف الأصلي
            lam_away: معدّل أهداف الضيف الأصلي
            elo_home: تقييم ELO للمضيف
            elo_away: تقييم ELO للضيف

        العائد:
            tuple يحتوي على (lambda_home_adjusted, lambda_away_adjusted)
        """
        # استخراج إعدادات ELO
        elo_hfa = safe_float(
            getattr(config, "ELO_HFA", 60.0),
            60.0
        )
        elo_lambda_scale = safe_float(
            getattr(config, "ELO_LAMBDA_SCALE", 400.0),
            400.0
        )

        # منع القسمة على صفر
        if elo_lambda_scale == 0:
            elo_lambda_scale = 400.0

        # حساب الأفضلية الإجمالية
        edge = (elo_home - elo_away) + elo_hfa

        # حساب عامل التعديل
        # تقييد edge لمنع overflow في الأُس
        clamped_edge = max(-800.0, min(800.0, edge))
        factor = 10.0 ** (clamped_edge / elo_lambda_scale)

        # تعديل λ
        adjusted_home = lam_home * factor
        adjusted_away = max(MIN_LAMBDA, lam_away / factor)

        return adjusted_home, adjusted_away

    # =========================================================================
    # التنبؤ الرئيسي
    # =========================================================================

    def predict(
        self,
        team1_name: str,
        team2_name: str,
        comp_code: str,
        topk: int = 0,
        use_elo: bool = False,
        preferred_season_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        التنبؤ بنتيجة مباراة واحدة.

        المعاملات:
            team1_name: اسم الفريق المضيف (أو معرّفه كنص)
            team2_name: اسم الفريق الضيف (أو معرّفه كنص)
            comp_code: رمز المسابقة (مثلاً "PL", "CL", "PD")
            topk: عدد أفضل النتائج المحتملة للإرجاع (0 = بدون)
            use_elo: تفعيل تعديل ELO في حساب λ
            preferred_season_year: سنة الموسم المفضّل (None = تلقائي)

        العائد:
            قاموس يحتوي على:
            - meta: بيانات وصفية (الإصدار، الموسم المُستخدم)
            - match: وصف المباراة
            - competition: رمز المسابقة
            - teams_found: معلومات الفرق (أسماء ومعرّفات)
            - model_inputs: مدخلات النموذج (λ, rho, إلخ)
            - probabilities: الاحتمالات (home_win, draw, away_win)
            - top_scorelines: أفضل النتائج المحتملة (إذا topk > 0)

        الاستثناءات:
            ValueError: إذا لم يتم العثور على الفرق أو النماذج
        """
        log("--- بدء عملية التنبؤ ---", "DEBUG")

        # --- الخطوة 1: تنظيف المدخلات ---
        comp_code = comp_code.strip().upper()

        log(
            f"الخطوة 1: طلب تنبؤ — "
            f"{team1_name} vs {team2_name} في {comp_code}",
            "DEBUG"
        )

        # --- الخطوة 2: البحث عن الفرق ---
        home_id = self.find_team_id(team1_name, comp_code)
        away_id = self.find_team_id(team2_name, comp_code)

        log(
            f"الخطوة 2: نتيجة البحث — "
            f"المضيف ID: {home_id}, الضيف ID: {away_id}",
            "DEBUG"
        )

        if not home_id:
            raise ValueError(
                f"لم يتم العثور على الفريق المضيف: '{team1_name}' "
                f"في مسابقة '{comp_code}'."
            )

        if not away_id:
            raise ValueError(
                f"لم يتم العثور على الفريق الضيف: '{team2_name}' "
                f"في مسابقة '{comp_code}'."
            )

        if home_id == away_id:
            raise ValueError(
                f"لا يمكن التنبؤ بمباراة بين نفس الفريق "
                f"(ID: {home_id})."
            )

        # --- الخطوة 3: اختيار الموسم ---
        season_key = self._select_season_key(comp_code, preferred_season_year)

        log(f"الخطوة 3: الموسم المُختار — {season_key}", "DEBUG")

        # --- الخطوة 4: استرجاع بيانات النماذج ---
        elo_home, elo_away = self._get_elo_ratings(home_id, away_id, season_key)
        rho = self._get_rho(season_key)

        log(
            f"الخطوة 4: بيانات النماذج — "
            f"ELO({home_id})={elo_home:.0f}, ELO({away_id})={elo_away:.0f}, "
            f"rho={rho:.4f}",
            "DEBUG"
        )

        # --- الخطوة 5: حساب λ الأساسي ---
        lam_home, lam_away, compute_details = self._compute_base_lambdas(
            home_id, away_id, season_key
        )

        log(
            f"الخطوة 5: λ الأساسي — "
            f"المضيف: {lam_home:.3f}, الضيف: {lam_away:.3f}",
            "DEBUG"
        )

        # --- الخطوة 6: تعديل ELO (اختياري) ---
        elo_adjusted = False
        if use_elo:
            lam_home_original = lam_home
            lam_away_original = lam_away

            lam_home, lam_away = self._adjust_lambdas_with_elo(
                lam_home, lam_away, elo_home, elo_away
            )

            elo_adjusted = True

            log(
                f"الخطوة 6: λ بعد تعديل ELO — "
                f"المضيف: {lam_home:.3f} (كان {lam_home_original:.3f}), "
                f"الضيف: {lam_away:.3f} (كان {lam_away_original:.3f})",
                "DEBUG"
            )
        else:
            log("الخطوة 6: تعديل ELO معطّل.", "DEBUG")

        # --- الخطوة 7: حساب مصفوفة الاحتمالات ---
        gmax = suggest_goal_cutoff(lam_home, lam_away)
        matrix = poisson_matrix_dc(lam_home, lam_away, rho, max_goals=gmax)
        p_home, p_draw, p_away = matrix_to_outcomes(matrix)

        # تطبيع وتحقق
        p_home = validate_probability(float(p_home))
        p_draw = validate_probability(float(p_draw))
        p_away = validate_probability(float(p_away))

        # تطبيع المجموع إلى 1.0
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total
        else:
            p_home = 1.0 / 3.0
            p_draw = 1.0 / 3.0
            p_away = 1.0 / 3.0

        log(
            f"الخطوة 7: الاحتمالات — "
            f"فوز المضيف: {p_home:.3f}, تعادل: {p_draw:.3f}, "
            f"فوز الضيف: {p_away:.3f}",
            "DEBUG"
        )

        # --- الخطوة 8: بناء النتيجة ---
        result: Dict[str, Any] = {
            "meta": {
                "version": getattr(config, "VERSION", "N/A"),
                "model_season_used": season_key,
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                "use_elo_adjust": use_elo,
            },
            "match": f"{team1_name} (Home) vs {team2_name} (Away)",
            "competition": comp_code,
            "teams_found": {
                "home": {
                    "name": team1_name,
                    "id": home_id,
                    "display_name": self.get_team_name(home_id),
                },
                "away": {
                    "name": team2_name,
                    "id": away_id,
                    "display_name": self.get_team_name(away_id),
                },
            },
            "model_inputs": {
                "lambda_home": round(lam_home, 4),
                "lambda_away": round(lam_away, 4),
                "rho": round(rho, 4),
                "gmax": int(gmax),
                "use_elo_adjust": use_elo,
                "elo_home": round(elo_home, 1),
                "elo_away": round(elo_away, 1),
                "elo_diff": round(elo_home - elo_away, 1),
                **{
                    k: v for k, v in compute_details.items()
                    if isinstance(v, (int, float, bool))
                },
            },
            "probabilities": {
                "home_win": round(float(p_home), 6),
                "draw": round(float(p_draw), 6),
                "away_win": round(float(p_away), 6),
            },
        }

        # --- الخطوة 9: أفضل النتائج المحتملة (اختياري) ---
        if topk and topk > 0:
            try:
                tops = top_scorelines(matrix, top_k=topk)
                result["top_scorelines"] = [
                    {
                        "home_goals": int(i),
                        "away_goals": int(j),
                        "prob": round(float(p), 6),
                    }
                    for i, j, p in tops
                ]
            except Exception as e:
                log(f"فشل حساب أفضل النتائج: {e}", "WARNING")
                result["top_scorelines"] = []

        log("الخطوة 9: النتيجة جاهزة.", "DEBUG")

        return result

    # =========================================================================
    # التنبؤ بالمعرّف مباشرة
    # =========================================================================

    def predict_by_id(
        self,
        home_team_id: int,
        away_team_id: int,
        comp_code: str,
        topk: int = 0,
        use_elo: bool = False,
        preferred_season_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        التنبؤ بنتيجة مباراة باستخدام معرّفات الفرق مباشرة.

        هذه الدالة تتجاوز عملية البحث بالاسم وتستخدم المعرّفات مباشرة.
        مفيدة عندما تكون المعرّفات معروفة مسبقاً.

        المعاملات:
            home_team_id: معرّف الفريق المضيف
            away_team_id: معرّف الفريق الضيف
            comp_code: رمز المسابقة
            topk: عدد أفضل النتائج المحتملة
            use_elo: تفعيل تعديل ELO
            preferred_season_year: سنة الموسم المفضّل

        العائد:
            قاموس النتيجة (نفس بنية predict())
        """
        # الحصول على أسماء الفرق للعرض
        home_name = self.get_team_name(home_team_id)
        away_name = self.get_team_name(away_team_id)

        # استخدام predict() مع المعرّفات كنص
        # (ستتخطى البحث بالاسم لأنها أعداد)
        return self.predict(
            team1_name=str(home_team_id),
            team2_name=str(away_team_id),
            comp_code=comp_code,
            topk=topk,
            use_elo=use_elo,
            preferred_season_year=preferred_season_year,
        )

    # =========================================================================
    # معلومات تشخيصية
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        الحصول على معلومات تشخيصية عن حالة المتنبئ.

        العائد:
            قاموس يحتوي على:
            - is_loaded: هل تم التحميل بنجاح
            - load_errors: أخطاء التحميل
            - models_status: حالة كل نموذج
            - teams_count: عدد الفرق
            - available_seasons: المواسم المتاحة
            - complete_seasons: المواسم الكاملة
        """
        models_status = {}
        for model_name in ("averages", "factors", "elo", "rho"):
            model_data = self.models.get(model_name, {})
            models_status[model_name] = {
                "loaded": bool(model_data),
                "season_count": (
                    len(model_data) if isinstance(model_data, dict) else 0
                ),
            }

        return {
            "is_loaded": self._loaded,
            "load_errors": self._load_errors.copy(),
            "models_status": models_status,
            "teams_count": len(self.teams_map),
            "available_seasons": self.available_seasons,
            "complete_seasons": self.complete_seasons,
            "config": {
                "version": getattr(config, "VERSION", "N/A"),
                "data_dir": str(config.DATA_DIR),
                "models_dir": str(config.MODELS_DIR),
                "elo_hfa": getattr(config, "ELO_HFA", "N/A"),
                "elo_lambda_scale": getattr(config, "ELO_LAMBDA_SCALE", "N/A"),
                "season_start_month": getattr(config, "CURRENT_SEASON_START_MONTH", "N/A"),
            },
        }

    def __repr__(self) -> str:
        """تمثيل نصي للمتنبئ."""
        status = "loaded" if self._loaded else "not loaded"
        seasons = len(self.complete_seasons)
        teams = len(self.teams_map)
        return (
            f"Predictor("
            f"status={status}, "
            f"complete_seasons={seasons}, "
            f"teams={teams}"
            f")"
        )
