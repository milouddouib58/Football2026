# 01_pipeline.py
# -----------------------------------------------------------------------------
# الوصف:
# السكريبت الرئيسي لمشروع التنبؤ. يقوم بتنفيذ عملية سحب البيانات (ETL)
# عبر الخطوات التالية:
# 1. جلب بيانات المباريات لآخر N مواسم للمسابقات المستهدفة.
# 2. جلب بيانات الفرق المشاركة في هذه المسابقات.
# 3. حفظ البيانات المجمعة في ملفات JSON محلية (`matches.json`, `teams.json`).
#
# التحسينات:
# - دعم التحديث التراكمي (incremental update) لتجنّب إعادة جلب كل شيء
# - التحقق من سلامة البيانات المُجلبة قبل الحفظ
# - إنشاء نسخة احتياطية تلقائية قبل الكتابة فوق الملفات القديمة
# - تقرير إحصائي مفصّل بعد انتهاء العملية
# - استخدام config.CURRENT_SEASON_START_MONTH بدلاً من قيمة مكتوبة يدوياً
# - معالجة أخطاء أكثر تفصيلاً وتسجيل أوضح
# - دعم وضع التشغيل الجاف (dry-run) للاختبار بدون كتابة ملفات
# - حفظ ملف تقرير العملية (pipeline_report.json) لتتبّع التاريخ
#
# الاستخدام:
# python 01_pipeline.py --years 3
# python 01_pipeline.py --years 5 --incremental
# python 01_pipeline.py --years 1 --dry-run
# -----------------------------------------------------------------------------

import sys
import os
import json
import shutil
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# استيراد الوحدات المشتركة
from common import config
from common.api_client import APIClient
from common.utils import log


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# الحد الأدنى المقبول لعدد المباريات لكل مسابقة في الموسم الواحد
# (إذا جاء أقل من ذلك فهذا يشير لمشكلة محتملة)
MIN_MATCHES_PER_COMPETITION_SEASON = 10

# الحد الأقصى المعقول لعدد المباريات لكل مسابقة في الموسم الواحد
MAX_MATCHES_PER_COMPETITION_SEASON = 500

# الحد الأقصى لعدد المواسم المسموح بجلبها
MAX_YEARS_ALLOWED = 20

# اسم ملف النسخة الاحتياطية
BACKUP_SUFFIX = ".backup"

# اسم ملف تقرير العملية
PIPELINE_REPORT_FILENAME = "pipeline_report.json"


# -----------------------------------------------------------------------------
# دوال مساعدة عامة
# -----------------------------------------------------------------------------

def get_current_season_start_year() -> int:
    """
    يحسب سنة بداية الموسم الكروي الحالي.

    يستخدم config.CURRENT_SEASON_START_MONTH لتحديد شهر بداية الموسم.
    مثلاً: إذا كان شهر البداية هو 7 (يوليو)، وكنا في يونيو 2025،
    فالموسم الحالي يبدأ في 2024.

    العائد:
        سنة بداية الموسم الحالي (int)
    """
    now = datetime.now()
    season_start_month = getattr(config, "CURRENT_SEASON_START_MONTH", 7)

    if now.month >= season_start_month:
        return now.year
    else:
        return now.year - 1


def calculate_target_years(current_season_start: int, years_to_fetch: int) -> List[int]:
    """
    حساب قائمة سنوات المواسم المراد جلبها.

    المعاملات:
        current_season_start: سنة بداية الموسم الحالي
        years_to_fetch: عدد المواسم المطلوب جلبها

    العائد:
        قائمة سنوات بداية المواسم (من الأحدث إلى الأقدم)
    """
    target_years = list(
        range(
            current_season_start,
            current_season_start - years_to_fetch,
            -1
        )
    )
    return target_years


def create_backup(file_path: Path) -> Optional[Path]:
    """
    إنشاء نسخة احتياطية من ملف موجود قبل الكتابة فوقه.

    المعاملات:
        file_path: مسار الملف الأصلي

    العائد:
        مسار ملف النسخة الاحتياطية، أو None إذا لم يكن الملف موجوداً
    """
    if not file_path.exists():
        return None

    # إنشاء اسم ملف النسخة الاحتياطية مع طابع زمني
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{BACKUP_SUFFIX}{file_path.suffix}"
    backup_path = file_path.parent / backup_name

    try:
        shutil.copy2(file_path, backup_path)
        log(f"تم إنشاء نسخة احتياطية: {backup_path}", "INFO")
        return backup_path
    except OSError as e:
        log(f"فشل إنشاء النسخة الاحتياطية لـ {file_path.name}: {e}", "WARNING")
        return None


def cleanup_old_backups(directory: Path, base_name: str, keep_last: int = 3):
    """
    حذف النسخ الاحتياطية القديمة والإبقاء على آخر N نسخة فقط.

    المعاملات:
        directory: مجلد النسخ الاحتياطية
        base_name: الاسم الأساسي للملف (بدون امتداد)
        keep_last: عدد النسخ الاحتياطية التي يتم الاحتفاظ بها
    """
    try:
        # البحث عن جميع النسخ الاحتياطية لهذا الملف
        pattern = f"{base_name}_*{BACKUP_SUFFIX}.*"
        backups = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)

        # حذف النسخ الزائدة (الأقدم)
        if len(backups) > keep_last:
            to_remove = backups[:len(backups) - keep_last]
            for old_backup in to_remove:
                try:
                    old_backup.unlink()
                    log(f"تم حذف نسخة احتياطية قديمة: {old_backup.name}", "DEBUG")
                except OSError:
                    pass
    except Exception:
        pass


def load_existing_matches(matches_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    تحميل المباريات الموجودة مسبقاً من ملف matches.json.
    تُستخدم في وضع التحديث التراكمي.

    المعاملات:
        matches_path: مسار ملف المباريات

    العائد:
        قاموس بمعرّف المباراة كمفتاح وبيانات المباراة كقيمة
    """
    if not matches_path.exists():
        log("لا يوجد ملف مباريات سابق. سيتم إنشاء ملف جديد.", "INFO")
        return {}

    try:
        with open(matches_path, "r", encoding="utf-8") as f:
            existing_list = json.load(f)

        if not isinstance(existing_list, list):
            log(
                f"تنسيق ملف المباريات غير متوقع (ليس قائمة). "
                f"سيتم تجاهل البيانات القديمة.",
                "WARNING"
            )
            return {}

        existing_dict = {}
        for match in existing_list:
            if isinstance(match, dict) and "id" in match:
                existing_dict[match["id"]] = match

        log(f"تم تحميل {len(existing_dict)} مباراة موجودة مسبقاً.", "INFO")
        return existing_dict

    except json.JSONDecodeError as e:
        log(f"خطأ في تحليل ملف المباريات الموجود: {e}. سيتم تجاهله.", "WARNING")
        return {}
    except Exception as e:
        log(f"خطأ أثناء تحميل المباريات الموجودة: {e}", "WARNING")
        return {}


def load_existing_teams(teams_path: Path) -> Dict[str, Any]:
    """
    تحميل بيانات الفرق الموجودة مسبقاً من ملف teams.json.
    تُستخدم في وضع التحديث التراكمي.

    المعاملات:
        teams_path: مسار ملف الفرق

    العائد:
        قاموس بيانات الفرق
    """
    if not teams_path.exists():
        log("لا يوجد ملف فرق سابق. سيتم إنشاء ملف جديد.", "INFO")
        return {}

    try:
        with open(teams_path, "r", encoding="utf-8") as f:
            existing_teams = json.load(f)

        if not isinstance(existing_teams, dict):
            log(
                f"تنسيق ملف الفرق غير متوقع (ليس قاموساً). "
                f"سيتم تجاهل البيانات القديمة.",
                "WARNING"
            )
            return {}

        log(f"تم تحميل {len(existing_teams)} فريق موجود مسبقاً.", "INFO")
        return existing_teams

    except json.JSONDecodeError as e:
        log(f"خطأ في تحليل ملف الفرق الموجود: {e}. سيتم تجاهله.", "WARNING")
        return {}
    except Exception as e:
        log(f"خطأ أثناء تحميل الفرق الموجودة: {e}", "WARNING")
        return {}


def validate_match(match: Any) -> bool:
    """
    التحقق من صحة بيانات مباراة واحدة.

    المعاملات:
        match: بيانات المباراة (يجب أن تكون قاموساً)

    العائد:
        True إذا كانت المباراة صالحة، False خلاف ذلك
    """
    # يجب أن تكون قاموساً
    if not isinstance(match, dict):
        return False

    # يجب أن تحتوي على معرّف
    if "id" not in match:
        return False

    # يجب أن تحتوي على بيانات الفريقين
    home_team = match.get("homeTeam")
    away_team = match.get("awayTeam")

    if not isinstance(home_team, dict) or not isinstance(away_team, dict):
        return False

    # يجب أن يكون لكل فريق معرّف
    if "id" not in home_team or "id" not in away_team:
        return False

    return True


def validate_matches_batch(matches: List[Dict], comp_code: str, year: int) -> Tuple[List[Dict], int]:
    """
    التحقق من صحة دفعة من المباريات وتصفية غير الصالحة.

    المعاملات:
        matches: قائمة المباريات المجلوبة
        comp_code: رمز المسابقة (للتسجيل)
        year: سنة الموسم (للتسجيل)

    العائد:
        tuple يحتوي على (قائمة المباريات الصالحة, عدد المباريات المرفوضة)
    """
    valid = []
    rejected = 0

    for match in matches:
        if validate_match(match):
            valid.append(match)
        else:
            rejected += 1

    if rejected > 0:
        log(
            f"  ⚠ تم رفض {rejected} مباراة غير صالحة من {comp_code} (موسم {year})",
            "WARNING"
        )

    # تحذير إذا كان العدد غير معقول
    if len(valid) > 0:
        if len(valid) < MIN_MATCHES_PER_COMPETITION_SEASON:
            log(
                f"  ⚠ عدد المباريات ({len(valid)}) أقل من المتوقع "
                f"لـ {comp_code} (موسم {year}). قد تكون البيانات ناقصة.",
                "WARNING"
            )
        elif len(valid) > MAX_MATCHES_PER_COMPETITION_SEASON:
            log(
                f"  ⚠ عدد المباريات ({len(valid)}) أكبر من المتوقع "
                f"لـ {comp_code} (موسم {year}). تحقق من البيانات.",
                "WARNING"
            )

    return valid, rejected


def save_json_safely(data: Any, file_path: Path, description: str) -> bool:
    """
    حفظ بيانات JSON بأمان مع كتابة مؤقتة ثم إعادة تسمية.
    هذا يمنع تلف الملف في حالة حدوث خطأ أثناء الكتابة.

    المعاملات:
        data: البيانات المراد حفظها
        file_path: مسار الملف النهائي
        description: وصف الملف (للتسجيل)

    العائد:
        True إذا تم الحفظ بنجاح، False خلاف ذلك
    """
    # إنشاء ملف مؤقت بنفس المجلد
    temp_path = file_path.with_suffix(".tmp")

    try:
        # الكتابة في الملف المؤقت أولاً
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

        # التحقق من صحة الملف المؤقت بقراءته
        with open(temp_path, "r", encoding="utf-8") as f:
            verification = json.load(f)

        # التحقق من تطابق الحجم
        if isinstance(data, list) and isinstance(verification, list):
            if len(data) != len(verification):
                log(
                    f"⚠ تحذير: حجم البيانات المحفوظة ({len(verification)}) "
                    f"لا يطابق الأصل ({len(data)}) لـ {description}.",
                    "WARNING"
                )
        elif isinstance(data, dict) and isinstance(verification, dict):
            if len(data) != len(verification):
                log(
                    f"⚠ تحذير: حجم البيانات المحفوظة ({len(verification)}) "
                    f"لا يطابق الأصل ({len(data)}) لـ {description}.",
                    "WARNING"
                )

        # نقل الملف المؤقت ليحلّ محل الملف النهائي
        temp_path.replace(file_path)

        # حساب حجم الملف
        file_size = file_path.stat().st_size
        size_str = format_file_size(file_size)

        log(f"✅ تم حفظ {description} بنجاح في: {file_path} ({size_str})", "INFO")
        return True

    except json.JSONDecodeError as e:
        log(f"❌ خطأ في التحقق من الملف المحفوظ لـ {description}: {e}", "ERROR")
        # حذف الملف المؤقت التالف
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False

    except IOError as e:
        log(f"❌ فشل في كتابة ملف {description}: {e}", "ERROR")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False

    except Exception as e:
        log(f"❌ خطأ غير متوقع أثناء حفظ {description}: {e}", "ERROR")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def format_file_size(size_bytes: int) -> str:
    """
    تحويل حجم الملف من بايت إلى تنسيق قابل للقراءة.

    المعاملات:
        size_bytes: الحجم بالبايت

    العائد:
        نص يمثّل الحجم (مثلاً "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def collect_unique_team_ids(all_matches: Dict[int, Dict]) -> Set[int]:
    """
    استخراج جميع معرّفات الفرق الفريدة من بيانات المباريات.

    المعاملات:
        all_matches: قاموس جميع المباريات

    العائد:
        مجموعة معرّفات الفرق الفريدة
    """
    team_ids = set()

    for match_id, match in all_matches.items():
        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})

        home_id = home_team.get("id")
        away_id = away_team.get("id")

        if home_id is not None:
            team_ids.add(int(home_id))
        if away_id is not None:
            team_ids.add(int(away_id))

    return team_ids


def collect_unique_competition_codes(all_matches: Dict[int, Dict]) -> Set[str]:
    """
    استخراج جميع رموز المسابقات الفريدة من بيانات المباريات.

    المعاملات:
        all_matches: قاموس جميع المباريات

    العائد:
        مجموعة رموز المسابقات الفريدة
    """
    comp_codes = set()

    for match_id, match in all_matches.items():
        competition = match.get("competition", {})
        code = competition.get("code")
        if code:
            comp_codes.add(str(code))

    return comp_codes


def generate_pipeline_report(
    start_time: datetime,
    end_time: datetime,
    years_to_fetch: int,
    target_years: List[int],
    target_competitions: Dict[str, int],
    all_matches: Dict[int, Dict],
    teams_data: Dict,
    fetch_stats: Dict,
    incremental: bool,
    matches_saved: bool,
    teams_saved: bool,
) -> Dict:
    """
    إنشاء تقرير مفصّل عن عملية سحب البيانات.

    المعاملات:
        start_time: وقت بدء العملية
        end_time: وقت انتهاء العملية
        years_to_fetch: عدد المواسم المطلوبة
        target_years: قائمة سنوات المواسم
        target_competitions: المسابقات المستهدفة
        all_matches: جميع المباريات المُجمّعة
        teams_data: بيانات الفرق
        fetch_stats: إحصائيات الجلب لكل مسابقة/موسم
        incremental: هل تم استخدام التحديث التراكمي
        matches_saved: هل تم حفظ المباريات بنجاح
        teams_saved: هل تم حفظ الفرق بنجاح

    العائد:
        قاموس التقرير
    """
    duration_seconds = (end_time - start_time).total_seconds()

    # حساب إحصائيات المباريات حسب المسابقة
    matches_by_competition = {}
    for match_id, match in all_matches.items():
        comp = match.get("competition", {})
        comp_code = comp.get("code", "UNKNOWN")
        if comp_code not in matches_by_competition:
            matches_by_competition[comp_code] = 0
        matches_by_competition[comp_code] += 1

    # حساب إحصائيات المباريات حسب الحالة
    matches_by_status = {}
    for match_id, match in all_matches.items():
        status = match.get("status", "UNKNOWN")
        if status not in matches_by_status:
            matches_by_status[status] = 0
        matches_by_status[status] += 1

    # جمع معرّفات الفرق الفريدة
    unique_team_ids = collect_unique_team_ids(all_matches)

    report = {
        "pipeline_version": getattr(config, "VERSION", "N/A"),
        "run_timestamp": end_time.isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "duration_readable": format_duration(duration_seconds),
        "parameters": {
            "years_to_fetch": years_to_fetch,
            "target_years": target_years,
            "incremental_mode": incremental,
            "target_competitions": list(target_competitions.keys()),
        },
        "results": {
            "total_matches": len(all_matches),
            "total_teams": len(teams_data) if teams_data else 0,
            "unique_team_ids_in_matches": len(unique_team_ids),
            "matches_by_competition": matches_by_competition,
            "matches_by_status": matches_by_status,
            "matches_saved_successfully": matches_saved,
            "teams_saved_successfully": teams_saved,
        },
        "fetch_details": fetch_stats,
    }

    return report


def format_duration(seconds: float) -> str:
    """
    تحويل المدة من ثوانٍ إلى تنسيق قابل للقراءة.

    المعاملات:
        seconds: المدة بالثواني

    العائد:
        نص يمثّل المدة (مثلاً "2 دقيقة و 30 ثانية")
    """
    if seconds < 60:
        return f"{seconds:.1f} ثانية"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes} دقيقة و {secs} ثانية"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} ساعة و {minutes} دقيقة"


def save_pipeline_report(report: Dict, report_path: Path) -> bool:
    """
    حفظ تقرير عملية السحب في ملف JSON.

    المعاملات:
        report: قاموس التقرير
        report_path: مسار ملف التقرير

    العائد:
        True إذا تم الحفظ بنجاح، False خلاف ذلك
    """
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        log(f"تم حفظ تقرير العملية في: {report_path}", "INFO")
        return True
    except Exception as e:
        log(f"فشل حفظ تقرير العملية: {e}", "WARNING")
        return False


def print_summary(
    all_matches: Dict[int, Dict],
    teams_data: Dict,
    target_competitions: Dict[str, int],
    target_years: List[int],
    fetch_stats: Dict,
    duration_seconds: float,
    incremental: bool,
    new_matches_count: int,
    matches_saved: bool,
    teams_saved: bool,
):
    """
    طباعة ملخص مفصّل لعملية سحب البيانات.

    المعاملات:
        all_matches: جميع المباريات المُجمّعة
        teams_data: بيانات الفرق
        target_competitions: المسابقات المستهدفة
        target_years: سنوات المواسم المستهدفة
        fetch_stats: إحصائيات الجلب
        duration_seconds: مدة العملية بالثواني
        incremental: هل تم استخدام التحديث التراكمي
        new_matches_count: عدد المباريات الجديدة (في الوضع التراكمي)
        matches_saved: هل تم حفظ المباريات بنجاح
        teams_saved: هل تم حفظ الفرق بنجاح
    """
    print("")
    log("=" * 70, "INFO")
    log("ملخص عملية سحب البيانات", "INFO")
    log("=" * 70, "INFO")

    # معلومات عامة
    log(f"  الوضع: {'تحديث تراكمي' if incremental else 'جلب كامل'}", "INFO")
    log(f"  المدة: {format_duration(duration_seconds)}", "INFO")
    log(f"  المسابقات المستهدفة: {list(target_competitions.keys())}", "INFO")
    log(f"  المواسم المستهدفة: {target_years}", "INFO")

    # إحصائيات المباريات
    log("", "INFO")
    log("  --- المباريات ---", "INFO")
    log(f"  إجمالي المباريات: {len(all_matches)}", "INFO")

    if incremental:
        log(f"  المباريات الجديدة: {new_matches_count}", "INFO")

    # تفصيل حسب المسابقة
    matches_by_comp = {}
    for match in all_matches.values():
        comp_code = match.get("competition", {}).get("code", "UNKNOWN")
        matches_by_comp[comp_code] = matches_by_comp.get(comp_code, 0) + 1

    for comp_code in sorted(matches_by_comp.keys()):
        count = matches_by_comp[comp_code]
        log(f"    {comp_code}: {count} مباراة", "INFO")

    # تفصيل الجلب حسب المسابقة والموسم
    log("", "INFO")
    log("  --- تفاصيل الجلب ---", "INFO")
    for key, stats in fetch_stats.items():
        fetched = stats.get("fetched", 0)
        valid = stats.get("valid", 0)
        rejected = stats.get("rejected", 0)
        status = stats.get("status", "unknown")

        status_icon = "✅" if status == "success" else ("⚠" if status == "empty" else "❌")
        detail = f"    {status_icon} {key}: {fetched} مجلوبة, {valid} صالحة"
        if rejected > 0:
            detail += f", {rejected} مرفوضة"
        log(detail, "INFO")

    # إحصائيات الفرق
    log("", "INFO")
    log("  --- الفرق ---", "INFO")
    teams_count = len(teams_data) if teams_data else 0
    unique_team_ids = collect_unique_team_ids(all_matches)
    log(f"  إجمالي الفرق في ملف teams.json: {teams_count}", "INFO")
    log(f"  فرق فريدة في المباريات: {len(unique_team_ids)}", "INFO")

    # حالة الحفظ
    log("", "INFO")
    log("  --- حالة الحفظ ---", "INFO")
    log(
        f"  matches.json: {'✅ تم الحفظ' if matches_saved else '❌ فشل الحفظ'}",
        "INFO"
    )
    log(
        f"  teams.json: {'✅ تم الحفظ' if teams_saved else '❌ فشل الحفظ'}",
        "INFO"
    )

    log("=" * 70, "INFO")
    print("")


# -----------------------------------------------------------------------------
# الدالة الرئيسية
# -----------------------------------------------------------------------------

def run_pipeline(
    years_to_fetch: int,
    incremental: bool = False,
    dry_run: bool = False,
    create_backups: bool = True,
):
    """
    الدالة الرئيسية لتشغيل عملية سحب وتنظيم البيانات.

    المعاملات:
        years_to_fetch: عدد المواسم السابقة التي سيتم جلب بياناتها.
        incremental: إذا True، يتم دمج البيانات الجديدة مع الموجودة بدلاً من استبدالها.
        dry_run: إذا True، يتم تنفيذ العملية بدون كتابة ملفات (للاختبار).
        create_backups: إذا True، يتم إنشاء نسخ احتياطية قبل الكتابة.
    """
    # تسجيل وقت البدء
    start_time = datetime.now(timezone.utc)

    # =========================================================================
    # 0. التحقق من المدخلات
    # =========================================================================
    log("=" * 70, "INFO")
    log("بدء عملية سحب البيانات (Pipeline)", "INFO")
    log(f"الوقت: {start_time.isoformat()}", "INFO")
    log("=" * 70, "INFO")

    if dry_run:
        log("⚠ وضع التشغيل الجاف (Dry Run): لن يتم كتابة أي ملفات.", "WARNING")

    if years_to_fetch < 1:
        log("قيمة years يجب أن تكون >= 1.", "ERROR")
        return

    if years_to_fetch > MAX_YEARS_ALLOWED:
        log(
            f"قيمة years ({years_to_fetch}) تتجاوز الحد الأقصى ({MAX_YEARS_ALLOWED}). "
            f"سيتم تقليصها.",
            "WARNING"
        )
        years_to_fetch = MAX_YEARS_ALLOWED

    log(f"عدد المواسم المطلوبة: {years_to_fetch}", "INFO")
    log(f"الوضع: {'تحديث تراكمي' if incremental else 'جلب كامل'}", "INFO")

    # =========================================================================
    # 1. الإعداد المبدئي
    # =========================================================================
    log("--- المرحلة 1: الإعداد المبدئي ---", "INFO")

    # إنشاء عميل API
    try:
        client = APIClient()
        log("تم إنشاء عميل API بنجاح.", "INFO")
    except Exception as e:
        log(f"فشل في إنشاء عميل API: {e}", "CRITICAL")
        return

    # تهيئة قاموس المباريات
    all_matches: Dict[int, Dict[str, Any]] = {}

    # إحصائيات الجلب
    fetch_stats: Dict[str, Dict] = {}

    # عداد المباريات الجديدة (للوضع التراكمي)
    new_matches_count = 0
    existing_matches_count = 0

    # =========================================================================
    # 2. تحميل البيانات الموجودة (في الوضع التراكمي)
    # =========================================================================
    if incremental:
        log("--- المرحلة 2: تحميل البيانات الموجودة (وضع تراكمي) ---", "INFO")
        matches_path = config.DATA_DIR / "matches.json"
        all_matches = load_existing_matches(matches_path)
        existing_matches_count = len(all_matches)
    else:
        log("--- المرحلة 2: تخطي (وضع جلب كامل) ---", "INFO")

    # =========================================================================
    # 3. جلب قائمة المسابقات المستهدفة
    # =========================================================================
    log("--- المرحلة 3: جلب المسابقات المستهدفة ---", "INFO")

    try:
        target_competitions = client.get_competitions()
    except Exception as e:
        log(f"خطأ أثناء جلب المسابقات: {e}", "CRITICAL")
        return

    if not target_competitions:
        log(
            "لم يتم العثور على المسابقات المستهدفة. "
            "تحقق من إعدادات config.TARGET_COMPETITIONS ومفتاح API.",
            "CRITICAL"
        )
        return

    log(f"المسابقات المستهدفة ({len(target_competitions)}):", "INFO")
    for code, comp_id in target_competitions.items():
        log(f"  {code} (ID: {comp_id})", "INFO")

    # =========================================================================
    # 4. تحديد سنوات المواسم المراد جلبها
    # =========================================================================
    log("--- المرحلة 4: تحديد المواسم ---", "INFO")

    current_season_start = get_current_season_start_year()
    target_years = calculate_target_years(current_season_start, years_to_fetch)

    log(f"الموسم الحالي يبدأ في: {current_season_start}", "INFO")
    log(f"المواسم المستهدفة (حسب سنة البداية): {target_years}", "INFO")

    # =========================================================================
    # 5. جلب بيانات المباريات لكل مسابقة وموسم
    # =========================================================================
    log("--- المرحلة 5: جلب بيانات المباريات ---", "INFO")

    total_fetched = 0
    total_valid = 0
    total_rejected = 0
    total_new = 0

    for comp_code, comp_id in target_competitions.items():
        log(f"", "INFO")
        log(f"📋 المسابقة: {comp_code} (ID: {comp_id})", "INFO")

        for year in target_years:
            fetch_key = f"{comp_code}_{year}"
            log(f"  📅 جارٍ جلب مباريات موسم {year}...", "INFO")

            try:
                matches_in_year = client.get_matches_for_season(year, comp_id)
            except Exception as e:
                log(f"  ❌ خطأ أثناء جلب مباريات {comp_code} (موسم {year}): {e}", "ERROR")
                fetch_stats[fetch_key] = {
                    "status": "error",
                    "error": str(e),
                    "fetched": 0,
                    "valid": 0,
                    "rejected": 0,
                    "new": 0,
                }
                continue

            if not matches_in_year:
                log(f"  ⚠ لم يتم العثور على مباريات منتهية لـ {comp_code} في موسم {year}.", "WARNING")
                fetch_stats[fetch_key] = {
                    "status": "empty",
                    "fetched": 0,
                    "valid": 0,
                    "rejected": 0,
                    "new": 0,
                }
                continue

            # التحقق من صحة المباريات
            fetched_count = len(matches_in_year)
            valid_matches, rejected_count = validate_matches_batch(
                matches_in_year, comp_code, year
            )

            # إضافة المباريات الصالحة إلى القاموس الشامل
            new_in_batch = 0
            for match in valid_matches:
                match_id = match["id"]

                # في الوضع التراكمي، نحسب المباريات الجديدة فقط
                if match_id not in all_matches:
                    new_in_batch += 1

                # الكتابة فوق أي بيانات قديمة بالأحدث (أو إضافة جديدة)
                all_matches[match_id] = match

            total_fetched += fetched_count
            total_valid += len(valid_matches)
            total_rejected += rejected_count
            total_new += new_in_batch

            log(
                f"  ✅ {comp_code} (موسم {year}): "
                f"{fetched_count} مجلوبة, {len(valid_matches)} صالحة"
                + (f", {new_in_batch} جديدة" if incremental else ""),
                "INFO"
            )

            fetch_stats[fetch_key] = {
                "status": "success",
                "fetched": fetched_count,
                "valid": len(valid_matches),
                "rejected": rejected_count,
                "new": new_in_batch,
            }

    new_matches_count = total_new

    log("", "INFO")
    log(
        f"إجمالي المباريات: {total_fetched} مجلوبة, "
        f"{total_valid} صالحة, {total_rejected} مرفوضة",
        "INFO"
    )
    log(f"إجمالي المباريات الفريدة في القاموس: {len(all_matches)}", "INFO")

    if incremental:
        log(
            f"المباريات الموجودة مسبقاً: {existing_matches_count} | "
            f"المباريات الجديدة: {new_matches_count}",
            "INFO"
        )

    # =========================================================================
    # 6. التحقق من وجود بيانات كافية
    # =========================================================================
    log("--- المرحلة 6: التحقق من البيانات ---", "INFO")

    if not all_matches:
        log(
            "خطأ فادح: لم يتم تجميع أي بيانات للمباريات. "
            "يرجى التحقق من مفتاح API وصلاحيات الخطة والاتصال بالإنترنت.",
            "CRITICAL"
        )
        return

    log(f"تم التحقق: {len(all_matches)} مباراة جاهزة للحفظ.", "INFO")

    # التحقق من تنوع المسابقات
    comp_codes_in_data = collect_unique_competition_codes(all_matches)
    log(f"المسابقات الموجودة في البيانات: {sorted(comp_codes_in_data)}", "INFO")

    missing_comps = set(target_competitions.keys()) - comp_codes_in_data
    if missing_comps:
        log(
            f"⚠ المسابقات التالية مفقودة من البيانات: {sorted(missing_comps)}",
            "WARNING"
        )

    # =========================================================================
    # 7. إنشاء مجلد البيانات
    # =========================================================================
    if not dry_run:
        log("--- المرحلة 7: إعداد مجلد البيانات ---", "INFO")

        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            log(f"مجلد البيانات جاهز: {config.DATA_DIR}", "INFO")
        except OSError as e:
            log(f"فشل في إنشاء مجلد البيانات {config.DATA_DIR}: {e}", "CRITICAL")
            return
    else:
        log("--- المرحلة 7: تخطي إنشاء المجلد (وضع تجريبي) ---", "INFO")

    # =========================================================================
    # 8. حفظ بيانات المباريات
    # =========================================================================
    log("--- المرحلة 8: حفظ بيانات المباريات ---", "INFO")

    matches_path = config.DATA_DIR / "matches.json"
    matches_saved = False

    if dry_run:
        log(f"[DRY RUN] سيتم حفظ {len(all_matches)} مباراة في {matches_path}", "INFO")
        matches_saved = True
    else:
        # إنشاء نسخة احتياطية
        if create_backups:
            create_backup(matches_path)
            cleanup_old_backups(config.DATA_DIR, "matches", keep_last=3)

        # تحويل القاموس إلى قائمة وحفظ
        matches_list = list(all_matches.values())
        matches_saved = save_json_safely(matches_list, matches_path, "بيانات المباريات")

    # =========================================================================
    # 9. جلب وحفظ بيانات الفرق
    # =========================================================================
    log("--- المرحلة 9: جلب وحفظ بيانات الفرق ---", "INFO")

    teams_data: Dict = {}
    teams_saved = False

    log("جارٍ جلب بيانات الفرق لجميع المسابقات المستهدفة...", "INFO")

    try:
        competition_ids = list(target_competitions.values())
        teams_data_fetched = client.get_teams_for_competitions(competition_ids)
    except Exception as e:
        log(f"خطأ أثناء جلب بيانات الفرق: {e}", "ERROR")
        teams_data_fetched = None

    if teams_data_fetched:
        # في الوضع التراكمي، دمج الفرق الجديدة مع الموجودة
        if incremental:
            teams_path = config.DATA_DIR / "teams.json"
            existing_teams = load_existing_teams(teams_path)

            # دمج: الفرق الجديدة تكتب فوق القديمة (تحديث)
            existing_teams.update(teams_data_fetched)
            teams_data = existing_teams
            log(
                f"تم دمج بيانات الفرق: {len(teams_data_fetched)} مجلوبة + "
                f"{len(existing_teams)} موجودة = {len(teams_data)} إجمالي",
                "INFO"
            )
        else:
            teams_data = teams_data_fetched

        log(f"إجمالي الفرق: {len(teams_data)}", "INFO")

        if dry_run:
            log(f"[DRY RUN] سيتم حفظ {len(teams_data)} فريق", "INFO")
            teams_saved = True
        else:
            teams_path = config.DATA_DIR / "teams.json"

            # إنشاء نسخة احتياطية
            if create_backups:
                create_backup(teams_path)
                cleanup_old_backups(config.DATA_DIR, "teams", keep_last=3)

            teams_saved = save_json_safely(teams_data, teams_path, "بيانات الفرق")
    else:
        log("⚠ لم يتم العثور على بيانات للفرق.", "WARNING")

    # =========================================================================
    # 10. حفظ تقرير العملية
    # =========================================================================
    log("--- المرحلة 10: حفظ تقرير العملية ---", "INFO")

    end_time = datetime.now(timezone.utc)
    duration_seconds = (end_time - start_time).total_seconds()

    report = generate_pipeline_report(
        start_time=start_time,
        end_time=end_time,
        years_to_fetch=years_to_fetch,
        target_years=target_years,
        target_competitions=target_competitions,
        all_matches=all_matches,
        teams_data=teams_data,
        fetch_stats=fetch_stats,
        incremental=incremental,
        matches_saved=matches_saved,
        teams_saved=teams_saved,
    )

    if not dry_run:
        report_path = config.DATA_DIR / PIPELINE_REPORT_FILENAME
        save_pipeline_report(report, report_path)

    # =========================================================================
    # 11. طباعة الملخص
    # =========================================================================
    print_summary(
        all_matches=all_matches,
        teams_data=teams_data,
        target_competitions=target_competitions,
        target_years=target_years,
        fetch_stats=fetch_stats,
        duration_seconds=duration_seconds,
        incremental=incremental,
        new_matches_count=new_matches_count,
        matches_saved=matches_saved,
        teams_saved=teams_saved,
    )

    # =========================================================================
    # 12. التحقق النهائي
    # =========================================================================
    if matches_saved and teams_saved:
        log("✅ انتهت عملية سحب البيانات بنجاح.", "INFO")
    elif matches_saved and not teams_saved:
        log(
            "⚠ تم حفظ المباريات بنجاح لكن فشل حفظ الفرق. "
            "يمكن إعادة تشغيل العملية.",
            "WARNING"
        )
    elif not matches_saved and teams_saved:
        log(
            "⚠ فشل حفظ المباريات لكن تم حفظ الفرق. "
            "يرجى إعادة تشغيل العملية.",
            "ERROR"
        )
    else:
        if not dry_run:
            log("❌ فشل حفظ كل من المباريات والفرق.", "ERROR")


# -----------------------------------------------------------------------------
# نقطة الدخول
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="بناء مجموعة بيانات محلية من موقع football-data.org.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  python 01_pipeline.py --years 3              # جلب آخر 3 مواسم (كتابة كاملة)
  python 01_pipeline.py --years 5 --incremental  # تحديث تراكمي لآخر 5 مواسم
  python 01_pipeline.py --years 1 --dry-run      # تشغيل تجريبي بدون كتابة
  python 01_pipeline.py --years 3 --no-backup    # بدون نسخ احتياطية
        """
    )

    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="عدد المواسم السابقة (حسب سنة البداية) لجلب بياناتها لكل مسابقة. (افتراضي: 3)"
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help="تحديث تراكمي: دمج البيانات الجديدة مع الموجودة بدلاً من استبدالها."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="تشغيل تجريبي: تنفيذ العملية بدون كتابة أي ملفات."
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="تعطيل إنشاء النسخ الاحتياطية قبل الكتابة فوق الملفات."
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            years_to_fetch=args.years,
            incremental=args.incremental,
            dry_run=args.dry_run,
            create_backups=not args.no_backup,
        )
    except KeyboardInterrupt:
        log("", "INFO")
        log("تم إيقاف العملية بواسطة المستخدم (Ctrl+C).", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"خطأ غير متوقع: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)
