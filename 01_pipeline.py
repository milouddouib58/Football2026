# -----------------------------------------------------------------------------
# 01_pipeline.py
# -----------------------------------------------------------------------------
# الوصف:
#   السكريبت الرئيسي لمشروع التنبؤ. يقوم بتنفيذ عملية سحب البيانات (ETL)
#   عبر الخطوات التالية:
#   1. جلب بيانات المباريات لآخر 3 مواسم (أو حسب المحدد) للمسابقات المستهدفة.
#   2. جلب بيانات الفرق المشاركة في هذه المسابقات.
#   3. حفظ البيانات المجمعة في ملفات JSON محلية (`matches.json`, `teams.json`).
#
# الاستخدام:
#   python 01_pipeline.py --years 3
# -----------------------------------------------------------------------------

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# استيراد الوحدات المشتركة
from common.config import config
from common.api_client import APIClient
from common.utils import log


def _get_current_season_start_year() -> int:
    """
    يحسب سنة بداية الموسم الكروي الحالي.
    تعتبر أن المواسم الأوروبية تبدأ في شهر يوليو (7).

    مثال:
    - إذا كان التاريخ هو أكتوبر 2024، سيعيد 2024 (لموسم 2024/25).
    - إذا كان التاريخ هو مارس 2025، سيعيد 2024 (لموسم 2024/25).

    Returns:
        int: سنة بداية الموسم الحالي.
    """
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def run_pipeline(years_to_fetch: int):
    """
    الدالة الرئيسية لتشغيل عملية سحب وتنظيم البيانات.

    Args:
        years_to_fetch (int): عدد المواسم السابقة التي سيتم جلب بياناتها.
    """
    log("--- بدء عملية سحب البيانات (وضع المواسم المتعددة) ---", "INFO")
    log(f"محاولة جلب بيانات لآخر {years_to_fetch} مواسم.", "INFO")

    # --- 1. الإعداد المبدئي ---
    client = APIClient()
    all_matches: Dict[int, Dict[str, Any]] = {}

    # --- 2. جلب قائمة المسابقات المستهدفة ---
    target_competitions = client.get_competitions()
    if not target_competitions:
        log("لم يتم العثور على المسابقات المستهدفة. الخروج من البرنامج.", "CRITICAL")
        return

    # --- 3. تحديد سنوات المواسم المراد جلبها ---
    current_season_start = _get_current_season_start_year()
    target_years = list(range(current_season_start, current_season_start - years_to_fetch, -1))
    log(f"المواسم المستهدفة (حسب سنة البداية): {target_years}", "INFO")

    # --- 4. جلب بيانات المباريات لكل مسابقة وموسم ---
    for code, comp_id in target_competitions.items():
        for year in target_years:
            log(f"جاري جلب مباريات {code} لموسم {year}...", "INFO")
            matches_in_year = client.get_matches_for_season(year, comp_id)

            if matches_in_year:
                log(f"تم العثور على {len(matches_in_year)} مباراة لـ {code} (موسم {year}).", "INFO")
                # استخدام ID المباراة كمفتاح يضمن عدم وجود تكرار
                for match in matches_in_year:
                    all_matches[match['id']] = match
            else:
                log(f"لم يتم العثور على مباريات منتهية لـ {code} في موسم {year}.", "WARNING")

    log(f"إجمالي عدد المباريات الفريدة التي تم تجميعها: {len(all_matches)}", "INFO")

    # --- 5. التحقق من البيانات وحفظها ---
    if not all_matches:
        log("خطأ فادح: لم يتم تجميع أي بيانات للمباريات. يرجى التحقق من مفتاح API وصلاحيات الخطة.", "CRITICAL")
        return

    # التأكد من وجود مجلد البيانات وإنشائه عند الحاجة
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log(f"فشل في إنشاء مجلد البيانات {config.DATA_DIR}: {e}", "ERROR")
        return

    # حفظ بيانات المباريات
    matches_path = config.DATA_DIR / "matches.json"
    try:
        with open(matches_path, 'w', encoding='utf-8') as f:
            # تحويل قيم القاموس إلى قائمة للحفظ
            json.dump(list(all_matches.values()), f, ensure_ascii=False, indent=2)
        log(f"تم حفظ بيانات المباريات بنجاح في: {matches_path}", "INFO")
    except IOError as e:
        log(f"فشل في كتابة ملف المباريات: {e}", "ERROR")

    # --- 6. جلب وحفظ بيانات الفرق ---
    log("جاري جلب بيانات الفرق لجميع المسابقات المستهدفة...", "INFO")
    teams_data = client.get_teams_for_competitions(list(target_competitions.values()))
    
    if teams_data:
        teams_path = config.DATA_DIR / "teams.json"
        try:
            with open(teams_path, 'w', encoding='utf-8') as f:
                json.dump(teams_data, f, ensure_ascii=False, indent=2)
            log(f"تم حفظ بيانات الفرق بنجاح في: {teams_path}", "INFO")
        except IOError as e:
            log(f"فشل في كتابة ملف الفرق: {e}", "ERROR")
    else:
        log("لم يتم العثور على بيانات للفرق.", "WARNING")

    log("--- انتهت عملية سحب البيانات بنجاح ---", "INFO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="بناء مجموعة بيانات محلية من موقع football-data.org."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="عدد المواسم السابقة (حسب سنة البداية) لجلب بياناتها لكل مسابقة."
    )
    args = parser.parse_args()

    run_pipeline(years_to_fetch=args.years)

