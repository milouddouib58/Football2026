# 03_predict.py
# -----------------------------------------------------------------------------
# CLI للتنبؤ بمباراة باستخدام النماذج الإحصائية المدرّبة عبر كلاس Predictor.
#
# يوفّر واجهة سطر أوامر بسيطة للتنبؤ بنتيجة مباراة واحدة باستخدام
# نموذج Dixon-Coles مع إمكانية تعديل ELO.
#
# التحسينات:
# - دعم التنبؤ بالمعرّف (--home-id, --away-id) بالإضافة للاسم
# - دعم حفظ النتيجة في ملف JSON (--save, --output)
# - دعم تحديد الموسم يدوياً (--season)
# - عرض منسّق للنتائج في الطرفية
# - معالجة أخطاء تفصيلية مع رموز خروج مناسبة
# - دعم وضع JSON فقط (--json) لسهولة الدمج مع أدوات أخرى
# - دعم وضع التشخيص (--diagnostics) لعرض حالة النماذج
# - دعم وضع التشغيل الجاف (--dry-run)
# - عرض الميزات والمدخلات المستخدمة
#
# الاستخدام:
#   python 03_predict.py --team1 "Manchester City" --team2 "Liverpool" --comp PL
#   python 03_predict.py --team1 "Real Madrid" --team2 "Barcelona" --comp PD --use-elo --topk 5
#   python 03_predict.py --home-id 65 --away-id 64 --comp PL --use-elo
#   python 03_predict.py --team1 "Bayern" --team2 "Dortmund" --comp BL1 --save
#   python 03_predict.py --diagnostics
# -----------------------------------------------------------------------------

import sys
import os
import json
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# استيراد الوحدات المشتركة
from common import config
from common.utils import log
from predictor import Predictor


# -----------------------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------------------

# رموز الخروج
EXIT_SUCCESS = 0
EXIT_ERROR_ARGS = 1
EXIT_ERROR_LOAD = 2
EXIT_ERROR_PREDICT = 3
EXIT_ERROR_SAVE = 4
EXIT_ERROR_UNEXPECTED = 99


# -----------------------------------------------------------------------------
# دوال العرض
# -----------------------------------------------------------------------------

def display_prediction_formatted(
    result: Dict[str, Any],
    show_inputs: bool = False,
    show_scorelines: bool = True,
):
    """
    عرض نتائج التنبؤ بشكل منسّق في الطرفية.

    المعاملات:
        result: قاموس النتيجة من predictor.predict()
        show_inputs: عرض مدخلات النموذج
        show_scorelines: عرض أفضل النتائج المحتملة
    """
    # استخراج البيانات
    match_str = result.get("match", "Unknown Match")
    comp = result.get("competition", "N/A")
    meta = result.get("meta", {})
    season = meta.get("model_season_used", "N/A")
    version = meta.get("version", "N/A")

    probs = result.get("probabilities", {})
    p_home = float(probs.get("home_win", 0.0))
    p_draw = float(probs.get("draw", 0.0))
    p_away = float(probs.get("away_win", 0.0))

    teams = result.get("teams_found", {})
    home_info = teams.get("home", {})
    away_info = teams.get("away", {})

    home_name = home_info.get("display_name", home_info.get("name", "Home"))
    away_name = away_info.get("display_name", away_info.get("name", "Away"))
    home_id = home_info.get("id", "?")
    away_id = away_info.get("id", "?")

    # شريط مرئي
    def bar(prob: float, width: int = 30) -> str:
        filled = int(prob * width)
        return "█" * filled + "░" * (width - filled)

    # تحديد التنبؤ الأرجح
    max_prob = max(p_home, p_draw, p_away)
    if max_prob == p_home:
        prediction_text = f"فوز {home_name}"
        prediction_emoji = "🏠"
    elif max_prob == p_draw:
        prediction_text = "تعادل"
        prediction_emoji = "🤝"
    else:
        prediction_text = f"فوز {away_name}"
        prediction_emoji = "✈️"

    # طباعة النتائج
    print("")
    print("=" * 65)
    print(f"  ⚽ تنبؤ المباراة (Dixon-Coles)")
    print("=" * 65)
    print(f"  المباراة  : {home_name} vs {away_name}")
    print(f"  المعرّفات : {home_id} vs {away_id}")
    print(f"  المسابقة  : {comp}")
    print(f"  الموسم    : {season}")
    print(f"  الإصدار   : {version}")
    print("-" * 65)
    print(f"  فوز المضيف : {p_home:7.2%}  {bar(p_home)}")
    print(f"  تعادل      : {p_draw:7.2%}  {bar(p_draw)}")
    print(f"  فوز الضيف  : {p_away:7.2%}  {bar(p_away)}")
    print("-" * 65)
    print(f"  {prediction_emoji} التنبؤ: {prediction_text} (ثقة: {max_prob:.1%})")
    print("=" * 65)

    # عرض مدخلات النموذج
    if show_inputs:
        model_inputs = result.get("model_inputs", {})
        if model_inputs:
            print("")
            print("  📊 مدخلات النموذج:")
            print("  " + "-" * 45)
            for key, value in model_inputs.items():
                if isinstance(value, float):
                    print(f"    {key:25s}: {value:.4f}")
                elif isinstance(value, bool):
                    print(f"    {key:25s}: {'نعم' if value else 'لا'}")
                else:
                    print(f"    {key:25s}: {value}")
            print("  " + "-" * 45)

    # عرض أفضل النتائج المحتملة
    if show_scorelines:
        scorelines = result.get("top_scorelines", [])
        if scorelines:
            print("")
            print(f"  🎯 أفضل {len(scorelines)} نتائج محتملة:")
            print("  " + "-" * 35)
            for i, sl in enumerate(scorelines, 1):
                hg = sl.get("home_goals", 0)
                ag = sl.get("away_goals", 0)
                prob = float(sl.get("prob", 0.0))
                print(f"    {i}. {hg} - {ag}  ({prob:.2%})")
            print("  " + "-" * 35)

    print("")


def display_diagnostics(predictor: Predictor):
    """
    عرض معلومات تشخيصية عن حالة المتنبئ والنماذج.

    المعاملات:
        predictor: كائن المتنبئ
    """
    diag = predictor.get_diagnostics()

    print("")
    print("=" * 60)
    print("  🩺 تشخيص حالة المتنبئ")
    print("=" * 60)

    # حالة التحميل
    status = "✅ جاهز" if diag["is_loaded"] else "❌ غير جاهز"
    print(f"  الحالة: {status}")
    print(f"  عدد الفرق: {diag['teams_count']}")
    print("")

    # حالة النماذج
    print("  حالة النماذج:")
    print("  " + "-" * 40)
    for model_name, model_status in diag.get("models_status", {}).items():
        loaded = "✅" if model_status["loaded"] else "❌"
        count = model_status["season_count"]
        print(f"    {loaded} {model_name:20s}: {count} موسم")
    print("  " + "-" * 40)
    print("")

    # المواسم الكاملة
    complete = diag.get("complete_seasons", [])
    print(f"  المواسم الكاملة ({len(complete)}):")
    if complete:
        for s in complete:
            print(f"    • {s}")
    else:
        print("    لا توجد مواسم كاملة.")
    print("")

    # أخطاء التحميل
    errors = diag.get("load_errors", [])
    if errors:
        print(f"  ⚠ أخطاء التحميل ({len(errors)}):")
        for err in errors:
            print(f"    • {err}")
        print("")

    # إعدادات
    cfg = diag.get("config", {})
    print("  الإعدادات:")
    print("  " + "-" * 40)
    for key, value in cfg.items():
        print(f"    {key:25s}: {value}")
    print("  " + "-" * 40)

    print("=" * 60)
    print("")


# -----------------------------------------------------------------------------
# دوال الحفظ
# -----------------------------------------------------------------------------

def save_result(
    result: Dict[str, Any],
    output_path: Path,
) -> bool:
    """
    حفظ نتيجة التنبؤ في ملف JSON.

    المعاملات:
        result: قاموس النتيجة
        output_path: مسار ملف الحفظ

    العائد:
        True إذا تم الحفظ بنجاح
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        log(f"تم حفظ نتيجة التنبؤ في: {output_path}", "INFO")
        return True

    except Exception as e:
        log(f"فشل حفظ نتيجة التنبؤ: {e}", "ERROR")
        return False


def generate_output_filename(
    team1: str,
    team2: str,
    comp: str,
) -> str:
    """
    توليد اسم ملف تلقائي لنتيجة التنبؤ.

    المعاملات:
        team1: اسم أو معرّف الفريق المضيف
        team2: اسم أو معرّف الفريق الضيف
        comp: رمز المسابقة

    العائد:
        اسم ملف (بدون مسار)
    """
    # تنظيف الأسماء لاستخدامها في اسم الملف
    def clean_name(name: str) -> str:
        # استبدال المسافات والأحرف الخاصة
        cleaned = name.strip().replace(" ", "_").replace("/", "-")
        # إزالة الأحرف غير الآمنة
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        cleaned = "".join(c for c in cleaned if c in safe_chars)
        return cleaned[:30]  # حد أقصى 30 حرف

    t1 = clean_name(team1)
    t2 = clean_name(team2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"prediction_{comp}_{t1}_vs_{t2}_{timestamp}.json"


# -----------------------------------------------------------------------------
# الدالة الرئيسية
# -----------------------------------------------------------------------------

def main():
    """
    نقطة الدخول الرئيسية — تحليل سطر الأوامر وتشغيل التنبؤ.
    """
    parser = argparse.ArgumentParser(
        description="التنبؤ بنتيجة مباراة كرة قدم باستخدام نموذج Dixon-Coles الإحصائي.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:

  # تنبؤ بسيط
  python 03_predict.py --team1 "Manchester City" --team2 "Liverpool" --comp PL

  # مع تعديل ELO وأفضل 5 نتائج
  python 03_predict.py --team1 "Real Madrid" --team2 "Barcelona" --comp PD --use-elo --topk 5

  # باستخدام معرّفات الفرق
  python 03_predict.py --home-id 65 --away-id 64 --comp PL --use-elo

  # حفظ النتيجة
  python 03_predict.py --team1 "Bayern" --team2 "Dortmund" --comp BL1 --save

  # حفظ في ملف محدد
  python 03_predict.py --team1 "PSG" --team2 "Lyon" --comp FL1 --save --output result.json

  # إخراج JSON فقط (للدمج مع أدوات أخرى)
  python 03_predict.py --team1 "Juventus" --team2 "Milan" --comp SA --json

  # تحديد الموسم يدوياً
  python 03_predict.py --team1 "Arsenal" --team2 "Chelsea" --comp PL --season 2024

  # عرض مدخلات النموذج
  python 03_predict.py --team1 "Napoli" --team2 "Inter" --comp SA --verbose

  # تشخيص حالة النماذج
  python 03_predict.py --diagnostics
        """
    )

    # --- مجموعة تحديد الفرق (بالاسم) ---
    name_group = parser.add_argument_group("تحديد الفرق بالاسم")
    name_group.add_argument(
        "--team1",
        type=str,
        default=None,
        help="اسم الفريق المضيف."
    )
    name_group.add_argument(
        "--team2",
        type=str,
        default=None,
        help="اسم الفريق الضيف."
    )

    # --- مجموعة تحديد الفرق (بالمعرّف) ---
    id_group = parser.add_argument_group("تحديد الفرق بالمعرّف")
    id_group.add_argument(
        "--home-id",
        type=int,
        default=None,
        help="معرّف الفريق المضيف (بدلاً من --team1)."
    )
    id_group.add_argument(
        "--away-id",
        type=int,
        default=None,
        help="معرّف الفريق الضيف (بدلاً من --team2)."
    )

    # --- إعدادات المسابقة ---
    comp_group = parser.add_argument_group("إعدادات المسابقة")
    comp_group.add_argument(
        "--comp",
        type=str,
        default=None,
        help="رمز المسابقة (مثلاً: PL, PD, SA, BL1, FL1, CL, DED, PPL, BSA)."
    )
    comp_group.add_argument(
        "--season",
        type=int,
        default=None,
        help="سنة بداية الموسم (اختياري — تلقائي إن لم يُحدد)."
    )

    # --- إعدادات التنبؤ ---
    pred_group = parser.add_argument_group("إعدادات التنبؤ")
    pred_group.add_argument(
        "--topk",
        type=int,
        default=0,
        help="عدد أفضل النتائج المحتملة للعرض (0 = بدون). (افتراضي: 0)"
    )
    pred_group.add_argument(
        "--use-elo",
        action="store_true",
        help="تفعيل تعديل ELO في حساب معدّلات الأهداف."
    )

    # --- إعدادات الإخراج ---
    output_group = parser.add_argument_group("إعدادات الإخراج")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="إخراج JSON فقط (بدون تنسيق مرئي). مفيد للدمج مع أدوات أخرى."
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="عرض مدخلات النموذج التفصيلية."
    )
    output_group.add_argument(
        "--save",
        action="store_true",
        help="حفظ نتيجة التنبؤ في ملف JSON."
    )
    output_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="مسار ملف الحفظ (يُستخدم مع --save). (افتراضي: تلقائي)"
    )

    # --- أدوات ---
    tools_group = parser.add_argument_group("أدوات")
    tools_group.add_argument(
        "--diagnostics",
        action="store_true",
        help="عرض معلومات تشخيصية عن حالة النماذج والخروج."
    )
    tools_group.add_argument(
        "--dry-run",
        action="store_true",
        help="عرض الإعدادات بدون تنفيذ التنبؤ."
    )

    args = parser.parse_args()

    # =========================================================================
    # وضع التشخيص
    # =========================================================================
    if args.diagnostics:
        try:
            predictor = Predictor()
            display_diagnostics(predictor)
            sys.exit(EXIT_SUCCESS)
        except Exception as e:
            log(f"فشل التشخيص: {e}", "CRITICAL")
            traceback.print_exc()
            sys.exit(EXIT_ERROR_LOAD)

    # =========================================================================
    # التحقق من المدخلات
    # =========================================================================

    # تحديد الفريق المضيف
    home_identifier = None
    if args.home_id is not None:
        home_identifier = str(args.home_id)
    elif args.team1 is not None:
        home_identifier = args.team1
    else:
        parser.error(
            "يجب تحديد الفريق المضيف عبر --team1 أو --home-id."
        )

    # تحديد الفريق الضيف
    away_identifier = None
    if args.away_id is not None:
        away_identifier = str(args.away_id)
    elif args.team2 is not None:
        away_identifier = args.team2
    else:
        parser.error(
            "يجب تحديد الفريق الضيف عبر --team2 أو --away-id."
        )

    # التحقق من وجود رمز المسابقة
    if not args.comp:
        parser.error(
            "يجب تحديد رمز المسابقة عبر --comp "
            "(مثلاً: PL, PD, SA, BL1, FL1)."
        )

    comp_code = args.comp.strip().upper()

    # =========================================================================
    # وضع التشغيل الجاف
    # =========================================================================
    if args.dry_run:
        print("")
        print("=" * 50)
        print("  [DRY RUN] إعدادات التنبؤ:")
        print("=" * 50)
        print(f"  المضيف    : {home_identifier}")
        print(f"  الضيف     : {away_identifier}")
        print(f"  المسابقة  : {comp_code}")
        print(f"  الموسم    : {args.season or 'تلقائي'}")
        print(f"  ELO       : {'نعم' if args.use_elo else 'لا'}")
        print(f"  Top-K     : {args.topk}")
        print(f"  حفظ       : {'نعم' if args.save else 'لا'}")
        print(f"  JSON فقط  : {'نعم' if args.json else 'لا'}")
        print("=" * 50)
        print("  [DRY RUN] لم يتم تنفيذ التنبؤ.")
        print("")
        sys.exit(EXIT_SUCCESS)

    # =========================================================================
    # تحميل المتنبئ
    # =========================================================================
    try:
        predictor = Predictor()

        if not predictor.is_loaded:
            errors = predictor.load_errors
            if errors:
                log("أخطاء أثناء تحميل النماذج:", "WARNING")
                for err in errors:
                    log(f"  • {err}", "WARNING")

            # نستمر حتى لو لم تُحمّل جميع النماذج
            # (قد تكون المسابقة المطلوبة متوفرة)

    except Exception as e:
        log(f"فشل تحميل المتنبئ: {e}", "CRITICAL")
        if not args.json:
            traceback.print_exc()
        sys.exit(EXIT_ERROR_LOAD)

    # =========================================================================
    # تنفيذ التنبؤ
    # =========================================================================
    try:
        result = predictor.predict(
            team1_name=home_identifier,
            team2_name=away_identifier,
            comp_code=comp_code,
            topk=args.topk,
            use_elo=args.use_elo,
            preferred_season_year=args.season,
        )

    except ValueError as e:
        log(f"خطأ في المدخلات: {e}", "ERROR")
        if args.json:
            error_json = {"error": str(e), "type": "ValueError"}
            print(json.dumps(error_json, ensure_ascii=False, indent=2))
        sys.exit(EXIT_ERROR_PREDICT)

    except Exception as e:
        log(f"فشل التنبؤ: {e}", "CRITICAL")
        if args.json:
            error_json = {"error": str(e), "type": type(e).__name__}
            print(json.dumps(error_json, ensure_ascii=False, indent=2))
        else:
            traceback.print_exc()
        sys.exit(EXIT_ERROR_PREDICT)

    # =========================================================================
    # عرض النتائج
    # =========================================================================
    if args.json:
        # إخراج JSON فقط (بدون أي تنسيق مرئي)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        # عرض منسّق
        display_prediction_formatted(
            result,
            show_inputs=args.verbose,
            show_scorelines=(args.topk > 0),
        )

    # =========================================================================
    # حفظ النتائج (اختياري)
    # =========================================================================
    if args.save:
        if args.output:
            output_path = Path(args.output)
        else:
            filename = generate_output_filename(
                home_identifier, away_identifier, comp_code
            )
            output_path = config.DATA_DIR / filename

        saved = save_result(result, output_path)

        if not saved:
            sys.exit(EXIT_ERROR_SAVE)

        if not args.json:
            print(f"  📁 تم حفظ النتيجة في: {output_path}")
            print("")

    # =========================================================================
    # خروج ناجح
    # =========================================================================
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("")
        log("تم إيقاف العملية بواسطة المستخدم (Ctrl+C).", "WARNING")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        log(f"خطأ غير متوقع: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(EXIT_ERROR_UNEXPECTED)
