# 05_train_ml_model.py
# -----------------------------------------------------------------------------
# الوصف:
# يقرأ هذا السكريبت مجموعة البيانات التي تم إنشاؤها (`ml_dataset.csv`)
# ويقوم بتدريب نموذج تعلم آلة (XGBoost Classifier) للتنبؤ بنتيجة المباراة.
#
# التحسينات:
# - حفظ بيانات وصفية (metadata) بجانب النموذج لضمان التوافق عند التحميل
# - دعم التقسيم الزمني (temporal split) لتجنب تسرب البيانات الزمني
# - معالجة المعاملات المُهملة (deprecated) في XGBoost الحديث
# - تحليل أهمية الميزات وحفظها
# - حفظ تقرير التقييم الكامل
# - تحقق شامل من صحة البيانات قبل التدريب
# - معالجة أخطاء أكثر تفصيلاً
# -----------------------------------------------------------------------------
import sys
import os
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# --- إضافة مسار المشروع الرئيسي إلى بايثون ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# قد لا يتوفر XGBoost في كل بيئة تشغيل
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# قد لا يتوفر sklearn في كل بيئة تشغيل
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    train_test_split = None
    StratifiedKFold = None
    cross_val_score = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    LabelEncoder = None

from common import config
from common.utils import log


# -----------------------------------------------------------------------------
# ثوابت التدريب الافتراضية
# -----------------------------------------------------------------------------
# قائمة الميزات المستخدمة في التدريب
DEFAULT_FEATURES = [
    'home_attack',
    'away_attack',
    'home_defense',
    'away_defense',
    'home_elo',
    'away_elo',
    'elo_diff',
    'home_avg_points',
    'away_avg_points',
]

# اسم العمود الهدف
TARGET_COLUMN = 'result'

# الفئات المتوقعة في العمود الهدف
# -1 = فوز الضيف، 0 = تعادل، 1 = فوز المضيف
EXPECTED_CLASSES = [-1, 0, 1]

# نسبة بيانات الاختبار
DEFAULT_TEST_SIZE = 0.2

# البذرة العشوائية لضمان قابلية التكرار
DEFAULT_RANDOM_STATE = 42

# معاملات XGBoost الافتراضية
DEFAULT_XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': DEFAULT_RANDOM_STATE,
    'early_stopping_rounds': 30,
    'verbosity': 0,
}


# -----------------------------------------------------------------------------
# دوال مساعدة
# -----------------------------------------------------------------------------
def check_dependencies() -> bool:
    """
    التحقق من توفر جميع المكتبات المطلوبة.

    العائد: True إذا كانت جميع المكتبات متوفرة، False خلاف ذلك.
    """
    missing = []
    if xgb is None:
        missing.append("xgboost")
    if LabelEncoder is None:
        missing.append("scikit-learn")
    if missing:
        log(
            f"المكتبات التالية غير متوفرة: {', '.join(missing)}. "
            f"يرجى تثبيتها عبر: pip install {' '.join(missing)}",
            "CRITICAL"
        )
        return False
    return True


def load_dataset(dataset_path: Path) -> Optional[pd.DataFrame]:
    """
    تحميل مجموعة البيانات من ملف CSV.

    المعاملات:
        dataset_path: مسار ملف CSV

    العائد: DataFrame يحتوي على البيانات، أو None في حالة الفشل
    """
    if not dataset_path.exists():
        log(
            f"ملف مجموعة البيانات غير موجود: {dataset_path}. "
            f"يرجى تشغيل 04_feature_generator.py أولاً.",
            "CRITICAL"
        )
        return None

    try:
        df = pd.read_csv(dataset_path)
        log(f"تم تحميل مجموعة البيانات بنجاح: {len(df)} صف × {len(df.columns)} عمود", "INFO")
        return df
    except pd.errors.EmptyDataError:
        log(f"ملف مجموعة البيانات فارغ: {dataset_path}", "CRITICAL")
        return None
    except pd.errors.ParserError as e:
        log(f"خطأ في تحليل ملف CSV: {e}", "CRITICAL")
        return None
    except Exception as e:
        log(f"خطأ غير متوقع أثناء تحميل مجموعة البيانات: {e}", "CRITICAL")
        return None


def validate_dataset(
    df: pd.DataFrame,
    features: List[str],
    target: str
) -> Tuple[bool, List[str]]:
    """
    التحقق من صحة مجموعة البيانات قبل التدريب.

    المعاملات:
        df: مجموعة البيانات
        features: قائمة أسماء أعمدة الميزات
        target: اسم العمود الهدف

    العائد: tuple يحتوي على (صحيح/خطأ, قائمة رسائل التحذير أو الأخطاء)
    """
    issues = []

    # التحقق من وجود العمود الهدف
    if target not in df.columns:
        issues.append(f"العمود الهدف '{target}' غير موجود في مجموعة البيانات.")
        return False, issues

    # التحقق من وجود أعمدة الميزات
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        issues.append(
            f"الأعمدة التالية غير موجودة في مجموعة البيانات: {', '.join(missing_features)}"
        )
        return False, issues

    # التحقق من عدم وجود صفوف فارغة بالكامل
    if df.empty:
        issues.append("مجموعة البيانات فارغة.")
        return False, issues

    # التحقق من الحد الأدنى لعدد الصفوف
    min_rows = 50
    if len(df) < min_rows:
        issues.append(
            f"مجموعة البيانات صغيرة جداً ({len(df)} صف). "
            f"يُوصى بحد أدنى {min_rows} صف للحصول على نتائج موثوقة."
        )  # تحذير وليس خطأ — نسمح بالاستمرار

    # التحقق من القيم المفقودة في الميزات
    feature_nulls = df[features].isnull().sum()
    features_with_nulls = feature_nulls[feature_nulls > 0]
    if not features_with_nulls.empty:
        for col_name, null_count in features_with_nulls.items():
            pct = (null_count / len(df)) * 100
            issues.append(
                f"العمود '{col_name}' يحتوي على {null_count} قيمة مفقودة ({pct:.1f}%)"
            )

    # التحقق من القيم المفقودة في الهدف
    target_nulls = df[target].isnull().sum()
    if target_nulls > 0:
        pct = (target_nulls / len(df)) * 100
        issues.append(
            f"العمود الهدف '{target}' يحتوي على {target_nulls} قيمة مفقودة ({pct:.1f}%)"
        )

    # التحقق من فئات الهدف
    unique_classes = sorted(df[target].dropna().unique().tolist())
    if unique_classes != sorted(EXPECTED_CLASSES):
        issues.append(
            f"فئات الهدف غير متوقعة: {unique_classes}. "
            f"المتوقع: {sorted(EXPECTED_CLASSES)}"
        )

    # التحقق من توازن الفئات
    class_counts = df[target].value_counts()
    total = len(df)
    for cls, count in class_counts.items():
        pct = (count / total) * 100
        if pct < 5:
            issues.append(
                f"الفئة {cls} ممثلة بشكل ضعيف جداً: {count} ({pct:.1f}%). "
                f"قد يؤثر ذلك على أداء النموذج."
            )

    # التحقق من القيم اللانهائية في الميزات
    inf_counts = np.isinf(df[features].select_dtypes(include=[np.number])).sum()
    features_with_inf = inf_counts[inf_counts > 0]
    if not features_with_inf.empty:
        for col_name, inf_count in features_with_inf.items():
            issues.append(
                f"العمود '{col_name}' يحتوي على {inf_count} قيمة لانهائية (inf)"
            )

    # تحديد ما إذا كانت المشاكل حرجة (تمنع التدريب)
    critical_keywords = ["غير موجود", "فارغة", "غير متوقعة"]
    has_critical = any(
        any(kw in issue for kw in critical_keywords)
        for issue in issues
    )
    return not has_critical, issues


def clean_dataset(
    df: pd.DataFrame,
    features: List[str],
    target: str
) -> pd.DataFrame:
    """
    تنظيف مجموعة البيانات من القيم المفقودة واللانهائية.

    المعاملات:
        df: مجموعة البيانات الأصلية
        features: قائمة أسماء أعمدة الميزات
        target: اسم العمود الهدف

    العائد: DataFrame نظيف بعد إزالة الصفوف المشكلة
    """
    original_len = len(df)
    # نسخة من البيانات لتجنب التعديل على الأصل
    df_clean = df.copy()

    # استبدال القيم اللانهائية بـ NaN
    numeric_features = df_clean[features].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_features:
        df_clean[numeric_features] = df_clean[numeric_features].replace([np.inf, -np.inf], np.nan)

    # حذف الصفوف التي تحتوي على قيم مفقودة في الميزات أو الهدف
    cols_to_check = features + [target]
    df_clean = df_clean.dropna(subset=cols_to_check)

    removed = original_len - len(df_clean)
    if removed > 0:
        log(
            f"تم حذف {removed} صف يحتوي على قيم مفقودة أو لانهائية "
            f"(من {original_len} إلى {len(df_clean)})",
            "WARNING"
        )

    return df_clean.reset_index(drop=True)


def encode_target(
    y: pd.Series,
    expected_classes: Optional[List[int]] = None
) -> Tuple[np.ndarray, LabelEncoder, Dict]:
    """
    ترميز العمود الهدف باستخدام LabelEncoder.

    المعاملات:
        y: السلسلة الهدف (تحتوي على -1, 0, 1)
        expected_classes: الفئات المتوقعة (اختياري)

    العائد: tuple يحتوي على:
        - المصفوفة المُرمّزة
        - كائن LabelEncoder
        - قاموس الربط (الفئة الأصلية -> الفئة المُرمّزة)
    """
    if expected_classes is None:
        expected_classes = EXPECTED_CLASSES

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # بناء قاموس الربط
    class_mapping = {}
    for original_class in expected_classes:
        encoded_value = int(le.transform([original_class])[0])
        class_mapping[int(original_class)] = encoded_value

    log(f"ربط الفئات: {class_mapping}", "DEBUG")
    log(f"فئات LabelEncoder: {le.classes_.tolist()}", "DEBUG")
    return y_encoded, le, class_mapping


def split_data_random(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    تقسيم البيانات عشوائياً مع الحفاظ على توازن الفئات (stratified).

    المعاملات:
        X: ميزات التدريب
        y_encoded: الهدف المُرمّز
        test_size: نسبة بيانات الاختبار
        random_state: البذرة العشوائية

    العائد: tuple يحتوي على (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    log(f"التقسيم العشوائي (Stratified):", "INFO")
    log(f" حجم مجموعة التدريب: {len(X_train)}", "INFO")
    log(f" حجم مجموعة الاختبار: {len(X_test)}", "INFO")
    return X_train, X_test, y_train, y_test


def split_data_temporal(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    date_column: str = 'match_date'
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    تقسيم البيانات زمنياً لتجنب تسرب البيانات الزمني.
    المباريات الأقدم للتدريب والأحدث للاختبار.

    المعاملات:
        df: مجموعة البيانات الكاملة (تحتوي على عمود التاريخ)
        X: ميزات التدريب
        y_encoded: الهدف المُرمّز
        test_size: نسبة بيانات الاختبار
        date_column: اسم عمود التاريخ

    العائد: tuple يحتوي على (X_train, X_test, y_train, y_test)
    """
    if date_column not in df.columns:
        log(
            f"عمود التاريخ '{date_column}' غير موجود. "
            f"سيتم استخدام التقسيم العشوائي بدلاً من ذلك.",
            "WARNING"
        )
        return split_data_random(X, y_encoded, test_size)

    # ترتيب البيانات حسب التاريخ
    sorted_indices = df[date_column].sort_values().index
    n_total = len(sorted_indices)
    n_train = int(n_total * (1 - test_size))

    train_indices = sorted_indices[:n_train]
    test_indices = sorted_indices[n_train:]

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]

    log(f"التقسيم الزمني (Temporal):", "INFO")
    log(f" حجم مجموعة التدريب: {len(X_train)}", "INFO")
    log(f" حجم مجموعة الاختبار: {len(X_test)}", "INFO")

    # عرض نطاق التواريخ لكل مجموعة
    try:
        train_dates = df.loc[train_indices, date_column]
        test_dates = df.loc[test_indices, date_column]
        log(
            f" فترة التدريب: {train_dates.min()} → {train_dates.max()}",
            "INFO"
        )
        log(
            f" فترة الاختبار: {test_dates.min()} → {test_dates.max()}",
            "INFO"
        )
    except Exception:
        pass

    return X_train, X_test, y_train, y_test


def build_xgb_model(params: Optional[Dict] = None) -> xgb.XGBClassifier:
    """
    إنشاء نموذج XGBoost Classifier بالمعاملات المحددة.

    المعاملات:
        params: قاموس المعاملات (اختياري، يستخدم الافتراضي إن لم يُحدد)

    العائد: كائن XGBClassifier جاهز للتدريب
    """
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()

    log("معاملات نموذج XGBoost:", "DEBUG")
    for key, value in params.items():
        log(f" {key}: {value}", "DEBUG")

    model = xgb.XGBClassifier(**params)
    return model


def train_model(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> xgb.XGBClassifier:
    """
    تدريب نموذج XGBoost على بيانات التدريب مع مراقبة الأداء على بيانات الاختبار.

    المعاملات:
        model: نموذج XGBoost غير مُدرّب
        X_train: ميزات التدريب
        y_train: الهدف المُرمّز للتدريب
        X_test: ميزات الاختبار
        y_test: الهدف المُرمّز للاختبار

    العائد: نموذج XGBoost مُدرّب
    """
    log("جارٍ تدريب النموذج...", "INFO")

    # تدريب النموذج مع مجموعة التقييم
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # عرض معلومات التوقف المبكر
    best_iteration = getattr(model, 'best_iteration', None)
    best_score = getattr(model, 'best_score', None)
    if best_iteration is not None:
        log(f"أفضل تكرار (best iteration): {best_iteration}", "INFO")
    if best_score is not None:
        log(f"أفضل نتيجة (best score): {best_score:.6f}", "INFO")

    log("تم الانتهاء من تدريب النموذج بنجاح.", "INFO")
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder
) -> Dict:
    """
    تقييم أداء النموذج على مجموعة الاختبار.

    المعاملات:
        model: نموذج XGBoost مُدرّب
        X_test: ميزات الاختبار
        y_test: الهدف المُرمّز للاختبار
        le: كائن LabelEncoder لعكس الترميز

    العائد: قاموس يحتوي على مقاييس الأداء
    """
    log("تقييم أداء النموذج على مجموعة الاختبار...", "INFO")

    # التوقعات
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # الدقة الإجمالية
    acc = accuracy_score(y_test, y_pred)
    log(f"دقة النموذج (Accuracy): {acc:.2%}", "INFO")

    # تقرير التصنيف
    class_names = [str(c) for c in le.classes_]
    report_text = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=False
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )

    print("\n" + "=" * 60)
    print("تقرير التصنيف")
    print("=" * 60)
    print(report_text)
    print("=" * 60 + "\n")

    # مصفوفة الالتباس
    cm = confusion_matrix(y_test, y_pred)
    log("مصفوفة الالتباس (Confusion Matrix):", "INFO")
    log(f" الفئات: {class_names}", "INFO")
    for i, row in enumerate(cm):
        log(f" {class_names[i]}: {row.tolist()}", "INFO")

    # توزيع الفئات في مجموعة الاختبار
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_distribution = {}
    for cls_encoded, count in zip(unique_test, counts_test):
        cls_original = int(le.inverse_transform([cls_encoded])[0])
        pct = (count / len(y_test)) * 100
        test_distribution[str(cls_original)] = {
            "count": int(count),
            "percentage": round(pct, 2)
        }
    log(f"توزيع الفئات في مجموعة الاختبار: {test_distribution}", "INFO")

    # تجميع نتائج التقييم
    evaluation_results = {
        "accuracy": round(float(acc), 6),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "test_distribution": test_distribution,
        "test_size": int(len(y_test)),
    }
    return evaluation_results


def analyze_feature_importance(
    model: xgb.XGBClassifier,
    features: List[str]
) -> Dict[str, float]:
    """
    تحليل أهمية الميزات في النموذج المُدرّب.

    المعاملات:
        model: نموذج XGBoost مُدرّب
        features: قائمة أسماء الميزات

    العائد: قاموس مرتب تنازلياً من (اسم_الميزة -> الأهمية)
    """
    log("تحليل أهمية الميزات...", "INFO")
    try:
        importance_scores = model.feature_importances_
    except AttributeError:
        log("لم يتمكن النموذج من حساب أهمية الميزات.", "WARNING")
        return {}

    # ربط الأسماء بالأهمية وترتيبها تنازلياً
    importance_dict = {}
    for feature_name, score in zip(features, importance_scores):
        importance_dict[feature_name] = round(float(score), 6)

    # ترتيب تنازلي حسب الأهمية
    importance_sorted = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # عرض النتائج
    print("\n" + "=" * 60)
    print("أهمية الميزات (Feature Importance)")
    print("=" * 60)
    for rank, (feat_name, feat_score) in enumerate(importance_sorted.items(), 1):
        bar = "█" * int(feat_score * 50)
        print(f" {rank}. {feat_name:25s} : {feat_score:.4f} {bar}")
    print("=" * 60 + "\n")

    return importance_sorted


def run_cross_validation(
    model_params: Dict,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    n_folds: int = 5
) -> Optional[Dict]:
    """
    إجراء تحقق متقاطع (Cross-Validation) لتقييم استقرار النموذج.

    المعاملات:
        model_params: معاملات نموذج XGBoost
        X: جميع الميزات
        y_encoded: جميع القيم الهدف المُرمّزة
        n_folds: عدد الطيّات

    العائد: قاموس يحتوي على نتائج التحقق المتقاطع، أو None في حالة الفشل
    """
    if cross_val_score is None or StratifiedKFold is None:
        log("مكتبة sklearn غير متوفرة لإجراء التحقق المتقاطع.", "WARNING")
        return None

    log(f"إجراء التحقق المتقاطع ({n_folds} طيّات)...", "INFO")
    try:
        # إنشاء نموذج جديد بدون early_stopping لأنه لا يتوافق مع cross_val_score مباشرة
        cv_params = model_params.copy()
        cv_params.pop('early_stopping_rounds', None)
        # تقليل عدد التكرارات للتحقق المتقاطع لتسريع العملية
        cv_params['n_estimators'] = min(cv_params.get('n_estimators', 200), 200)

        cv_model = xgb.XGBClassifier(**cv_params)

        cv_strategy = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=DEFAULT_RANDOM_STATE
        )

        scores = cross_val_score(
            cv_model, X, y_encoded,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1
        )

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        log(f"نتائج التحقق المتقاطع ({n_folds} طيّات):", "INFO")
        log(f" الدقة لكل طيّة: {[round(s, 4) for s in scores.tolist()]}", "INFO")
        log(f" متوسط الدقة: {mean_score:.4f} ± {std_score:.4f}", "INFO")

        cv_results = {
            "n_folds": n_folds,
            "scores": [round(float(s), 6) for s in scores.tolist()],
            "mean_accuracy": round(mean_score, 6),
            "std_accuracy": round(std_score, 6),
        }
        return cv_results

    except Exception as e:
        log(f"فشل التحقق المتقاطع: {e}", "WARNING")
        return None


def save_model(
    model: xgb.XGBClassifier,
    model_path: Path
) -> bool:
    """
    حفظ نموذج XGBoost المُدرّب في ملف.

    المعاملات:
        model: نموذج XGBoost مُدرّب
        model_path: مسار ملف الحفظ

    العائد: True إذا تم الحفظ بنجاح، False خلاف ذلك
    """
    try:
        model_path.parent.mkdir(exist_ok=True, parents=True)
        model.save_model(str(model_path))
        log(f"تم حفظ نموذج XGBoost في: {model_path}", "INFO")
        # التحقق من صحة الحفظ
        file_size = model_path.stat().st_size
        log(f"حجم ملف النموذج: {file_size:,} بايت", "INFO")
        return True
    except Exception as e:
        log(f"فشل حفظ النموذج: {e}", "CRITICAL")
        return False


def save_model_metadata(
    metadata_path: Path,
    features: List[str],
    le: LabelEncoder,
    class_mapping: Dict,
    evaluation_results: Dict,
    feature_importance: Dict[str, float],
    cv_results: Optional[Dict],
    xgb_params: Dict,
    dataset_info: Dict,
    model_classes: List
) -> bool:
    """
    حفظ البيانات الوصفية للنموذج المُدرّب في ملف JSON.
    هذا يضمن التوافق عند تحميل النموذج لاحقاً في app.py.

    المعاملات:
        metadata_path: مسار ملف الحفظ
        features: قائمة أسماء الميزات المستخدمة
        le: كائن LabelEncoder
        class_mapping: قاموس ربط الفئات
        evaluation_results: نتائج التقييم
        feature_importance: أهمية الميزات
        cv_results: نتائج التحقق المتقاطع (اختياري)
        xgb_params: معاملات XGBoost المستخدمة
        dataset_info: معلومات عن مجموعة البيانات
        model_classes: الفئات من model.classes_

    العائد: True إذا تم الحفظ بنجاح، False خلاف ذلك
    """
    try:
        metadata = {
            "version": getattr(config, "VERSION", "N/A"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "XGBClassifier",
            "model_file": "xgboost_model.json",
            "features": features,
            "target_column": TARGET_COLUMN,
            "expected_classes": EXPECTED_CLASSES,
            "label_encoder_classes": [int(c) for c in le.classes_.tolist()],
            "model_classes": [int(c) for c in model_classes],
            "class_mapping": {str(k): int(v) for k, v in class_mapping.items()},
            "class_mapping_description": {
                "-1": "فوز الضيف (Away Win)",
                "0": "تعادل (Draw)",
                "1": "فوز المضيف (Home Win)",
            },
            "xgb_params": {
                k: v for k, v in xgb_params.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            "evaluation": evaluation_results,
            "feature_importance": feature_importance,
            "cross_validation": cv_results,
            "dataset_info": dataset_info,
        }
        metadata_path.parent.mkdir(exist_ok=True, parents=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        log(f"تم حفظ البيانات الوصفية للنموذج في: {metadata_path}", "INFO")
        return True
    except Exception as e:
        log(f"فشل حفظ البيانات الوصفية: {e}", "WARNING")
        return False


def save_evaluation_report(
    report_path: Path,
    evaluation_results: Dict,
    feature_importance: Dict[str, float],
    cv_results: Optional[Dict]
) -> bool:
    """
    حفظ تقرير التقييم الكامل في ملف نصي قابل للقراءة.

    المعاملات:
        report_path: مسار ملف التقرير
        evaluation_results: نتائج التقييم
        feature_importance: أهمية الميزات
        cv_results: نتائج التحقق المتقاطع (اختياري)

    العائد: True إذا تم الحفظ بنجاح، False خلاف ذلك
    """
    try:
        report_path.parent.mkdir(exist_ok=True, parents=True)

        lines = []
        lines.append("=" * 70)
        lines.append("تقرير تقييم نموذج تعلم الآلة (XGBoost)")
        lines.append(f"تاريخ الإنشاء: {datetime.now(timezone.utc).isoformat()}")
        lines.append("=" * 70)
        lines.append("")

        # الدقة
        lines.append(f"الدقة الإجمالية (Accuracy): {evaluation_results.get('accuracy', 'N/A'):.2%}")
        lines.append(f"حجم مجموعة الاختبار: {evaluation_results.get('test_size', 'N/A')}")
        lines.append("")

        # توزيع الفئات
        lines.append("توزيع الفئات في مجموعة الاختبار:")
        test_dist = evaluation_results.get("test_distribution", {})
        for cls, info in test_dist.items():
            lines.append(f" الفئة {cls}: {info.get('count', 0)} ({info.get('percentage', 0):.1f}%)")
        lines.append("")

        # مصفوفة الالتباس
        lines.append("مصفوفة الالتباس:")
        class_names = evaluation_results.get("class_names", [])
        cm = evaluation_results.get("confusion_matrix", [])
        if class_names and cm:
            header = " " + " ".join(f"{cn:>8}" for cn in class_names)
            lines.append(header)
            for i, row in enumerate(cm):
                row_str = f" {class_names[i]:>4}: " + " ".join(f"{v:>8}" for v in row)
                lines.append(row_str)
        lines.append("")

        # أهمية الميزات
        lines.append("أهمية الميزات:")
        for rank, (feat, score) in enumerate(feature_importance.items(), 1):
            bar = "█" * int(score * 50)
            lines.append(f" {rank}. {feat:25s} : {score:.4f} {bar}")
        lines.append("")

        # التحقق المتقاطع
        if cv_results:
            lines.append(f"التحقق المتقاطع ({cv_results.get('n_folds', 'N/A')} طيّات):")
            lines.append(f" الدقة لكل طيّة: {cv_results.get('scores', [])}")
            lines.append(
                f" المتوسط: {cv_results.get('mean_accuracy', 0):.4f} "
                f"± {cv_results.get('std_accuracy', 0):.4f}"
            )
            lines.append("")

        lines.append("=" * 70)

        report_text = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        log(f"تم حفظ تقرير التقييم في: {report_path}", "INFO")
        return True
    except Exception as e:
        log(f"فشل حفظ تقرير التقييم: {e}", "WARNING")
        return False


# -----------------------------------------------------------------------------
# الدالة الرئيسية
# -----------------------------------------------------------------------------
def run_ml_trainer(
    use_temporal_split: bool = False,
    run_cv: bool = True,
    cv_folds: int = 5,
    custom_params: Optional[Dict] = None,
    custom_features: Optional[List[str]] = None
):
    """
    الدالة الرئيسية لتدريب نموذج تعلم الآلة (XGBoost).

    المعاملات:
        use_temporal_split: استخدام التقسيم الزمني بدلاً من العشوائي
        run_cv: تشغيل التحقق المتقاطع
        cv_folds: عدد طيّات التحقق المتقاطع
        custom_params: معاملات XGBoost مخصصة (اختياري)
        custom_features: قائمة ميزات مخصصة (اختياري)
    """
    log("=" * 70, "INFO")
    log("بدء تدريب نموذج تعلم الآلة (XGBoost)", "INFO")
    log(f"الوقت: {datetime.now(timezone.utc).isoformat()}", "INFO")
    log("=" * 70, "INFO")

    # =========================================================================
    # 0. التحقق من المتطلبات
    # =========================================================================
    if not check_dependencies():
        log("لا يمكن المتابعة بدون المكتبات المطلوبة.", "CRITICAL")
        return

    # =========================================================================
    # 1. تحميل مجموعة البيانات
    # =========================================================================
    log("--- المرحلة 1: تحميل مجموعة البيانات ---", "INFO")
    dataset_path = config.DATA_DIR / "ml_dataset.csv"
    df = load_dataset(dataset_path)
    if df is None:
        return

    # تحديد الميزات المستخدمة
    features = custom_features if custom_features else DEFAULT_FEATURES.copy()
    target = TARGET_COLUMN
    log(f"الميزات المستخدمة ({len(features)}): {features}", "INFO")
    log(f"العمود الهدف: {target}", "INFO")

    # =========================================================================
    # 2. التحقق من صحة البيانات
    # =========================================================================
    log("--- المرحلة 2: التحقق من صحة البيانات ---", "INFO")
    is_valid, validation_issues = validate_dataset(df, features, target)
    if validation_issues:
        log("مشاكل تم اكتشافها في البيانات:", "WARNING")
        for issue in validation_issues:
            log(f" ⚠ {issue}", "WARNING")
    if not is_valid:
        log("البيانات تحتوي على مشاكل حرجة تمنع التدريب.", "CRITICAL")
        return

    # =========================================================================
    # 3. تنظيف البيانات
    # =========================================================================
    log("--- المرحلة 3: تنظيف البيانات ---", "INFO")
    df_clean = clean_dataset(df, features, target)
    if df_clean.empty:
        log("لم تبقَ أي بيانات بعد التنظيف.", "CRITICAL")
        return

    # معلومات مجموعة البيانات
    dataset_info = {
        "original_rows": int(len(df)),
        "cleaned_rows": int(len(df_clean)),
        "removed_rows": int(len(df) - len(df_clean)),
        "features_count": len(features),
        "features": features,
        "source_file": str(dataset_path),
    }

    # عرض إحصائيات وصفية
    log("إحصائيات البيانات المُنظّفة:", "INFO")
    desc = df_clean[features].describe()
    print("\n" + desc.to_string() + "\n")

    # عرض توزيع الفئات
    class_distribution = df_clean[target].value_counts().sort_index()
    log("توزيع الفئات في البيانات:", "INFO")
    for cls_value, cls_count in class_distribution.items():
        pct = (cls_count / len(df_clean)) * 100
        log(f" الفئة {cls_value}: {cls_count} ({pct:.1f}%)", "INFO")

    # =========================================================================
    # 4. إعداد الميزات والهدف
    # =========================================================================
    log("--- المرحلة 4: إعداد الميزات والهدف ---", "INFO")
    X = df_clean[features].copy()
    y = df_clean[target].copy()

    # ترميز الهدف
    y_encoded, le, class_mapping = encode_target(y)

    # =========================================================================
    # 5. تقسيم البيانات
    # =========================================================================
    log("--- المرحلة 5: تقسيم البيانات ---", "INFO")
    if use_temporal_split:
        log("استخدام التقسيم الزمني (Temporal Split)...", "INFO")
        X_train, X_test, y_train, y_test = split_data_temporal(
            df_clean, X, y_encoded,
            test_size=DEFAULT_TEST_SIZE,
            date_column='match_date'
        )
    else:
        log("استخدام التقسيم العشوائي الطبقي (Stratified Random Split)...", "INFO")
        X_train, X_test, y_train, y_test = split_data_random(
            X, y_encoded,
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE
        )

    # عرض توزيع الفئات في مجموعتي التدريب والاختبار
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    log("توزيع الفئات في مجموعة التدريب:", "INFO")
    for cls_enc, cnt in zip(train_unique, train_counts):
        cls_orig = int(le.inverse_transform([cls_enc])[0])
        pct = (cnt / len(y_train)) * 100
        log(f" الفئة {cls_orig}: {cnt} ({pct:.1f}%)", "INFO")
    log("توزيع الفئات في مجموعة الاختبار:", "INFO")
    for cls_enc, cnt in zip(test_unique, test_counts):
        cls_orig = int(le.inverse_transform([cls_enc])[0])
        pct = (cnt / len(y_test)) * 100
        log(f" الفئة {cls_orig}: {cnt} ({pct:.1f}%)", "INFO")

    # =========================================================================
    # 6. إنشاء وتدريب النموذج
    # =========================================================================
    log("--- المرحلة 6: إنشاء وتدريب النموذج ---", "INFO")
    xgb_params = custom_params if custom_params else DEFAULT_XGB_PARAMS.copy()
    model = build_xgb_model(xgb_params)
    model = train_model(model, X_train, y_train, X_test, y_test)

    # التحقق من أن model.classes_ تتطابق مع ما هو متوقع
    model_classes = model.classes_.tolist()
    log(f"فئات النموذج (model.classes_): {model_classes}", "INFO")
    expected_encoded_classes = sorted(le.transform(EXPECTED_CLASSES).tolist())
    if sorted(model_classes) != expected_encoded_classes:
        log(
            f"تحذير: فئات النموذج ({model_classes}) "
            f"لا تتطابق مع الفئات المُرمّزة المتوقعة ({expected_encoded_classes})",
            "WARNING"
        )

    # =========================================================================
    # 7. تقييم الأداء
    # =========================================================================
    log("--- المرحلة 7: تقييم الأداء ---", "INFO")
    evaluation_results = evaluate_model(model, X_test, y_test, le)

    # =========================================================================
    # 8. تحليل أهمية الميزات
    # =========================================================================
    log("--- المرحلة 8: تحليل أهمية الميزات ---", "INFO")
    feature_importance = analyze_feature_importance(model, features)

    # =========================================================================
    # 9. التحقق المتقاطع (اختياري)
    # =========================================================================
    cv_results = None
    if run_cv:
        log("--- المرحلة 9: التحقق المتقاطع ---", "INFO")
        cv_results = run_cross_validation(xgb_params, X, y_encoded, n_folds=cv_folds)
    else:
        log("--- المرحلة 9: تم تخطي التحقق المتقاطع ---", "INFO")

    # =========================================================================
    # 10. حفظ النموذج والبيانات الوصفية
    # =========================================================================
    log("--- المرحلة 10: حفظ النموذج والملفات ---", "INFO")

    # التأكد من وجود مجلد النماذج
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # حفظ النموذج
    model_path = config.MODELS_DIR / "xgboost_model.json"
    model_saved = save_model(model, model_path)
    if not model_saved:
        log("فشل حفظ النموذج. العملية لم تكتمل بنجاح.", "CRITICAL")
        return

    # حفظ البيانات الوصفية
    metadata_path = config.MODELS_DIR / "xgboost_metadata.json"
    save_model_metadata(
        metadata_path=metadata_path,
        features=features,
        le=le,
        class_mapping=class_mapping,
        evaluation_results=evaluation_results,
        feature_importance=feature_importance,
        cv_results=cv_results,
        xgb_params=xgb_params,
        dataset_info=dataset_info,
        model_classes=model_classes,
    )

    # حفظ تقرير التقييم النصي
    report_path = config.MODELS_DIR / "training_report.txt"
    save_evaluation_report(
        report_path=report_path,
        evaluation_results=evaluation_results,
        feature_importance=feature_importance,
        cv_results=cv_results,
    )

    # =========================================================================
    # 11. التحقق النهائي من صحة النموذج المحفوظ
    # =========================================================================
    log("--- المرحلة 11: التحقق النهائي ---", "INFO")
    try:
        # تحميل النموذج المحفوظ والتحقق من عمله
        verification_model = xgb.XGBClassifier()
        verification_model.load_model(str(model_path))

        # اختبار توقع واحد
        sample_input = X_test.iloc[[0]]
        sample_prediction = verification_model.predict(sample_input)
        sample_proba = verification_model.predict_proba(sample_input)

        log("التحقق من النموذج المحفوظ:", "INFO")
        log(f" عينة إدخال: {sample_input.values.tolist()[0]}", "DEBUG")
        log(f" التوقع: {sample_prediction[0]}", "DEBUG")
        log(f" الاحتمالات: {[round(p, 4) for p in sample_proba[0].tolist()]}", "DEBUG")

        # التحقق من تطابق الفئات
        loaded_classes = verification_model.classes_.tolist()
        if loaded_classes == model_classes:
            log("✅ فئات النموذج المحفوظ تتطابق مع الأصل.", "INFO")
        else:
            log(
                f"⚠ فئات النموذج المحفوظ ({loaded_classes}) "
                f"تختلف عن الأصل ({model_classes})",
                "WARNING"
            )

        # مقارنة التوقعات بين النموذج الأصلي والمحفوظ
        original_pred = model.predict(sample_input)
        if np.array_equal(sample_prediction, original_pred):
            log("✅ توقعات النموذج المحفوظ تتطابق مع الأصل.", "INFO")
        else:
            log("⚠ توقعات النموذج المحفوظ تختلف عن الأصل!", "WARNING")
    except Exception as e:
        log(f"⚠ فشل التحقق من النموذج المحفوظ: {e}", "WARNING")

    # =========================================================================
    # ملخص نهائي
    # =========================================================================
    log("", "INFO")
    log("=" * 70, "INFO")
    log("ملخص نتائج التدريب", "INFO")
    log("=" * 70, "INFO")
    log(f" حجم مجموعة البيانات الأصلية: {dataset_info['original_rows']} صف", "INFO")
    log(f" حجم مجموعة البيانات النظيفة: {dataset_info['cleaned_rows']} صف", "INFO")
    log(f" عدد الميزات: {len(features)}", "INFO")
    log(f" حجم مجموعة التدريب: {len(X_train)} صف", "INFO")
    log(f" حجم مجموعة الاختبار: {len(X_test)} صف", "INFO")
    log(f" الدقة على مجموعة الاختبار: {evaluation_results['accuracy']:.2%}", "INFO")
    if cv_results:
        log(
            f" متوسط دقة التحقق المتقاطع: "
            f"{cv_results['mean_accuracy']:.2%} ± {cv_results['std_accuracy']:.2%}",
            "INFO"
        )
    log(f" ملف النموذج: {model_path}", "INFO")
    log(f" ملف البيانات الوصفية: {metadata_path}", "INFO")
    log(f" ملف التقرير: {report_path}", "INFO")
    log("=" * 70, "INFO")
    log("انتهت عملية التدريب بنجاح ✅", "INFO")
    log("=" * 70, "INFO")


# -----------------------------------------------------------------------------
# نقطة الدخول
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="تدريب نموذج XGBoost للتنبؤ بنتائج مباريات كرة القدم"
    )
    parser.add_argument(
        "--temporal-split", action="store_true", default=False,
        help="استخدام التقسيم الزمني بدلاً من العشوائي (يتطلب عمود match_date)"
    )
    parser.add_argument(
        "--no-cv", action="store_true", default=False,
        help="تخطي التحقق المتقاطع (Cross-Validation)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="عدد طيّات التحقق المتقاطع (افتراضي: 5)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=500,
        help="عدد الأشجار في XGBoost (افتراضي: 500)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05,
        help="معدل التعلم (افتراضي: 0.05)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="أقصى عمق للشجرة (افتراضي: 4)"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=30,
        help="عدد الجولات بدون تحسّن قبل التوقف المبكر (افتراضي: 30)"
    )

    args = parser.parse_args()

    # بناء معاملات XGBoost من سطر الأوامر
    custom_xgb_params = DEFAULT_XGB_PARAMS.copy()
    custom_xgb_params['n_estimators'] = args.n_estimators
    custom_xgb_params['learning_rate'] = args.learning_rate
    custom_xgb_params['max_depth'] = args.max_depth
    custom_xgb_params['early_stopping_rounds'] = args.early_stopping

    try:
        run_ml_trainer(
            use_temporal_split=args.temporal_split,
            run_cv=not args.no_cv,
            cv_folds=args.cv_folds,
            custom_params=custom_xgb_params,
        )
    except KeyboardInterrupt:
        log("تم إيقاف التدريب بواسطة المستخدم.", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"خطأ غير متوقع: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)
