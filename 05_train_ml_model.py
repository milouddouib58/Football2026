# 05_train_ml_model.py
# -----------------------------------------------------------------------------
# الوصف:
# يقرأ هذا السكريبت مجموعة البيانات التي تم إنشاؤها (`ml_dataset.csv`)
# ويقوم بتدريب نموذج تعلم آلة (XGBoost Classifier) للتنبؤ بنتيجة المباراة.
# -----------------------------------------------------------------------------
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from common import config
from common.utils import log

def run_ml_trainer():
    log("--- بدء تدريب نموذج تعلم الآلة (XGBoost) ---", "INFO")

    # --- 1. تحميل مجموعة البيانات ---
    try:
        df = pd.read_csv(config.DATA_DIR / "ml_dataset.csv")
    except FileNotFoundError:
        log("ملف ml_dataset.csv غير موجود. يرجى تشغيل 04_feature_generator.py أولاً.", "CRITICAL")
        return

    # --- 2. إعداد البيانات ---
    features = [
        'home_attack', 'away_attack', 'home_defense', 'away_defense',
        'home_elo', 'away_elo', 'elo_diff',
        'home_avg_points', 'away_avg_points'
    ]
    target = 'result'
    X = df[features]
    y = df[target]

    # ترميز الأهداف: (-1, 0, 1) -> (0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    log(f"فئات الهدف بعد الترميز: {dict(zip(le.classes_, le.transform(le.classes_)))}", "DEBUG")

    # تقسيم البيانات (ملاحظة: يفضّل مستقبلاً تقسيم زمني لتجنب تسرب زمني)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    log(f"حجم مجموعة التدريب: {len(X_train)} | حجم مجموعة الاختبار: {len(X_test)}", "INFO")

    # --- 3. تدريب نموذج XGBoost ---
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    log("جاري تدريب النموذج...", "INFO")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)

    # --- 4. تقييم الأداء ---
    log("تقييم أداء النموذج على مجموعة الاختبار...", "INFO")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log(f"دقة النموذج (Accuracy): {accuracy:.2%}", "INFO")
    report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str))
    print("\n--- تقرير التصنيف ---\n", report)

    # --- 5. حفظ النموذج المدرب ---
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    model_path = config.MODELS_DIR / "xgboost_model.json"
    model.save_model(model_path)
    log(f"تم حفظ نموذج تعلم الآلة المدرب في: {model_path}", "INFO")
    log("--- انتهت عملية التدريب بنجاح ---", "INFO")

if __name__ == "__main__":
    run_ml_trainer()
