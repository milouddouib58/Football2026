# -----------------------------------------------------------------------------
# 05_train_ml_model.py
# -----------------------------------------------------------------------------
# الوصف:
#   يقرأ هذا السكريبت مجموعة البيانات التي تم إنشاؤها (`ml_dataset.csv`)
#   ويقوم بتدريب نموذج تعلم آلة (XGBoost Classifier) للتنبؤ بنتيجة المباراة
#   (فوز المضيف، تعادل، فوز الضيف).
#   في النهاية، يقوم بحفظ النموذج المدرب في ملف لاستخدامه لاحقًا.
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

    # --- 2. إعداد البيانات للتدريب ---
    # تحديد الميزات (X) والهدف (y)
    features = [
        'home_attack', 'away_attack', 'home_defense', 'away_defense',
        'home_elo', 'away_elo', 'elo_diff',
        'home_avg_points', 'away_avg_points'
    ]
    target = 'result'

    X = df[features]
    y = df[target]

    # XGBoost يتطلب أن تكون التصنيفات (labels) تبدأ من 0 (0, 1, 2)
    # حاليًا هي (-1, 0, 1). سنقوم بتحويلها.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # الآن: 0 -> فوز الضيف (-1), 1 -> تعادل (0), 2 -> فوز المضيف (1)
    
    log(f"فئات الهدف بعد الترميز: {dict(zip(le.classes_, le.transform(le.classes_)))}", "DEBUG")

    # تقسيم البيانات إلى مجموعات تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    log(f"حجم مجموعة التدريب: {len(X_train)} | حجم مجموعة الاختبار: {len(X_test)}", "INFO")

    # --- 3. تدريب نموذج XGBoost ---
    model = xgb.XGBClassifier(
        objective='multi:softprob', # للتصنيف متعدد الفئات مع إخراج احتمالات
        num_class=3,              # عدد الفئات (فوز/تعادل/خسارة)
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,         # يمكن زيادة هذا الرقم للحصول على دقة أفضل
        learning_rate=0.1,
        max_depth=3
    )

    log("جاري تدريب النموذج...", "INFO")
    model.fit(X_train, y_train)

    # --- 4. تقييم أداء النموذج ---
    log("تقييم أداء النموذج على مجموعة الاختبار...", "INFO")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log(f"**دقة النموذج (Accuracy): {accuracy:.2%}**", "INFO")
    
    # عرض تقرير تصنيف مفصل
    report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str))
    print("\n--- تقرير التصنيف ---\n", report)

    # --- 5. حفظ النموذج المدرب ---
    config.MODELS_DIR.mkdir(exist_ok=True)
    model_path = config.MODELS_DIR / "xgboost_model.json"
    model.save_model(model_path)

    log(f"تم حفظ نموذج تعلم الآلة المدرب في: {model_path}", "INFO")
    log("--- انتهت عملية التدريب بنجاح ---", "INFO")


if __name__ == "__main__":
    run_ml_trainer()

