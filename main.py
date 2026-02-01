import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# تنظیمات مسیرها
DATA_PATH = os.path.join("data", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
MODEL_DIR = 'models'


def train_engine():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    # ۱. بارگذاری بهینه داده‌ها (استفاده از dtypes برای کاهش مصرف رم)
    print("⏳ Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # ۲. پیش‌پردازش سریع
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # ۳. کدگذاری برچسب‌ها
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    joblib.dump(le, f'{MODEL_DIR}/label_encoder.pkl')

    print(f"📦 Detected Categories: {list(le.classes_)}")

    # ۴. جداسازی داده‌ها
    X = df.drop('Label', axis=1)
    y = df['Label']

    # ۵. Feature Selection (انتخاب ویژگی‌های مهم برای افزایش سرعت داشبورد)
    # ترافیک شبکه ویژگی‌های تکراری زیاد دارد، ما ۳۰ تا از بهترین‌ها را برمی‌داریم
    print("🎯 Selecting Top Features for high-speed inference...")
    temp_model = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    temp_model.fit(X, y)

    importances = pd.Series(temp_model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(30).index.tolist()

    # ذخیره لیست ویژگی‌های منتخب (برای اینکه در داشبورد بدانیم چه ستون‌هایی لازم است)
    joblib.dump(top_features, f'{MODEL_DIR}/selected_features.pkl')

    X = X[top_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ۶. آموزش مدل نهایی
    print(f"🚀 Training Final Model on {len(top_features)} optimized features...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,  # جلوگیری از حجیم شدن فایل مدل
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    # ۷. ارزیابی سریع
    preds = model.predict(X_test)
    print(f"\n✅ Training Complete! Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # ۸. ذخیره مدل فشرده شده
    joblib.dump(model, f'{MODEL_DIR}/trained_model.pkl', compress=3)
    print(f"💾 Optimized model saved in '{MODEL_DIR}/' folder.")


if __name__ == "__main__":
    train_engine()