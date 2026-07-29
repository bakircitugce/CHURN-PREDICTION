"""
churn_model_training.py
------------------------
Telco Customer Churn modelini egitir ve Streamlit uygulamasinin kullanacagi
artifact'lari (model, scaler, kolon listeleri) 'artifacts/' klasorune kaydeder.

Duzeltilen hatalar (orijinal scripte kiyasla):
1. Hardcoded Windows yolu kaldirildi -> komut satirindan / varsayilan dosya adindan okunuyor.
2. feature_engineering ve grab_col_names artik utils.py'den import ediliyor (kod tekrari yok).
3. Sifira bolme koruma (NEW_Increase, NEW_AVG_Service_Fee) utils.py icinde.
4. F1-score hesaplamasinda ZeroDivisionError koruma eklendi.
5. Tum artifact'lar 'artifacts/' klasorune, tutarli isimlerle kaydediliyor.
6. random_state=46 tum bilesenlerde (seed, split) tutarli hale getirildi.

Calistirmak icin:
    python churn_model_training.py --data telco_customer_churn.csv
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from utils import data_preprocessing_train

SEED = 46
ARTIFACT_DIR = "artifacts"


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main(data_path, epochs, batch_size):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    set_seeds()

    print(f"Veri okunuyor: {data_path}")
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)

    X, y, num_cols_upper = data_preprocessing_train(df)
    print("X shape:", X.shape, "| Churn orani:", round(float(y.mean()), 4))

    scaler = MinMaxScaler()
    X[num_cols_upper] = scaler.fit_transform(X[num_cols_upper])

    dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    dump(X.columns.tolist(), os.path.join(ARTIFACT_DIR, "original_col_names.joblib"))
    dump(num_cols_upper, os.path.join(ARTIFACT_DIR, "num_cols_upper.joblib"))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print("Train:", X_train.shape, "| Val:", X_val.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train.astype("float32"), y_train.astype("float32"))
    ).shuffle(buffer_size=len(X_train)).batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val.astype("float32"), y_val.astype("float32"))
    ).batch(batch_size)

    # class weights - imbalanced churn hedefi icin
    class_0_count = int((y_train == 0).sum())
    class_1_count = int((y_train == 1).sum())
    total = len(y_train)
    class_weights = {
        0: total / (2 * class_0_count),
        1: total / (2 * class_1_count),
    }
    print("Class weights:", class_weights)

    model = build_model(X_train.shape[1])
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_auc", patience=20, verbose=1,
        restore_best_weights=True, mode="max"
    )
    checkpoint_path = os.path.join(ARTIFACT_DIR, "best_telco_churn_model.keras")
    model_checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_auc", verbose=1,
        save_best_only=True, mode="max"
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weights,
    )

    val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_ds, verbose=0)
    denom = (val_prec + val_rec)
    f1 = 2 * val_prec * val_rec / denom if denom > 0 else 0.0

    print("\n=== VALIDATION METRICS ===")
    print(f"Loss:      {val_loss:.4f}")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")
    print(f"AUC:       {val_auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    y_pred_prob = model.predict(X_val.astype("float32"), verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    comparison = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)

    pred = lr.predict(X_val)
    prob = lr.predict_proba(X_val)[:, 1]

    comparison.append([
        "Logistic Regression",
        accuracy_score(y_val, pred),
        precision_score(y_val, pred),
        recall_score(y_val, pred),
        roc_auc_score(y_val, prob)
    ])

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_val)
    prob = rf.predict_proba(X_val)[:, 1]

    comparison.append([
        "Random Forest",
        accuracy_score(y_val, pred),
        precision_score(y_val, pred),
        recall_score(y_val, pred),
        roc_auc_score(y_val, prob)
    ])

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=SEED
    )

    xgb_model.fit(X_train, y_train)

    pred = xgb_model.predict(X_val)
    prob = xgb_model.predict_proba(X_val)[:, 1]

    comparison.append([
        "XGBoost",
        accuracy_score(y_val, pred),
        precision_score(y_val, pred),
        recall_score(y_val, pred),
        roc_auc_score(y_val, prob)
    ])

    # ANN
    comparison.append([
        "ANN",
        val_acc,
        val_prec,
        val_rec,
        val_auc
    ])

    comparison_df = pd.DataFrame(
        comparison,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "AUC"
        ]
    )

    print(comparison_df)

    comparison_df.to_csv(
        os.path.join(ARTIFACT_DIR, "model_comparison.csv"),
        index=False
    )

    final_model_path = os.path.join(ARTIFACT_DIR, "final_telco_churn_model.keras")
    model.save(final_model_path)
    print(f"\nModel kaydedildi: {final_model_path}")
    print(f"Artifact klasoru: {os.path.abspath(ARTIFACT_DIR)}")

    # Streamlit dashboard'da kullanilmak uzere egitimde kullanilan ham veriyi de sakla
    df.to_csv(os.path.join(ARTIFACT_DIR, "training_data_snapshot.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="telco_customer_churn.csv",
        help="Telco churn CSV dosyasinin yolu"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.data, args.epochs, args.batch_size)