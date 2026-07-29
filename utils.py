"""
utils.py
--------
Egitim scripti (churn_model_training.py) ve Streamlit uygulamasi (app.py)
tarafindan ORTAK kullanilan yardimci fonksiyonlar.

Ayni kodu iki yerde kopyalamamak icin tum feature engineering / preprocessing
mantigi burada tutulur. Bir sey degistirmen gerekirse tek yerden degistirirsin.
"""

import numpy as np
import pandas as pd


# ============================================================
# KOLON TIPI TESPITI
# ============================================================

def _find_id_column(dataframe):
    """customerID kolonunu buyuk/kucuk harf farkina bakmadan bulur (veya None)."""
    for col in dataframe.columns:
        if col.strip().lower() == "customerid":
            return col
    return None


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Kategorik, numerik ve kardinal kolonlari ayirir."""

    id_col = _find_id_column(dataframe)

    cat_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtypes == "O"
    ]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and dataframe[col].dtypes != "O"
    ]

    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # BUGFIX: musteri ID kolonu sayisal formatta olsa bile (ornegin 1,2,3...)
    # asla "numerik feature" olarak ele alinmamali - ne cat_cols ne num_cols'a girmeli.
    if id_col is not None:
        cat_cols = [c for c in cat_cols if c != id_col]
        num_cols = [c for c in num_cols if c != id_col]
        if id_col not in cat_but_car:
            cat_but_car = cat_but_car + [id_col]

    return cat_cols, num_cols, cat_but_car


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(dataframe):
    dataframe = dataframe.copy()

    # TotalCharges bazen bosluk karakteri iceriyor (IBM Telco datasetinde bilinen bir durum)
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
    total_charges_median = dataframe["TotalCharges"].median()
    if pd.isna(total_charges_median):
        total_charges_median = 0
    dataframe["TotalCharges"] = dataframe["TotalCharges"].fillna(total_charges_median)

    bins = [-1, 12, 24, 36, 48, 60, 72]
    labels = ["0-1 Year", "1-2 Year", "2-3 Year", "3-4 Year", "4-5 Year", "5-6 Year"]
    dataframe["NEW_TENURE_YEAR"] = pd.cut(dataframe["tenure"], bins=bins, labels=labels)
    # 72 ay ustu (varsa) veya kapsam disi degerler icin guvenlik agi
    dataframe["NEW_TENURE_YEAR"] = dataframe["NEW_TENURE_YEAR"].astype(object).fillna("5-6 Year")

    dataframe["NEW_Engaged"] = dataframe["Contract"].apply(
        lambda x: 1 if x in ["One year", "Two year"] else 0
    )

    dataframe["NEW_noProt"] = dataframe.apply(
        lambda x: 1 if (
            x["OnlineBackup"] != "Yes"
            or x["DeviceProtection"] != "Yes"
            or x["TechSupport"] != "Yes"
        ) else 0,
        axis=1
    )

    dataframe["NEW_Young_Not_Engaged"] = dataframe.apply(
        lambda x: 1 if (x["NEW_Engaged"] == 0 and x["SeniorCitizen"] == 0) else 0,
        axis=1
    )

    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    dataframe["NEW_TotalServices"] = (dataframe[service_cols] == "Yes").sum(axis=1)

    dataframe["NEW_FLAG_ANY_STREAMING"] = dataframe.apply(
        lambda x: 1 if (x["StreamingTV"] == "Yes" or x["StreamingMovies"] == "Yes") else 0,
        axis=1
    )

    dataframe["NEW_FLAG_AutoPayment"] = dataframe["PaymentMethod"].apply(
        lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0
    )

    # --- Bolme islemlerinde sifira bolme koruma (BUGFIX) ---
    safe_tenure = dataframe["tenure"].replace(0, np.nan)
    dataframe["NEW_AVG_Charges"] = dataframe["TotalCharges"] / (dataframe["tenure"] + 1)

    safe_monthly = dataframe["MonthlyCharges"].replace(0, np.nan)
    dataframe["NEW_Increase"] = (dataframe["NEW_AVG_Charges"] / safe_monthly).fillna(1.0)

    dataframe["NEW_AVG_Service_Fee"] = (
        dataframe["MonthlyCharges"] / (dataframe["NEW_TotalServices"] + 1)
    )

    del safe_tenure, safe_monthly

    return dataframe


# ============================================================
# EGITIM ICIN PREPROCESSING
# ============================================================

def map_churn_column(dataframe):
    """
    Churn kolonunu 0/1'e cevirir. Buyuk/kucuk harf ve bosluk farklarina
    dayanikli calisir, ayrica veri zaten 0/1 sayisal ise dogrudan kullanir.
    Eslesmeyen deger varsa NaN uretip sessizce devam etmek yerine ACIK bir
    hata firlatir (BUGFIX - eskiden churn orani sessizce NaN/0 cikiyordu).

    Hem egitim scripti hem de Streamlit dashboard'u tarafindan kullanilir,
    boylece ikisi de AYNI mantikla Churn'u yorumlar.
    """
    raw = dataframe["Churn"]

    if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
        return raw.astype(int)

    normalized = raw.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    mapped = normalized.map(mapping)

    if mapped.isna().any():
        bad_values = sorted(raw[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            "Churn kolonunda beklenmeyen deger(ler) bulundu: "
            f"{bad_values}. Beklenen degerler: 'Yes'/'No' (veya 0/1, True/False). "
            "Datasetindeki Churn kolonunun icerigini kontrol et."
        )

    return mapped.astype(int)


def data_preprocessing_train(dataframe):
    dataframe = dataframe.copy()
    dataframe = feature_engineering(dataframe)

    dataframe["Churn"] = map_churn_column(dataframe)

    id_col = _find_id_column(dataframe)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5)

    if "Churn" in cat_cols:
        cat_cols.remove("Churn")
    if id_col is not None and id_col in cat_cols:
        cat_cols.remove(id_col)

    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=True, dtype=int)
    dataframe.columns = [col.replace(" ", "_").upper() for col in dataframe.columns]

    num_cols_upper = [col.replace(" ", "_").upper() for col in num_cols]

    y = dataframe["CHURN"]
    drop_cols = ["CHURN"]
    if id_col is not None and id_col.upper() in dataframe.columns:
        drop_cols.append(id_col.upper())
    X = dataframe.drop(drop_cols, axis=1)

    return X, y, num_cols_upper


# ============================================================
# TAHMIN ICIN PREPROCESSING (train sirasinda uretilen kolonlarla hizalar)
# ============================================================

def data_preprocess_prediction(dataframe, original_columns, scaler, num_cols_upper):
    dataframe = dataframe.copy()
    dataframe = feature_engineering(dataframe)

    id_col = _find_id_column(dataframe)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5)
    if id_col is not None and id_col in cat_cols:
        cat_cols.remove(id_col)

    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=True, dtype=int)
    dataframe.columns = [col.replace(" ", "_").upper() for col in dataframe.columns]

    if id_col is not None and id_col.upper() in dataframe.columns:
        X = dataframe.drop([id_col.upper()], axis=1)
    else:
        X = dataframe.copy()

    # Egitimde olup burada olmayan kolonlari 0 ile doldur (unseen category / tekil satir)
    for col in original_columns:
        if col not in X.columns:
            X[col] = 0

    # Egitimde olmayip burada olusan fazladan kolonlari at, sirayi hizala
    X = X[original_columns]

    X[num_cols_upper] = scaler.transform(X[num_cols_upper])

    return X.astype("float32")


# ============================================================
# CHURN + CLV TAHMINI
# ============================================================

def predict_churn_and_clv(
    new_customer_df,
    model,
    original_columns,
    scaler,
    num_cols_upper,
    forecast_months=12,
    margin_rate=1.00,
    threshold=0.50
):
    processed = data_preprocess_prediction(
        new_customer_df, original_columns, scaler, num_cols_upper
    )

    churn_probabilities = model.predict(processed, verbose=0)

    result_df = new_customer_df.copy()
    result_df["CHURN_PROBABILITY"] = churn_probabilities.flatten()
    result_df["CHURN_PROBABILITY_PERCENT"] = result_df["CHURN_PROBABILITY"] * 100
    result_df["RETENTION_PROBABILITY"] = 1 - result_df["CHURN_PROBABILITY"]

    result_df["ESTIMATED_CLV"] = (
        result_df["MonthlyCharges"]
        * result_df["RETENTION_PROBABILITY"]
        * forecast_months
        * margin_rate
    )

    result_df["PREDICTION"] = result_df["CHURN_PROBABILITY"].apply(
        lambda x: "CAYACAK" if x >= threshold else "KALACAK"
    )

    # Basit bir risk segmenti - dashboard'da kullanmak icin faydali
    def risk_segment(p):
        if p >= 0.7:
            return "Yuksek Risk"
        elif p >= 0.4:
            return "Orta Risk"
        else:
            return "Dusuk Risk"

    result_df["RISK_SEGMENT"] = result_df["CHURN_PROBABILITY"].apply(risk_segment)

    result_df["RETENTION_RECOMMENDATIONS"] = result_df.apply(
        lambda row: generate_retention_recommendations(row, row["RISK_SEGMENT"]),
        axis=1
    )

    return result_df


# ============================================================
# GERI KAZANMA (RETENTION) ONERI MOTORU
# ============================================================

def generate_retention_recommendations(row, risk_segment):
    """
    Musterinin ozelliklerine bakarak kural tabanli, somut geri kazanma
    onerileri uretir. Bu bir "AI tahmini" degil, is bilgisiyle (domain
    knowledge) yazilmis basit ama etkili bir kural motorudur - churn
    nedenine gore farkli aksiyon onerir.

    Dondurulen: [{"title": ..., "reason": ...}, ...] - oncelik sirali,
    en fazla 4 oneri.
    """

    def get(col, default=None):
        return row[col] if col in row and pd.notna(row[col]) else default

    recs = []

    if risk_segment == "Dusuk Risk":
        return [{
            "title": "Standart Sadakat Iletisimi",
            "reason": "Churn riski dusuk - ozel bir mudahaleye gerek yok, "
                      "duzenli memnuniyet anketleriyle takip yeterli."
        }]

    contract = get("Contract")
    payment_method = get("PaymentMethod")
    tenure = get("tenure", 0)
    monthly_charges = get("MonthlyCharges", 0)
    online_security = get("OnlineSecurity")
    tech_support = get("TechSupport")
    internet_service = get("InternetService")
    senior = get("SeniorCitizen", 0)
    streaming_tv = get("StreamingTV")
    streaming_movies = get("StreamingMovies")
    partner = get("Partner")
    dependents = get("Dependents")

    # 1) Sozlesme tipi - en guclu churn suruculerinden biri
    if contract == "Month-to-month":
        recs.append({
            "title": "Sozlesme Yukseltme Teklifi",
            "reason": "Ay-ay sozlesmesi olan musterilerde churn riski en yuksek. "
                      "1 yillik sozlesmeye gecerse %10-15 indirim veya 1 ay bedava "
                      "gibi bir tesvik sunulmasi onerilir."
        })

    # 2) Odeme yontemi - electronic check genelde en riskli grup
    if payment_method == "Electronic check":
        recs.append({
            "title": "Otomatik Odemeye Gecis Tesviki",
            "reason": "Elektronik cek ile odeyen musterilerde churn orani genelde "
                      "daha yuksek cikiyor. Otomatik banka/kredi karti odemesine "
                      "gecerse kucuk bir indirim veya odul puani sunulabilir."
        })

    # 3) Destek/guvenlik hizmeti eksikligi
    if internet_service in ["DSL", "Fiber optic"] and (
        online_security != "Yes" or tech_support != "Yes"
    ):
        recs.append({
            "title": "Ucretsiz Deneme: Guvenlik / Teknik Destek Paketi",
            "reason": "Online Guvenlik ve Teknik Destek hizmeti olmayan musteriler "
                      "sorun yasadiginda daha kolay ayriliyor. 1-3 ay ucretsiz "
                      "deneme sunmak baglanmayi artirabilir."
        })

    # 4) Yeni musteri (dusuk tenure) - onboarding riski
    if tenure < 6:
        recs.append({
            "title": "Karsilama / Onboarding Aramasi",
            "reason": "Musteri henuz yeni (6 aydan az). Ilk aylardaki deneyim "
                      "kritik - kisisel bir karsilama aramasi veya kullanim "
                      "rehberligi churn'u onemli olcude azaltabilir."
        })

    # 5) Yuksek fatura + yuksek risk -> fiyat hassasiyeti
    if monthly_charges > 80 and risk_segment == "Yuksek Risk":
        recs.append({
            "title": "Kisisellestirilmis Indirim / Paket Kucultme Secenegi",
            "reason": "Aylik ucret yuksek ve churn riski yuksek - fiyat "
                      "hassasiyeti olabilir. Ozel indirim ya da ihtiyaca gore "
                      "daha kucuk bir paket secenegi sunulmasi onerilir."
        })

    # 6) Yasli musteri - kullanim kolayligi / destek
    if senior == 1 or senior == "1" or senior == "Yes":
        recs.append({
            "title": "Basitlestirilmis Destek Hatti",
            "reason": "Yasli musteri segmentinde kullanim zorlugu churn nedeni "
                      "olabilir. Oncelikli telefon destegi veya yerinde kurulum "
                      "yardimi sunulmasi onerilir."
        })

    # 7) Streaming hizmeti var ama bagli degil - paket birlestirme firsati
    if (streaming_tv == "Yes" or streaming_movies == "Yes") and contract == "Month-to-month":
        recs.append({
            "title": "Paket Birlestirme (Bundle) Teklifi",
            "reason": "Streaming hizmetlerini kullanan ama ay-ay sozlesmesi olan "
                      "musterilere, sozlesmeye baglayan avantajli bir bundle "
                      "paketi sunulabilir."
        })

    # 8) Yalniz yasayan / bagliligi dusuk musteri
    if partner == "No" and dependents == "No" and contract == "Month-to-month":
        recs.append({
            "title": "Sadakat Programina Davet",
            "reason": "Partner/bakmakla yukumlu kisisi olmayan, ay-ay sozlesmeli "
                      "musterilerde marka bagliligi genelde daha dusuk. Sadakat "
                      "puani veya arkadasini getir programina davet edilebilir."
        })

    if not recs:
        recs.append({
            "title": "Genel Memnuniyet Aramasi",
            "reason": "Belirgin bir risk faktoru tespit edilmedi, ancak model "
                      "riskli gorduyu icin genel bir memnuniyet aramasi yapilmasi "
                      "onerilir."
        })

    return recs[:4]
