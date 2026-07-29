"""
app.py
------
Telco Customer Churn - Streamlit Uygulamasi

Uc sekme:
  1) Yonetici Dashboardu  -> egitim verisi uzerinden churn/gelir/segment analizleri
  2) Tekil Musteri Tahmini -> tek musteri icin churn olasiligi + CLV
  3) Toplu Tahmin (CSV)    -> yuklenen CSV'deki tum musteriler icin churn + CLV

Calistirmak icin:
    streamlit run app.py

Beklenen dosyalar (churn_model_training.py calistirildiktan sonra otomatik olusur):
    artifacts/final_telco_churn_model.keras
    artifacts/scaler.joblib
    artifacts/original_col_names.joblib
    artifacts/num_cols_upper.joblib
    artifacts/training_data_snapshot.csv
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load
from tensorflow.keras.models import load_model

from utils import map_churn_column, predict_churn_and_clv

ARTIFACT_DIR = "artifacts"

st.set_page_config(
    page_title="Telco Churn & CLV",
    page_icon="📡",
    layout="wide",
)



# ============================================================
# ARTIFACT YUKLEME (cache'lenir, tekrar tekrar diskten okumaz)
# ============================================================


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(ARTIFACT_DIR, "final_telco_churn_model.keras")
    scaler_path = os.path.join(ARTIFACT_DIR, "scaler.joblib")
    cols_path = os.path.join(ARTIFACT_DIR, "original_col_names.joblib")
    num_cols_path = os.path.join(ARTIFACT_DIR, "num_cols_upper.joblib")

    missing = [
        p
        for p in [model_path, scaler_path, cols_path, num_cols_path]
        if not os.path.exists(p)
    ]
    if missing:
        return None

    model = load_model(model_path)
    scaler = load(scaler_path)
    original_columns = load(cols_path)
    num_cols_upper = load(num_cols_path)
    return model, scaler, original_columns, num_cols_upper


@st.cache_data
def load_training_snapshot():
    path = os.path.join(ARTIFACT_DIR, "training_data_snapshot.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def generate_business_insights(df):
    insights = []

    # Contract
    contract = (
        df.groupby("Contract")["IS_CHURN"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    highest_contract = contract.index[0]
    highest_contract_rate = contract.iloc[0]

    insights.append(
        f"📌 {highest_contract} contract customers have the highest churn rate ({highest_contract_rate:.1f}%)."
    )

    # Internet
    internet = (
        df.groupby("InternetService")["IS_CHURN"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    highest_internet = internet.index[0]
    highest_internet_rate = internet.iloc[0]

    insights.append(
        f"🌐 {highest_internet} users show the highest churn rate ({highest_internet_rate:.1f}%)."
    )

    # Payment
    payment = (
        df.groupby("PaymentMethod")["IS_CHURN"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    highest_payment = payment.index[0]

    insights.append(f"💳 {highest_payment} is the riskiest payment method.")

    # Revenue
    monthly_loss = df.loc[
        df["IS_CHURN"] == 1,
        "MonthlyCharges"
    ].sum()

    insights.append(
        f"💰 Estimated monthly revenue at risk: ${monthly_loss:,.0f}"
    )

    # Recommendation
    insights.append(
        "🎯 Recommendation: Target Month-to-Month Fiber Optic customers with long-term contract offers."
    )

    return insights


artifacts = load_artifacts()

st.title("📡 Telco Customer Churn & CLV Platformu")

if artifacts is None:
    st.error(
        "Model dosyalari bulunamadi. Once `python churn_model_training.py --data "
        "telco_customer_churn.csv` komutunu calistirip `artifacts/` klasorunu olusturman gerekiyor."
    )
    st.stop()

model, scaler, original_columns, num_cols_upper = artifacts

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Yönetici Dashboardu",
    "🧑 Tekil Tahmin",
    "📁 Toplu Tahmin",
    "🤖 Model Performansı",
])

# ============================================================
# TAB 1 - YONETICI DASHBOARDU
# ============================================================

with tab1:
    df = load_training_snapshot()

    if df is None:
        st.warning(
            "Dashboard icin egitim verisi anlik goruntusu (`training_data_snapshot.csv`) bulunamadi. "
            "Egitim scriptini tekrar calistirdiginda otomatik olusacak."
        )
    else:
        # BUGFIX: Churn kolonu veri setine gore "Yes"/"No" ya da zaten 0/1
        # sayisal olabilir. map_churn_column ikisini de dogru yorumlar,
        # boylece dashboard'da churn orani hep 0 gozukmez.
        df["IS_CHURN"] = map_churn_column(df)
        churn_rate = df["IS_CHURN"].mean()
        total_customers = len(df)
        monthly_revenue_at_risk = df.loc[
            df["IS_CHURN"] == 1, "MonthlyCharges"
        ].sum()
        avg_tenure = df["tenure"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Musteri", f"{total_customers:,}")
        c2.metric("Churn Orani", f"%{churn_rate * 100:.1f}")
        c3.metric("Kaybedilen Aylik Gelir", f"${monthly_revenue_at_risk:,.0f}")
        c4.metric("Ortalama Tenure (ay)", f"{avg_tenure:.1f}")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            contract_churn = (
                df.groupby("Contract")["IS_CHURN"]
                .mean()
                .mul(100)
                .reset_index(name="Churn Orani (%)")
            )
            fig = px.bar(
                contract_churn,
                x="Contract",
                y="Churn Orani (%)",
                title="Sozlesme Tipine Gore Churn Orani",
                color="Churn Orani (%)",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            internet_churn = (
                df.groupby("InternetService")["IS_CHURN"]
                .mean()
                .mul(100)
                .reset_index(name="Churn Orani (%)")
            )
            fig = px.bar(
                internet_churn,
                x="InternetService",
                y="Churn Orani (%)",
                title="Internet Servis Tipine Gore Churn Orani",
                color="Churn Orani (%)",
                color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🎯 Süper Kampanya Simülasyonu")

        discount = st.slider("İndirim Oranı (%)", 5, 50, 20)

        campaign_success = st.slider("Kampanya Başarı Oranı (%)", 10, 100, 50)

        high_risk = df[df["IS_CHURN"] == 1]

        saved_customers = int(len(high_risk) * campaign_success / 100)

        saved_revenue = (
            high_risk["MonthlyCharges"].sum()
            * campaign_success / 100
            * 12
        )

        campaign_cost = saved_revenue * discount / 100

        net_gain = saved_revenue - campaign_cost

        c1, c2, c3 = st.columns(3)

        c1.metric("Kurtarılacak Müşteri", saved_customers)

        c2.metric("Beklenen Gelir", f"${saved_revenue:,.0f}")

        c3.metric("Net Kazanç", f"${net_gain:,.0f}")

        roi = (net_gain / campaign_cost) * 100 if campaign_cost > 0 else 0

        st.success(f"📈 Kampanya ROI: %{roi:.1f}")

        fig = px.pie(
            names=["Kalan Kazanç", "İndirim Maliyeti"],
            values=[net_gain, campaign_cost],
            title="Kampanya Finansal Etkisi",
        )

        st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            plot_df = df.copy()
            plot_df["Churn Durumu"] = plot_df["IS_CHURN"].map(
                {1: "Churn Oldu", 0: "Kaldi"}
            )
            fig = px.histogram(
                plot_df,
                x="tenure",
                color="Churn Durumu",
                barmode="overlay",
                title="Tenure Dagilimi (Churn'e Gore)",
                nbins=30,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            payment_churn = (
                df.groupby("PaymentMethod")["IS_CHURN"]
                .mean()
                .mul(100)
                .reset_index(name="Churn Orani (%)")
                .sort_values("Churn Orani (%)", ascending=False)
            )
            fig = px.bar(
                payment_churn,
                x="PaymentMethod",
                y="Churn Orani (%)",
                title="Odeme Yontemine Gore Churn Orani",
                color="Churn Orani (%)",
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Aylik Ucret Dagilimi")
        box_df = df.copy()
        box_df["Churn Durumu"] = box_df["IS_CHURN"].map(
            {1: "Churn Oldu", 0: "Kaldi"}
        )
        fig = px.box(
            box_df,
            x="Churn Durumu",
            y="MonthlyCharges",
            color="Churn Durumu",
            title="Churn Durumuna Gore Aylik Ucret",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 - TEKIL MUSTERI TAHMINI
# ============================================================

with tab2:
    st.subheader("Musteri Bilgilerini Gir")

    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Cinsiyet", ["Female", "Male"])
            senior = st.selectbox("Yasli Musteri mi?", ["No", "Yes"])
            partner = st.selectbox("Partneri Var mi?", ["Yes", "No"])
            dependents = st.selectbox(
                "Bakmakla Yukumlu Kisi Var mi?", ["Yes", "No"]
            )
            tenure = st.number_input(
                "Tenure (ay)", min_value=0, max_value=100, value=12
            )

        with c2:
            phone_service = st.selectbox("Telefon Servisi", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Coklu Hat", ["Yes", "No", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Servisi", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Guvenlik", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Yedekleme", ["Yes", "No", "No internet service"]
            )

        with c3:
            device_protection = st.selectbox(
                "Cihaz Koruma", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Teknik Destek", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Sozlesme Tipi", ["Month-to-month", "One year", "Two year"]
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            paperless_billing = st.selectbox("Kagitsiz Fatura", ["Yes", "No"])
        with c5:
            payment_method = st.selectbox(
                "Odeme Yontemi",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with c6:
            monthly_charges = st.number_input(
                "Aylik Ucret ($)", min_value=0.0, value=70.0, step=0.5
            )

        total_charges = st.number_input(
            "Toplam Odenen Ucret ($)",
            min_value=0.0,
            value=float(monthly_charges * max(tenure, 1)),
            step=1.0,
        )

        threshold = st.slider("Churn Karar Esigi", 0.0, 1.0, 0.50, 0.01)

        submitted = st.form_submit_button("Tahmin Et", use_container_width=True)

    if submitted:
        new_customer = pd.DataFrame({
            "customerID": ["MANUAL_INPUT"],
            "gender": [gender],
            "SeniorCitizen": [1 if senior == "Yes" else 0],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        })

        result = predict_churn_and_clv(
            new_customer_df=new_customer,
            model=model,
            original_columns=original_columns,
            scaler=scaler,
            num_cols_upper=num_cols_upper,
            forecast_months=12,
            margin_rate=0.35,
            threshold=threshold,
        )

        row = result.iloc[0]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Olasiligi", f"%{row['CHURN_PROBABILITY_PERCENT']:.1f}")
        r2.metric("Tahmini CLV (12 ay)", f"${row['ESTIMATED_CLV']:,.2f}")
        r3.metric("Risk Segmenti", row["RISK_SEGMENT"])

        if row["PREDICTION"] == "CAYACAK":
            st.error(
                f"⚠️ Sonuc: Musterinin CAYMASI bekleniyor (esik: %{threshold*100:.0f})"
            )
        else:
            st.success(
                f"✅ Sonuc: Musterinin KALMASI bekleniyor (esik: %{threshold*100:.0f})"
            )

        st.progress(float(row["CHURN_PROBABILITY"]))

        st.divider()
        st.subheader("💡 Geri Kazanma Onerileri")
        recommendations = row["RETENTION_RECOMMENDATIONS"]
        for i, rec in enumerate(recommendations, start=1):
            with st.expander(f"{i}. {rec['title']}", expanded=(i == 1)):
                st.write(rec["reason"])


# ============================================================
# TAB 3 - TOPLU TAHMIN (CSV YUKLEME)
# ============================================================

with tab3:
    st.subheader("CSV Yukleyerek Toplu Churn + CLV Tahmini")
    st.caption(
        "Yuklenecek CSV, orijinal Telco veri setiyle ayni kolon yapisina sahip olmali "
        "(Churn kolonu olmadan da calisir)."
    )

    uploaded_file = st.file_uploader("CSV Dosyasi Sec", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)
        st.write(f"{len(bulk_df)} musteri yuklendi.")

        threshold_bulk = st.slider(
            "Churn Karar Esigi", 0.0, 1.0, 0.50, 0.01, key="bulk_threshold"
        )

        if st.button("Toplu Tahmini Calistir", use_container_width=True):
            with st.spinner("Tahminler hesaplaniyor..."):
                bulk_result = predict_churn_and_clv(
                    new_customer_df=bulk_df,
                    model=model,
                    original_columns=original_columns,
                    scaler=scaler,
                    num_cols_upper=num_cols_upper,
                    forecast_months=12,
                    margin_rate=0.35,
                    threshold=threshold_bulk,
                )

            st.divider()
            b1, b2, b3 = st.columns(3)
            b1.metric(
                "Ortalama Churn Olasiligi",
                f"%{bulk_result['CHURN_PROBABILITY_PERCENT'].mean():.1f}",
            )
            b2.metric(
                "Risk Altindaki Toplam CLV",
                f"${bulk_result.loc[bulk_result['PREDICTION']=='CAYACAK', 'ESTIMATED_CLV'].sum():,.0f}",
            )
            b3.metric(
                "Yuksek Riskli Musteri Sayisi",
                int((bulk_result["RISK_SEGMENT"] == "Yuksek Risk").sum()),
            )

            fig = px.histogram(
                bulk_result,
                x="CHURN_PROBABILITY_PERCENT",
                color="RISK_SEGMENT",
                title="Churn Olasilik Dagilimi",
                nbins=30,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Oneri listesinden okunabilir kolonlar turet (tablo ve CSV icin)
            bulk_result["TOP_ONERI"] = bulk_result[
                "RETENTION_RECOMMENDATIONS"
            ].apply(lambda recs: recs[0]["title"] if recs else "")
            bulk_result["ONERI_DETAY"] = bulk_result[
                "RETENTION_RECOMMENDATIONS"
            ].apply(
                lambda recs: " | ".join(
                    f"{r['title']}: {r['reason']}" for r in recs
                )
            )

            display_cols = [
                c
                for c in [
                    "customerID",
                    "CHURN_PROBABILITY_PERCENT",
                    "RISK_SEGMENT",
                    "ESTIMATED_CLV",
                    "PREDICTION",
                    "TOP_ONERI",
                ]
                if c in bulk_result.columns
            ]
            st.dataframe(
                bulk_result[display_cols].sort_values(
                    "CHURN_PROBABILITY_PERCENT", ascending=False
                ),
                use_container_width=True,
            )

            st.divider()
            st.subheader("💡 En Cok Onerilen Aksiyonlar (Riskli Musteriler)")
            st.caption(
                "Dusuk Risk disindaki musteriler icin hangi aksiyon kac musteriyi "
                "ilgilendiriyor - kampanya onceliklendirmesi icin kullanabilirsin."
            )

            at_risk = bulk_result[bulk_result["RISK_SEGMENT"] != "Dusuk Risk"]
            if len(at_risk) > 0:
                oneri_counts = (
                    at_risk["TOP_ONERI"].value_counts().reset_index()
                )
                oneri_counts.columns = ["Oneri", "Etkilenen Musteri Sayisi"]

                fig2 = px.bar(
                    oneri_counts,
                    x="Etkilenen Musteri Sayisi",
                    y="Oneri",
                    orientation="h",
                    title="Aksiyon Bazinda Etkilenen Musteri Sayisi",
                    color="Etkilenen Musteri Sayisi",
                    color_continuous_scale="Teal",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(
                    "Yuksek/Orta riskli musteri bulunamadi - harika bir tablo!"
                )

            st.divider()
            st.subheader("🔝 En Riskli 10 Musteri - Detayli Onerileri")
            top10 = bulk_result.sort_values(
                "CHURN_PROBABILITY_PERCENT", ascending=False
            ).head(10)

            for _, cust_row in top10.iterrows():
                cust_id = cust_row.get("customerID", "N/A")
                header = (
                    f"{cust_id} — %{cust_row['CHURN_PROBABILITY_PERCENT']:.1f} churn riski "
                    f"— CLV ${cust_row['ESTIMATED_CLV']:,.0f}"
                )
                with st.expander(header):
                    for i, rec in enumerate(
                        cust_row["RETENTION_RECOMMENDATIONS"], start=1
                    ):
                        st.markdown(f"**{i}. {rec['title']}**")
                        st.write(rec["reason"])

            csv_export = bulk_result.drop(
                columns=["RETENTION_RECOMMENDATIONS"]
            ).copy()
            csv_bytes = csv_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Sonuclari CSV Olarak Indir",
                data=csv_bytes,
                file_name="churn_clv_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# TAB 4 - MODEL PERFORMANSI
# ============================================================

with tab4:
    st.header("Model Performans Karşılaştırması")

    comparison_path = os.path.join(ARTIFACT_DIR, "model_comparison.csv")

    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)

        st.dataframe(comparison, use_container_width=True)

        fig = px.bar(
            comparison,
            x="Model",
            y="AUC",
            color="Model",
            title="Model AUC Karşılaştırması",
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            comparison,
            x="Model",
            y="Recall",
            color="Model",
            title="Model Recall Karşılaştırması",
        )

        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
            comparison,
            x="Model",
            y="Precision",
            color="Model",
            title="Model Precision Karşılaştırması",
        )

        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Önce modeli yeniden eğitiniz.")