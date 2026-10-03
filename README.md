# 🏦 End-to-End Credit Risk & Default Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-brightgreen.svg)](https://peps.python.org/pep-0008/)

Endüstri standardı makine öğrenmesi pratikleri (MLOps) kullanılarak geliştirilmiş, bankacılık ve finans sektörü odaklı bir **Kredi Temerrüt Tahmini (Credit Default Risk)** projesidir. Ham müşteri başvuru verilerinden hareketle, kredi geri ödememe riskini olasılıksal olarak tahmin eden modüler bir Scikit-learn Pipeline mimarisi sunar.

---

## 📌 İş Problemi ve Yaklaşım (Business Context)

Bankacılıkta temerrüt (default), bir müşterinin yasal süre içerisinde borcunu ödeyememesi durumudur. Yanlış kredi kararları bankalar için yüksek kredi zarar provizyonu ve finansal kayıp doğururken, aşırı katı kurallar ise potansiyel karlı müşterilerin kaybına yol açar.

Bu projede amaç:
- Başvuru aşamasındaki müşterileri **Güvenli (0)** veya **Riskli / Temerrüt (1)** olarak sınıflandırmak.
- Veri sızıntısını (**Data Leakage**) tamamen engelleyen kapalı devre bir dönüşüm hattı kurmak.
- Üretim (Production) ortamına girmeye hazır, tek formatlı girdiyle çıkarım yapabilen bir boru hattı (`.pkl`) sunmaktır.

---

## 🛠️ Mimari ve Mühendislik Prensipleri

Proje geliştirilirken aşağıdaki standartlara sadık kalınmıştır:
1. **Modüler Tasarım:** Veri alma, modelleme ve çıkarım süreçleri `src/` altında bağımsız betiklere ayrılmıştır.
2. **Feature Engineering Entegrasyonu:** Müşterinin borç/gelir dengesini yansıtan `Loan to Income` metriği hem eğitimde hem de çıkarım modülünde dinamik olarak türetilir.
3. **Data Leakage Önleme:** `ColumnTransformer` ve `Pipeline` kullanılarak eksik veri tamamlama (`SimpleImputer`) ve z-score standardizasyonu (`StandardScaler`) yalnızca eğitim setine (`X_train`) uygulanmış, test seti hiçbir aşamada dönüştürücüleri etkilememiştir.
4. **Sınıf Dengesizliği (Class Imbalance):** Finansal verilerde batık müşterilerin azınlıkta olması sebebiyle modeller `class_weight='balanced'` ile eğitilmiş ve dengeli değerlendirilmiştir.

---

## 📊 Model Performans Metrikleri

Eğitim sürecinde doğrusal (`Logistic Regression`) ve ağaç tabanlı (`Random Forest`) iki algoritma `ROC-AUC` metriği üzerinden karşılaştırılmıştır (%80 Train - %20 Stratified Test):

| Algoritma | ROC-AUC Skoru | Accuracy | Precision (Sınıf 1) | Recall (Sınıf 1) | F1-Skor (Sınıf 1) |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.9873 | %94 | 0.71 | 0.96 | 0.81 |
| **Random Forest (Best)** | **1.0000** | **%100** | **1.00** | **1.00** | **1.00** |

*En yüksek ROC-AUC skorunu üreten **Random Forest Pipeline**, diskte `models/credit_default_pipeline.pkl` konumuna serileştirilmiştir.*

---

## 📂 Proje Dizin Yapısı

```text
credit-risk-prediction/
├── data/                    # Ham ve işlenmiş veriler (.gitignore ile korunur)
├── models/                  # Eğitilmiş serileştirilmiş boru hatları (.pkl)
├── src/                     # Kaynak kod modülleri
│   ├── make_dataset.py      # Veriyi uzak kaynaktan çeken otomatik script
│   ├── train.py             # Feature engineering, pipeline inşası ve model eğitimi
│   └── predict.py           # Canlı ortam çıkarım (inference) ve test modülü
├── .gitignore               # Gereksiz/büyük dosyaları dışlayan git kural seti
├── requirements.txt         # Proje bağımlılıkları ve kütüphane versiyonları
└── README.md                # Kapsamlı proje dokümantasyonu
```

---

## 🚀 Setup & Execution

Follow these steps to run the pipeline locally:

### 1. Clone the Repository
```bash
git clone https://github.com/ErmanDurak/credit-risk-prediction.git
cd credit-risk-prediction
```

### 2. Configure Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fetch the Dataset
```bash
python src/make_dataset.py
```

### 5. Train and Export the Pipeline
```bash
python src/train.py
```

### 6. Run Inference on Test Profiles
```bash
python src/predict.py
```