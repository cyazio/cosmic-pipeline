# 🌌 Cosmic Data Extraction & Processing Pipeline — v10

> **TUA Astro Hackathon 2026** | Real-time space weather analysis platform for the May 2024 G5 Geomagnetic Storm

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Dash](https://img.shields.io/badge/Plotly_Dash-2.x-00CC96?style=flat&logo=plotly&logoColor=white)](https://dash.plotly.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![NOAA](https://img.shields.io/badge/Data-NOAA_DSCOVR-0072B5?style=flat)](https://www.swpc.noaa.gov)
[![NASA](https://img.shields.io/badge/Data-NASA_DONKI-FC3D21?style=flat)](https://kauai.ccmc.gsfc.nasa.gov/DONKI)

---

## 📡 Olay / Event

| | |
|---|---|
| **Olay** | Mayıs 2024 G5 Jeomanyetik Fırtınası |
| **X8.7 Flare** | 14 Mayıs 2024, 16:51 UTC — Solar Cycle 25'in en güçlü patlaması |
| **G5 Fırtınası** | 10–11 Mayıs 2024 — Dünya genelinde kutup ışığı (Samsun, Türkiye dahil) |
| **Kp-index** | Kp = 9 (maksimum) |

---

## 🚀 Özellikler / Features

### Gerçek Zamanlı İzleme
- **NOAA DSCOVR API** — 60 saniyede bir güneş rüzgarı plasma ve manyetometre verisi
- **Canlı Kp hesabı** — Newell (2007) fiziksel modeli ile anlık hesaplama
- **Fallback sistemi** — API kesilirse son geçerli veri otomatik devreye girer
- **API durum göstergesi** — Dashboard'da bağlantı durumu canlı takibi

### Anomali Tespiti
- **Isolation Forest (IF)** — Yüksek boyutlu uzayda anomali puanlaması
- **Local Outlier Factor (LOF)** — Komşuluk tabanlı yerel anomali tespiti
- **MAD Z-Score** — Medyana dayalı robust istatistiksel eşikleme
- Ensemble yaklaşımı ile yanlış pozitif oranı minimize edilmiştir

### Makine Öğrenmesi — BiLSTM
- **Çok değişkenli giriş**: Bz, Bt, güneş rüzgarı hızı, yoğunluk
- **Hedef**: Kp-index 3 saatlik tahmini
- **Model kaydetme/yükleme**: TensorFlow SavedModel formatı (TF 2.x uyumlu)
- **5-Fold Stratified Cross-Validation** ile F1 güvenilirliği doğrulanmış

### Uydu Risk Analizi
- **Baker et al. (2018)** Van Allen kuşağı modeline dayalı risk skoru
- **LEO** (< 700 km) ve **GEO** (35.786 km) yörüngeleri için ayrı risk hesabı
- Türk uzay varlıkları için gerçek zamanlı risk kartları

### Dashboard (Plotly Dash)
8 interaktif sekme:

| Sekme | İçerik |
|-------|--------|
| ⚡ Gerçek Zamanlı | Canlı telemetri, normalizasyon, alarm akışı |
| 🛸 NASA DONKI | Flare, CME ve jeomanyetik fırtına arşivi |
| 🔬 Analiz | Ham / temizlenmiş / referans veri karşılaştırması |
| 🤖 Algoritmalar | IF + LOF anomali skoru ve eşik görselleştirmesi |
| 🧠 BiLSTM | Model tahmin sonuçları ve CV metrikleri |
| 📈 Kp-Index | Newell modeli ile hesaplanan Kp zaman serisi |
| 🌍 Olay Analizi | G5 fırtınası detaylı incelemesi |
| 🛡 Risk & Uyarı | Uydu risk skorları ve alarm log tablosu |

---

## 📊 Veri Kaynakları / Data Sources

| Kaynak | Tür | İçerik |
|--------|-----|--------|
| [NOAA DSCOVR](https://www.swpc.noaa.gov/products/real-time-solar-wind) | Gerçek zamanlı | Plasma (hız, yoğunluk, sıcaklık) + Manyetometre (Bx, By, Bz, Bt) |
| [NASA DONKI FLR](https://kauai.ccmc.gsfc.nasa.gov/DONKI) | Arşiv | Solar flare kayıtları (sınıf, zaman, konum) |
| [NASA DONKI GST](https://kauai.ccmc.gsfc.nasa.gov/DONKI) | Arşiv | Jeomanyetik fırtına başlangıç/bitiş ve Kp değerleri |
| [NASA DONKI CME](https://kauai.ccmc.gsfc.nasa.gov/DONKI) | Arşiv | Koronal kütle fırlatma hızı ve açısal genişlik |
| Newell et al. (2007) | Model | `dΦ/dt = v^(4/3) · Bt^(2/3) · sin^(8/3)(θ/2)` |

---

## 🛠 Kurulum / Installation

### Gereksinimler
- Python 3.9+
- NASA API Key (isteğe bağlı — `DEMO_KEY` ile de çalışır)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/kullanici-adi/cosmic-pipeline.git
cd cosmic-pipeline

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. (İsteğe bağlı) NASA API key tanımla
echo "NASA_API_KEY=your_key_here" > .env

# 4. Çalıştır
python pipeline_v10.py

# 5. Tarayıcıda aç
# → http://127.0.0.1:8050
```

### `requirements.txt`

```
numpy
pandas
requests
scipy
scikit-learn
tensorflow
plotly
dash
python-dotenv
```

---

## 🏗 Proje Yapısı / Project Structure

```
cosmic-pipeline/
│
├── pipeline_v10.py          # Ana uygulama
├── requirements.txt         # Python bağımlılıkları
├── .env.example             # API key şablonu
│
├── bilstm_solar_v7/         # Kaydedilmiş BiLSTM modeli (anomali)
├── bilstm_kp_v7/            # Kaydedilmiş BiLSTM modeli (Kp tahmini)
│
└── README.md
```

---

## 🔬 Metodoloji / Methodology

```
Ham Veri (NOAA/NASA)
    │
    ├─▶ Veri Temizleme (interpolasyon + outlier kaldırma)
    │       └─▶ MAE / RMSE metrikleri ile doğrulama
    │
    ├─▶ Anomali Tespiti
    │       ├─▶ Isolation Forest (unsupervised)
    │       ├─▶ Local Outlier Factor
    │       └─▶ MAD Z-Score → Ensemble karar
    │
    ├─▶ Özellik Mühendisliği
    │       └─▶ Bz, Bt, v, n → MinMaxScaler → BiLSTM giriş dizisi
    │
    ├─▶ BiLSTM Tahmin
    │       ├─▶ Kp-index (3 saatlik pencere)
    │       └─▶ 5-Fold CV → F1 / Precision / Recall
    │
    └─▶ Risk Skoru
            ├─▶ LEO: risk = base + 0.08 × Kp
            └─▶ GEO: risk = base + 0.11 × Kp
```

---

## 📈 Performans / Performance

| Metrik | Değer |
|--------|-------|
| BiLSTM Kp F1 (CV) | ~0.82 |
| Anomali Precision | ~0.79 |
| Anomali Recall | ~0.85 |
| Temizleme Doğruluğu | > %90 (sensör başına) |

> Değerler 5-Fold CV ortalamasıdır; gerçek API verisiyle değişkenlik gösterebilir.

---

## 🌍 Türkiye Bağlantısı

Mayıs 2024 G5 fırtınası sırasında Samsun, Türkiye'den **kutup ışığı (aurora borealis)** gözlemlendi. Bu proje, o tarihi olayın kapsamlı bir veri analizi platformu sunar ve Türk uzay varlıklarına yönelik gerçek zamanlı risk skoru hesaplar.

---

## 👥 Takım / Team

**TUA Astro Hackathon 2026** — Türkiye Uzay Ajansı

---

## 📄 Lisans / License

MIT License — Detaylar için `LICENSE` dosyasına bakınız.

---

## 📚 Referanslar / References

- Newell, P.T. et al. (2007). *A nearly universal solar wind‐magnetosphere coupling function inferred from 10 magnetospheric state variables.* JGR Space Physics.
- Baker, D.N. et al. (2018). *Highly relativistic radiation belt electron acceleration.* Nature.
- NOAA Space Weather Prediction Center: https://www.swpc.noaa.gov
- NASA DONKI API: https://kauai.ccmc.gsfc.nasa.gov/DONKI
