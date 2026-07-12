# PitchCalibStudio

PitchCalibStudio, sabit futbol/futsal/halı saha kamerasındaki saha
landmark'larını YOLO pose modeliyle bulmak ve bu noktalardan homography
kalibrasyonu üretmek için hazırlanmış bağımsız bir yerel uygulamadır.

Bu proje `SahaAIStudio`dan tamamen ayrıdır; oradan kod içe aktarmaz, ona
bağımlı değildir ve onun dosyalarını değiştirmez. Daha sonra üretilen
`calibration.json`, oyuncu takip sonuçlarını görüntü koordinatından normalize
saha koordinatına taşımak için iki proje arasında bir köprü olabilir.

## Neden object detection değil pose?

Normal detection modeli yalnızca sahanın çevresine bir kutu çizer. Kalibrasyon
için saha köşesi, orta çizgi kesişimi ve merkez gibi **sabit ve anlamı bilinen
noktalar** gerekir. Bu nedenle:

- Tek sınıf: `pitch`
- Çoklu landmark/keypoint
- YOLO görevi: `pose`

`yolo26m.pt` bir detection ağırlığıdır ve bu projede başlangıç modeli olarak
kullanılamaz. Yerel `models/yolo26m-pose.pt` doğru pose ağırlığıdır.

## Başlatma

En kolay yöntem, Windows Explorer'da `start_pitchcalibstudio.pyw` dosyasına
çift tıklamaktır. Uygulama tarayıcıda `http://127.0.0.1:8502` adresini açar.

İlk kurulum gerekirse:

```powershell
python -m pip install -r requirements.txt
```

Alternatif başlatma:

```powershell
python -m streamlit run app.py
```

Uygulama çalışma anında Roboflow API veya bulut inference kullanmaz. Model,
dataset ve çıktıların tamamı yereldir.

## Dataset yerleşimi

İndirilen üç YOLOv8 dataset şu kökte kalabilir:

```text
FieldDetectionModel/
├─ datasets/
│  ├─ Cancha_F5.v1i.yolov8/
│  ├─ futsal-pitch-detection.v9i.yolov8/
│  └─ keypoints.v1i.yolov8/
└─ PitchCalibStudio/
```

`halisaha` adına benzeyen düşük kaliteli klasörler audit/merge sırasında
bilinçli olarak yok sayılır.

5 Temmuz 2026 tarihli gerçek audit sonucu:

| Dataset | Sınıf | kpt_shape | Görsel/etiket |
|---|---:|---:|---:|
| Cancha_F5 | `cancha` | `[14, 3]` | 318 / 318 |
| futsal-pitch-detection | `grass` | `[16, 3]` | 420 / 420 |
| keypoints | `pitch` | `[22, 3]` | 221 / 221 |

Üçü de YOLO pose satırları içerir, fakat nokta sayıları ve indeks anlamları
aynı değildir. Bu nedenle **doğrudan birleştirilmemelidir**.

## Önerilen çalışma sırası

### 1. Dataset Audit

Uygulamada `Dataset Audit` sekmesine girip **Audit çalıştır** düğmesine basın.
Araç şunları kontrol eder:

- `data.yaml`, `names`, `nc`, `kpt_shape` ve split yolları
- Görsel/etiket sayıları, eksik veya boş etiketler
- YOLO pose satır uzunluğu ve keypoint boyutu
- Normalize bbox ve keypoint değerleri
- Büyük indeks numaralı örnek görseller

CLI karşılığı:

```powershell
python tools/audit_pose_datasets.py `
  --datasets_root ..\datasets `
  --output outputs\audits `
  --preview-count 20
```

Raporlar tarihlendirilmiş olarak `outputs/audits/`, görseller
`outputs/previews/` altına yazılır.

### 2. Keypoint sırasını kontrol et

`Keypoint Preview` sekmesinde her datasetin `data.yaml` dosyasını sırayla
seçin. Aynı indeksin farklı görüntülerde hangi fiziksel saha noktasını
gösterdiğini not edin. `index.html` ve `contact_sheet.jpg` hızlı inceleme için
üretilir.

```powershell
python tools/preview_keypoints.py `
  --data ..\datasets\Cancha_F5.v1i.yolov8\data.yaml `
  --split train --count 50 `
  --output outputs\previews\cancha_check
```

### 3. Şema ve remap'i tamamla

`configs/pitch_schema.yaml`, Rocktail datasetinin `0..21` indeks düzenini
ortak hedef kabul eden **22 noktalı** doğrulanmış şemadır. `0..21` aralığı
21 değil, toplam 22 nokta demektir. Dört dış köşe birim saha dikdörtgenine
sabitlenerek 87 etiketli kare üzerinde yapılan geometrik kontrolde medyan
izdüşüm hatası yaklaşık 2 pikseldir.

Hedef düzen:

```text
Sol kale çizgisi:  0, 1, 2, 3, 4, 5
Sol ceza alanı:    6, 7
Orta bölüm:        8, 9, 10, 11, 12, 13
Sağ kale çizgisi: 14, 15, 16, 17, 18, 19
Sağ ceza alanı:   20, 21
```

Datasetler farklı sayıda ve sırada nokta içerdiğinden:

1. `dataset_remap_template.yaml` dosyasının bir kullanıcı kopyasında her
   dataset için `source_to_target` değerlerini doldurun.
2. Görsel doğrulama bittikten sonra yalnız o dataset için
   `confirmed_order: true` yapın.

Örnek söz dizimi:

```yaml
source_to_target:
  0: 3       # kaynak 0, hedef şemadaki 3
  1: 0
  2: null    # bu kaynak noktayı kullanma
confirmed_order: true
```

Hedef indeks tekrarı, aralık dışı indeks, eksik class mapping veya
`confirmed_order: false` durumunda merge durur. Bu koruma bilerek kaldırılmadı.

Bu kurulumda doğrulanan kullanıcı remap'iyle oluşturulan birleşmiş dataset
959 görsel içerir: 767 train, 95 validation ve 97 test. Tüm etiketler
`kpt_shape: [22, 3]` biçimindedir.

### 4. Datasetleri birleştir

`Merge Datasets` sekmesinde doğrulanmış schema ve remap dosyalarını seçin.
Çıktı klasörü doluysa üzerine sessizce yazılmaz.

```powershell
python tools/merge_pose_datasets.py `
  --datasets_root ..\datasets `
  --output data\merged\pitch_pose `
  --schema configs\pitch_schema.yaml `
  --remap configs\dataset_remap_user.yaml `
  --seed 42
```

Çıktı:

```text
data/merged/pitch_pose/
├─ images/{train,val,test}/
├─ labels/{train,val,test}/
├─ data.yaml
├─ merge_report.json
└─ merge_report.csv
```

### 5. Eğit

`Train` sekmesindeki değerleri ayarlayın. Komut önceden gösterilir; eğitim
yalnız onay kutusu işaretlenip **Eğitimi başlat** düğmesine basıldığında başlar.

Varsayılan:

```powershell
python train_pitch_pose.py `
  --data data\merged\pitch_pose\data.yaml `
  --model models\yolo26m-pose.pt `
  --epochs 150 --imgsz 960 --batch 8 --device 0
```

Kısa CPU testi:

```powershell
python train_pitch_pose.py `
  --data data\merged\pitch_pose\data.yaml `
  --model models\yolo26m-pose.pt `
  --epochs 5 --imgsz 640 --batch 4 --device cpu
```

Run'lar `outputs/training_runs/` altında tutulur. `best.pt` ve `last.pt`,
tarihli adlarla `models/` klasörüne kopyalanır; komut, kullanılan `data.yaml`
ve validation metrikleri run klasöründe saklanır.

Yatay/dikey flip olasılığı varsayılan olarak `0` tutulur. Landmark'ların
semantik sağ/sol eşleri için doğru `flip_idx` kesinleşmeden flip augmentasyonu
açmak etiket anlamlarını bozabilir.

### 6. Tek karede test et

`Image Viewer` sekmesinde bir saha karesi ve eğitilmiş `.pt` modeli seçin.
Uygulama:

- En iyi `pitch` instance'ını seçer.
- Bbox, indeks, ad ve confidence değerlerini çizer.
- En az dört geçerli schema noktasıyla RANSAC homography kurar.
- Saha çizgilerini görüntüye geri projekte eder.
- Top-view ve ayrıntılı JSON üretir.

```powershell
python calibrate_image.py `
  --image frame.jpg `
  --model models\pitch_pose_best_YYYYMMDD_HHMMSS.pt `
  --schema configs\pitch_schema.yaml --conf 0.25
```

### 7. Videoyu kalibre et

Kamera sabitse videonun tamamını kalibrasyon modeliyle çalıştırmak gerekmez.
İlk birkaç saniyeden örneklenen kareler puanlanır; reprojection error,
confidence ve kareler arası kararlılığa göre en iyi kalibrasyon seçilir.

```powershell
python calibrate_video.py `
  --video match.mp4 `
  --model models\pitch_pose_best_YYYYMMDD_HHMMSS.pt `
  --schema configs\pitch_schema.yaml `
  --seconds 5 --sample-fps 5 --conf 0.25
```

`outputs/calibrations/<tarih_video>/calibration.json` içinde iki matris bulunur:

- `H_image_to_field`: görüntü pikseli → normalize saha koordinatı
- `H_field_to_image`: normalize saha koordinatı → görüntü pikseli

`Test Transform` sekmesi, oyuncu bbox'larının alt-orta noktalarını
`H_image_to_field` ile dönüştürür ve top-view üzerinde gösterir.

## Klasörler

```text
PitchCalibStudio/
├─ configs/                 # uygulama eşikleri, şema ve remap
├─ data/
│  ├─ raw/                  # UI yüklemeleri
│  ├─ audited/
│  └─ merged/
├─ models/                  # başlangıç ve eğitilmiş pose modelleri
├─ outputs/
│  ├─ audits/
│  ├─ previews/
│  ├─ training_runs/
│  ├─ predictions/
│  └─ calibrations/
├─ src/                     # dataset, keypoint, homography, çizim yardımcıları
└─ tools/                   # audit, preview, remap ve merge CLI araçları
```

Tüm önemli çıktılar tarihlendirilir. Merge klasörünün veya bir run'ın üzerine
sessizce yazılmaz.

## Sık sorunlar

**Yanlış keypoint sırası:** Model görsel olarak nokta bulur ama saha çizgileri
yamuk/ters çıkar. Önizlemeleri kontrol edip remap ve schema sırasını düzeltin.

**Yanlış `kpt_shape`:** Audit satır uzunluklarını reddeder. Detection formatı
(`class x y w h`) pose formatı değildir.

**Model pitch buluyor, homography kurulamıyor:** En az dört tane aynı düzlemde,
anlamı doğru tanımlanmış ve yeterli confidence'a sahip schema noktası gerekir.

**Az görünür nokta:** `conf` değerini kontrollü biçimde düşürün veya kameranın
gördüğü landmark'ları daha iyi temsil eden veri ekleyin. Dört noktanın altı
geometrik olarak yeterli değildir.

**Saha ters dönüyor:** Sağ/sol veya üst/alt landmark eşlemesi yanlıştır.
`field_xy` ve remap indekslerini düzeltin; görüntüyü rastgele flip etmeyin.

**`yolo26m-pose.pt` bulunamıyor:** `models/` altında gerçekten `-pose.pt`
dosyası olduğundan emin olun. `yolo26m.pt` ile değiştirmeyin.

**CUDA yok:** `device=cpu` kullanın. Küçük smoke test için düşük `imgsz`,
batch ve epoch seçin; gerçek eğitim çok daha yavaş olacaktır.

**Auto kabul “Hayır”:** Bu mutlaka tahmin yok demek değildir. Reprojection
error, saha poligon alanı veya confidence eşiği
`configs/app_config.yaml` sınırlarını geçmemiştir. Overlay'i görsel kontrol
etmeden eşikleri gevşetmeyin.
