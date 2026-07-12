# Saha AI Eğitim ve Deney Pipeline'ı

Bu klasör, mevcut `app/process_video_to_csv_v8.py` takip scriptini değiştirmeden onun
etrafında tekrarlanabilir bir veri seçme, etiketleme, eğitim ve karşılaştırma döngüsü
kurar.

Tüm komutları `saha_ai_project` klasöründe çalıştırın:

```cmd
cd D:\Dev\GitHub\sosyaltactics\saha_ai_project
```

Yollar `pathlib` ile yönetilir. Göreli yollar proje köküne göre çözülür; bu nedenle
komutlar Windows CMD, PowerShell ve PyCharm terminalinde aynı şekilde kullanılabilir.

## Gereksinimler

V8 ile gelen gereksinimleri kurmak için:

```cmd
python -m pip install -r app\requirements_saha_tracking.txt
```

Temel paketler `opencv-python`, `pandas`, `numpy`, `tqdm`, `ultralytics` ve
`openpyxl` paketleridir. Bilgisayar çevrimdışıysa paketleri ve kullanılacak `.pt`
modelini önceden yerel diske koyun. Örneğin `--model models/base/yolo26m.pt`
kullanabilirsiniz; Ultralytics'in modeli internetten indirmesine güvenmeyin.

## Baştan sona kullanım

### 1. Klasörleri hazırla

```cmd
python scripts/00_setup_project.py
```

Bu komut eksik klasörleri, kökteki `classes.txt` dosyasını ve
`experiments/configs/default_config.json` dosyasını oluşturur. Var olan dosyaları
korur. Varsayılan dosyaları yeniden yazmak için `--force` kullanılabilir.

### 2. Ham videoları yerleştir

Maç videolarını şu klasöre kopyalayın:

```text
videos/raw/
```

Desteklenen uzantılar: `.mp4`, `.mov`, `.avi`, `.mkv`.

### 3. Düzenli aday kareleri çıkar

```cmd
python scripts/01_extract_candidate_frames.py --every-sec 2
```

Kareler `frames/candidate_frames/<video_adı>/` altına, kaynak-zaman eşleşmeleri ise
`frames/candidate_frames/frame_index.csv` dosyasına yazılır. Video başına sınır
istenirse örneğin `--max-frames-per-video 300` eklenebilir; `0` sınırsızdır.

### 4. Baseline V8 deneyini çalıştır

```cmd
python scripts/02_run_baseline_experiments.py --videos-dir videos/raw --sahi
```

Her video için V8 ayrı bir alt klasöre çıktı üretir:

```text
experiments/outputs/baseline_v8/<video_adı>/
```

Kalibrasyon penceresi interaktiftir ve script tarafından gizlenmez. Her video
başladığında kalibrasyonu ekranda tamamlayın. Daha önceki kalibrasyonu yeniden
kullanmak için `--no-force-calibration` ekleyin.

Yerel model örneği:

```cmd
python scripts/02_run_baseline_experiments.py --videos-dir videos/raw --model models/base/yolo26m.pt --sahi
```

### 5. Zor/hatalı kareleri seç

```cmd
python scripts/03_select_hard_frames.py --outputs-root experiments/outputs/baseline_v8 --videos-dir videos/raw --top-k 200
```

Script; eksik oyuncu, kayıp top, düşük tespit/pozisyon güveni, yüksek interpolasyon,
saha dışı konum, bilinmeyen takım, slice tespiti, ani oyuncu sayısı değişimi ve
gerçek dışı top sıçraması sinyallerini birlikte puanlar. Çok benzer ardışık kareleri
azaltmak için varsayılan minimum zaman aralığı 1 saniyedir:

```cmd
python scripts/03_select_hard_frames.py --min-gap-sec 1.0
```

Çıktılar:

```text
frames/hard_frames/baseline_v8/<video_adı>/*.jpg
frames/hard_frames/baseline_v8/hard_frames_index.csv
```

### 6. Manuel etiketleme klasörünü hazırla

```cmd
python scripts/04_prepare_label_staging.py
```

Önce en yüksek puanlı zor kareler, sonra videolar arasında dengeli biçimde normal
aday kareler eklenir. Aynı videonun birbirine çok yakın kareleri ve yinelenen
görseller elenir.

Çıktılar:

```text
frames/label_staging/images/
frames/label_staging/labels_empty/
frames/label_staging/label_index.csv
frames/label_staging/classes.txt
```

`labels_empty` yalnızca boş dosya şablonlarıdır. Bunları tamamlanmış manuel etiket
sanmayın.

### 7. Manuel etiketleme

Etiketleme aracında şu klasörü açın:

```text
frames/label_staging/images
```

Sınıflar:

```text
0 player
1 ball
2 outside_person
```

`outside_person`; aktif saha dışındaki seyirci, kenar çizgisi görevlisi veya diğer
kişiler anlamına gelir. Her görsel için YOLO biçimindeki `.txt` dosyasını şu klasöre
kaydedin:

```text
labels/manual
```

Her etiket satırı şu beş normalize değeri içermelidir:

```text
class_id center_x center_y width height
```

### 8. YOLO veri kümesini oluştur

```cmd
python scripts/05_build_yolo_dataset.py --images-dir frames/label_staging/images --labels-dir labels/manual --out-dir datasets/dataset_v1
```

Etiketi olmayan görseller varsayılan olarak uyarıyla atlanır. Bilerek negatif örnek
olarak eklemek isterseniz `--include-unlabeled` kullanın. Bölme `--seed 42` ile
tekrarlanabilir biçimde yapılır. Script etiket satırlarını doğrular ve sorunları
dosya/satır numarasıyla bildirir.

Üretilen yapı:

```text
datasets/dataset_v1/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  data.yaml
  dataset_summary.json
```

### 9. Eğitim komutunu üret

```cmd
python scripts/06_train_commands.py --dataset datasets/dataset_v1 --base-model models/base/yolo26m.pt --name player_ball_v1
```

Komut ekrana basılır ve şuraya kaydedilir:

```text
experiments/reports/train_command_player_ball_v1.txt
```

Script varsayılan olarak eğitimi başlatmaz. Hazır olduğunuzda aynı komuta `--run`
ekleyin:

```cmd
python scripts/06_train_commands.py --dataset datasets/dataset_v1 --base-model models/base/yolo26m.pt --name player_ball_v1 --run
```

Başarılı eğitimden sonra beklenen model:

```text
models/trained/player_ball_v1/weights/best.pt
```

### 10. Özel modeli V8 ile test et

```cmd
python app/process_video_to_csv_v8.py --video videos/raw/Video_Project2.mp4 --out experiments/outputs/custom_v1/Video_Project2 --sample-sec 0.1 --model models/trained/player_ball_v1/weights/best.pt --force-calibration --field-mapper poly2 --conf 0.12 --sahi --yolo-classes 0,1
```

Özel veri kümesinde top sınıfı `1` olduğu için `--yolo-classes 0,1` önemlidir.
V8'in COCO varsayılanı `0,32` (person, sports ball) baseline model içindir.
`outside_person` sınıfı `2`, aktif oyuncu tracker'ına bilerek verilmez.

### 11. Baseline ve özel modeli karşılaştır

```cmd
python scripts/07_compare_experiments.py --runs experiments/outputs/baseline_v8 experiments/outputs/custom_v1
```

Raporlar:

```text
experiments/reports/experiment_comparison.csv
experiments/reports/experiment_comparison.md
```

Bu rapordaki değerler ground-truth doğruluk ölçümleri değil; iki koşuyu hızlı
karşılaştırmaya yarayan sezgisel kalite sinyalleridir.

## İterasyon döngüsü

```text
baseline
  → hard frames
  → manual labels
  → dataset_v1
  → train player_ball_v1
  → test
  → compare
  → yeni hard frames
  → dataset_v2
  → train player_ball_v2
```

Yeni turda özel model çıktısını ayrı bir deney klasörüne yazın, aynı zor-kare
scriptini o klasöre yöneltin ve yeni etiketleri veri kümesine ekleyerek
`datasets/dataset_v2` oluşturun. Deney ve model adlarını değiştirmek eski sonuçların
üzerine yazılmasını önler.

Örnek ikinci tur:

```cmd
python scripts/03_select_hard_frames.py --outputs-root experiments/outputs/custom_v1 --videos-dir videos/raw --out-dir frames/hard_frames/custom_v1 --top-k 200
python scripts/04_prepare_label_staging.py --hard-index frames/hard_frames/custom_v1/hard_frames_index.csv
python scripts/05_build_yolo_dataset.py --images-dir frames/label_staging/images --labels-dir labels/manual --out-dir datasets/dataset_v2
python scripts/06_train_commands.py --dataset datasets/dataset_v2 --base-model models/trained/player_ball_v1/weights/best.pt --name player_ball_v2
```

## Script özeti

- `00_setup_project.py`: klasörleri ve varsayılan yapılandırmayı oluşturur.
- `01_extract_candidate_frames.py`: düzenli video karelerini çıkarır.
- `02_run_baseline_experiments.py`: V8'i video klasörü üzerinde çalıştırır.
- `03_select_hard_frames.py`: sezgisel hataları puanlar ve zor kareleri çıkarır.
- `04_prepare_label_staging.py`: manuel etiketleme paketini hazırlar.
- `05_build_yolo_dataset.py`: etiketleri doğrular ve YOLO veri kümesini böler.
- `06_train_commands.py`: eğitim komutunu üretir veya isteğe bağlı çalıştırır.
- `07_compare_experiments.py`: deney çıktılarını CSV ve Markdown olarak kıyaslar.
