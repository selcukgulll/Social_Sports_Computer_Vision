# Saha Tracking V8

Bu sürüm V7 üzerine şu iyileştirmeleri ekler:

## 1) Daha güçlü curve-aware kalibrasyon
- Ana dört saha sınırı yine eğri/3-handle'lı.
- Kullanıcı uçları tam köşeye kadar götürmek zorunda değil.
- Sistem ana çizgileri uzatıp köşeleri **kesişimden** bulur.
- Sol/sağ ve üst/alt rolleri kullanıcı yanlış verse bile görüntü geometrisine göre otomatik rol atamaya çalışır.

## 2) Yeni manuel referans çizgileri
Kalibrasyon ekranında artık bunlar da var:
- Orta Çizgi
- Sol Ceza Sahası Ön Çizgisi
- Sağ Ceza Sahası Ön Çizgisi

Bunlar özellikle:
- perspektif
- balık gözü / yassı çizgi
- saha oranı tahmini

konularında poly mapper'a ek kısıt sağlar.

## 3) Oyuncu kimliği için daha detaylı appearance
Artık sadece üst-alt renk değil:
- upper_hist
- mid_hist
- lower_hist
- shoe_hist
- head_hist
- skin_ratio
- head_dark_ratio

çıkarılıyor ve tracker reID cost içinde kullanılıyor.

## Önerilen komut

```bash
python process_video_to_csv_v8.py --video "Video_Project2.mp4" --out "output_v8" --sample-sec 0.1 --model yolo11x.pt --force-calibration --field-mapper poly2 --conf 0.12 --sahi
```

## Manuel oyuncu seed ile

```bash
python process_video_to_csv_v8.py --video "Video_Project2.mp4" --out "output_v8_manual" --sample-sec 0.1 --model yolo11x.pt --force-calibration --field-mapper poly2 --conf 0.12 --manual-initial-players --sahi
```

## Kalibrasyon notu

Kalibrasyon ekranındaki ana mantık:
- Sol Kale Çizgisi / Sol Kısa Kenar
- Sağ Kale Çizgisi / Sağ Kısa Kenar
- Üst Taç Çizgisi / Üst Uzun Kenar
- Alt Taç Çizgisi / Alt Uzun Kenar
- Orta Çizgi
- Sol Ceza Sahası Ön Çizgisi
- Sağ Ceza Sahası Ön Çizgisi

Özellikle orta çizgi ve ceza sahası ön çizgilerini iyi oturtursan kuş bakışı mapping daha stabil olur.
