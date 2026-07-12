# player_ball_v1 — İlk değerlendirme

## Veri

- CVAT dışa aktarımı: `saha_labels_v1.zip`
- Etiketli kare: 31
- Bölünme: 22 train / 6 validation / 3 test
- Kutular: 399 player / 23 ball / 171 outside_person
- Geçersiz etiket satırı: 0

Bu kareler aynı ana videodan ve birbirine yakın zamanlardan geldiği için validation
ve test metrikleri gerçek genelleme başarısını olduğundan yüksek gösterebilir.

## Eğitim

- Başlangıç modeli: `models/base/yolo26m.pt`
- Çıktı: `models/trained/player_ball_v1/weights/best.pt`
- Ayarlar: imgsz 1024, batch 2, nbs 8, 60 epoch, mosaic kapalı
- En iyi epoch: 50
- Validation: precision 0.923, recall 0.750, mAP50 0.781, mAP50-95 0.340

Validation sınıf sonuçları:

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| player | 0.988 | 0.950 | 0.960 | 0.522 |
| ball | 0.965 | 0.500 | 0.590 | 0.191 |
| outside_person | 0.815 | 0.800 | 0.794 | 0.306 |

## Ayrılmış üç test karesi

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| player | 0.952 | 0.929 | 0.949 | 0.469 |
| ball | 1.000 | 0.000 | 0.000 | 0.000 |
| outside_person | 0.628 | 0.910 | 0.777 | 0.286 |

Test bölümünde yalnızca bir top örneği vardır ve model bu örneği kaçırmıştır.

## Test videosu taraması

Dokuz test klibi 1 FPS örnekleme ve `conf=0.10` ile tarandı. Yalnız
`1_clip_0047_start_00-00-16-313.mp4` aktif maçı içeriyordu. Diğer kliplerde ana
saha çoğunlukla boştu; yan sahadaki kişiler ağırlıklı olarak `outside_person`
olarak bulundu.

Aktif klipte 128 örnek kare:

| Confidence | Ortalama player | Medyan player | Top bulunan kare |
|---:|---:|---:|---:|
| 0.10 | 14.60 | 15 | 28 / 128 |
| 0.25 | 13.68 | 14 | 17 / 128 |
| 0.50 | 12.90 | 13 | 5 / 128 |

## Sonuç

Oyuncu tespiti ilk tur için umut verici. Top sınıfı yetersiz örnek nedeniyle
kararsız. Sonraki veri turunda farklı maç ve ışık koşullarından, topun net
göründüğü en az 100–200 ek kare öncelikli olmalıdır. Gerçek test seti eğitim
videosundan bağımsız bir maçtan oluşturulmalıdır.
