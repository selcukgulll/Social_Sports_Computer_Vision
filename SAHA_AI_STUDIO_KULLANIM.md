# Saha AI Studio

## Tracking sonucunu kare kare inceleme

1. `6. Test & Tracking > Tracking Çalıştır` bölümünde video, model ve çıktı
   klasörünü seçip tracking'i başlatın.
2. İşlem bitince `Frame İncele` otomatik açılır. Daha önce alınmış bir sonuç
   için bu bölümde `Çıktıyı yükle` düğmesini kullanın.
3. `Kare` düğmeleri videoda tek kare, `Analiz` düğmeleri modelin gerçekten
   çalıştırıldığı örnekleme kareleri arasında ilerler.
4. `Sonraki kopma adayı`; oyuncu sayısının düştüğü, yeni ID başladığı, düşük
   güven görüldüğü veya boşluk doldurma kullanıldığı bir sonraki kareye gider.
5. Sağdaki tablodan bir oyuncu ID'sini seçerek yalnızca o track'i ve yakın
   geçmişteki hareket izini vurgulayabilirsiniz.

`raw`, tracker'ın ürettiği ham satırları gösterir. `interpolated`, kısa süre
görülmeyen nesneler için boşlukları doldurulmuş sonucu gösterir. Tracking
örneklemesi örneğin `0.2 sn` ise model videonun her ham karesinde çalışmaz.
Bu nedenle ara karelerde gösterilen son kutu açıkça “tutuluyor” olarak
işaretlenir; gerçek bir yeni detection değildir.

İnceleyici videoyu veya CSV'nin tamamını belleğe yüklemez. Video kareleri
sınırlı bir önbellekten okunur; CSV ilk açılışta çıktı klasöründeki
`.tracking_review.sqlite` indeksine dönüştürülür.

## Yalnız top modeli

`7. Top Modeli` sekmesi oyuncu modelinden bağımsız bir `ball` modeli hazırlar.
Hazır dataset altı Roboflow kaynağından 8.031 kare ve 7.172 top kutusu içerir.
GoPro ve blur kaynaklarındaki toplam 2.276 segmentasyon polygonu bbox'a
çevrilmiştir; 2.003 topsuz kare yanlış pozitifleri azaltmak için negatif örnek
olarak korunmuştur.
Çıktı sınıf yapısı yalnızca `0: ball` şeklindedir.

Yeni Ball v8 akışı iki ayrı adımdır:

1. `1. Dataseti Hazırla` içinde CVAT etiket ve orijinal kare klasörlerini
   seçip `1. CVAT test setini hazırla` düğmesine basın. Daha önce oluşturulan
   train/validation verisi aynen korunur; augmentation yeniden çalışmaz ve CVAT
   kareleri yalnız gerçek-case `test` split'i olarak yenilenir.
2. `2. Eğit` içinde epoch, cihaz ve `Düşük / Dengeli / Yüksek` donanım yükünü
   seçip `2. Eğitimi başlat` düğmesine basın. Seçilen yük; batch, veri worker
   sayısı ve CPU thread sınırını birlikte ayarlar. Bu bir kesin kullanım yüzdesi
   değil, kaynak yoğunluğu profilidir.
3. Eğitim bittikten sonra aynı ekrandaki `3. F1 / test raporu al` düğmesi,
   CVAT'tan hazırlanan gerçek-case test splitinde precision, recall, F1 ve mAP
   raporunu üretir.

Önerilen başlangıç ayarı YOLO26m, 1024 px, dengeli yük ve 80 epoch'tur. Eğitim
tamamlandığında `Yalnız Top Testi` alt sekmesinden video ile yeni `best.pt`
seçilir. Bu akış oyuncu tespiti ve saha kalibrasyonu çalıştırmaz. Sonuç
otomatik olarak `Frame İncele` alt sekmesine yüklenir; `B1` top ID'si,
confidence, ham/interpolated kutular ve hareket izi kare kare incelenebilir.
Çok uzaktaki toplar kaçıyorsa tiled inference ayrıca açılabilir.

## Player/Ball v8 master dataset

`3. Dataset Hazırlama` sekmesindeki `Categories kaynaklarından master dataset oluştur`
bölümü artık varsayılan olarak şu planı kullanır:

`saha_ai_project/datasets/Categories/dataset_plan_player_v8.json`

Bu plan `Halisaha` klasörünü ana kaynak kabul eder, `Football` klasöründeki genel
futbol verilerini kontrollü sayıda destek olarak ekler ve `TestData` içindeki
hedefe en benzeyen 200 görüntüyü train'e karıştırmadan `test` split'ine ayırır.

Oluşan dataset:

- `train`: 6.094 görüntü
- `val`: 1.273 görüntü
- `test`: 200 görüntü
- sınıflar: `0: player`, `1: ball`, `2: outside_person`

Önerilen v8 akışı:

1. `3. Dataset Hazırlama` içinde `Master dataset oluştur`.
2. `5. Eğitim` içinde dataset otomatik olarak `player_ball_v8_master` seçili gelir.
3. Başlangıç modeli mümkünse iyi çalışan `player_ball_v2-7/weights/best.pt` olur.
4. Eğitim adı `player_ball_v8`, görüntü boyutu `1024`, batch `2` ile başlatılır.

Aynı anda top modeli veya PitchCalib eğitimi çalışırken player eğitimi
başlatılmamalıdır; tek GPU'da eğitimler sırayla yapılmalıdır.
