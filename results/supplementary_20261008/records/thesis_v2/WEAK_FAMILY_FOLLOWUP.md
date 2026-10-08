# 0.80 drift/noise hedefi: başarısız denemeden sonra karar

Sonraki somut deney sırası, senaryo bazında ek analiz ve geçme/durma koşulları [DRIFT_NOISE_RECOVERY_PLAN.md](/Users/ozanbabac5/wdn_thesis/thesis_v2/DRIFT_NOISE_RECOVERY_PLAN.md) dosyasındadır. Bu ek plan yeni bir eğitim sonucu değildir.

## Ne oldu?

Dört yeni aday tamamlandı; hiçbir aday hem drift hem noise için 0.80 validation F1'e ulaşmadı.
Calibration ile seçilen adayda drift **0.4590**, noise **0.4054**, genel F1 **0.8176**, replay **0.8657**.
Önceki seçilmiş redesign drift/noise 0.5312/0.4675 veriyordu. Yeni seçilmiş paket iki zayıf ailede de geriye gitti; ana modelin yerine konmadı.

Seçilmeyen joint adayın genel F1'i 0.8484 olsa da drift/noise 0.2593/0.3429. Bu yüksek genel skor, istenen tüm-aile başarısı değildir.
Yeni yerel ağaç kontrolü drift/noise 0.5538/0.4810: küçük bir değişim var, fakat 0.80'e yakın değil. Validation'a bakarak bu aday kazanan ilan edilmedi.

## 1. Sorun yalnız router/eşik değil

Seçilen adayda drift expert AUPRC 0.6161, noise expert AUPRC 0.4684.
Kendi bağımsız calibration eşikleri, karışımın 33 drift kaçırmasının yalnız 2'sini ve 42 noise kaçırmasının 6'sını yakalayabiliyor.
Bu eşikler teorik optimum değildir; yine de birleştiricide hazır, kusursuz bir expert'in sistematik olarak susturulduğu açıklamasına güçlü destek vermiyor.

Drift ilk üç saatte 0/21; noise 3/26. Alarm devamlılığının ötesinde ilk kanıtı üretme sorunu sürüyor.
Drift kaçırmalarının 32/33'ü, noise kaçırmalarının 34/42'si aynı sensörde ilk alarmdan önce veya hiç alarm gelmeyen sensörlerde.
Bu sayımlar gerçek olay sınırlarıyla yapılan tanıdır; model girdisi veya uygulanabilir skor artışı değildir.

## 2. Calibration'daki iyi sonuç model seçimini yanıltıyor

| Aday | Senaryo dışı TRAIN drift AUPRC | Senaryo dışı TRAIN noise AUPRC | Calibration noise AUPRC | Validation noise AUPRC |
|---|---:|---:|---:|---:|
| Yerel ağaçlar | 0.5994 | 0.6135 | 0.9235 | 0.5803 |
| Ortak bağlam, seçilen | 0.5715 | 0.5029 | 0.8961 | 0.4684 |
| Joint pressure–flow | 0.5736 | 0.4647 | 0.9270 | 0.5857 |
| Ortak bağlam, Optuna ayarı | 0.5368 | 0.5395 | 0.8827 | 0.4980 |

OOF değerleri, ilgili senaryoyu eğitimde görmemiş **expert** skorlarıdır. OOF skorlarıyla eğitilen birleştiricinin kendi fit başarısı değildir.
Fold modelleri daha az veriyle eğitildiğinden farkın tamamını aşırı öğrenmeye bağlayamayız; tek calibration olayının temsiliyetinin zayıf olduğunu gösteren pratik bir uyarıdır.
Seçilen drift expert'inin dışarıda bırakılmış eğitim senaryosu 16'daki AUPRC'si yalnızca 0.2361. Zayıf olaydaki sorun validation açılmadan da görülebiliyordu.

**Sonraki seçim değişmeli:** önce birden fazla TRAIN senaryosu dışarıda bırakıldığında iki zayıf expert'in de ayrım gücünü korumasını şart koşmak; ardından calibration ile ortak eşik/FPR belirlemek.
Bu tur seçim kuralı geriye dönük değiştirilmedi. Birleştirici de değerlendirilecekse tüm pipeline'ı dış senaryodan ayıran nested değerlendirme gerekir; aynı OOF skorlarını rastgele bölmek yeterli değildir.
Önerilen değişiklik tek başına expert'i güçlendirmez; calibration'da iyi görünüp genellemeyen adaya kaynak harcamayı azaltır.

## 3. Joint referansın ortalama normal hatası düşmedi

Normal gözlemlere karşı referans MAE:

| Bölüm | Basınç referansı | Joint referans |
|---|---:|---:|
| Calibration | 0.1892 m | 0.2043 m |
| Validation | 0.1678 m | 0.1810 m |

Denediğimiz şey iki kanalın birimleri normalize edilerek ortak düşük-rank modelde öğrenilmesiydi; boru hidrolik denklemlerini çözen bir model değildi.
Bu nedenle "flow işe yaramıyor" sonucu çıkmaz. Ortalama normal hata düşmedi; fakat sonraki senaryo bazlı incelemede joint drift expert'i en zor TRAIN senaryosunda AUPRC'yi 0.2165'ten 0.4057'ye yükseltti. Birleştirilmiş OOF AUPRC ile senaryo ortalama/en kötü AUPRC aynı ölçü değildir; yeni plan bu ayrımı korur.

## Bundan sonra ne öneriyorum?

1. **Yeni büyük Optuna araması açma.** Eski güçlü checkpoint'leri ve bu başarısızlık raporunu koru. Mevcut özelliklerde daha büyük modelin 0.80'e ulaşacağına dair kanıt yok.
2. **Normal referansı fiziksel olarak iyileştiren küçük kontrol yap.** Mevcut gözlenen basınç/debi ve boru özelliklerinden, basınç farkı–debi tutarlılığına dayalı, belirsizlik üreten ayrı kestirim. Önce yalnız TRAIN senaryo dışı normal hata ve erken-drift ayrımı iyileşiyor mu ölç. Flow saldırılarını/eksikleri sağlam kestirimle ele al; temiz simülasyon hedefi, gerçek talep truth'u veya saldırılan sensör listesi kullanma. Bu deney henüz uygulanmadı.
3. **Noise için gerçek normal/atak durum filtresini karşılaştır.** Bu turdaki skor maksimumu/EWMA, olasılıksal bir olay-durum modeli değildir. Normal ölçeği TRAIN'den öğrenen, bilinmeyen başlangıç/bitiş ve değişen varyansı ele alan nedensel filtre ayrı bir adaydır. Mevcut varyans-state özelliklerini yalnız yeniden adlandırmak veya alarmı sabit süre açık tutmak yeterli değil; yararı bağımsız senaryoda gösterilmeli.
4. **Veri yetersizliği sürerse yalnız TRAIN çeşitliliğini genişletmeyi ayrı onayla ele al.** Drift ve noise için dörder bağımsız eğitim senaryosu var; calibration/validation'da birer olay bulunuyor. Aynı parametre aralıkları ve %50 missing ile daha fazla eğitim senaryosu düşünülebilir; mevcut calibration/validation/test örnekleri ve saldırı şiddetleri değiştirilmez. Bu, sabit veri kapsamının genişlemesidir; mevcut turda yapılmadı ve açık kullanıcı onayı olmadan yapılmamalı.

Önceliğim, her yeni aday için önce senaryo dışı ayrım gücü şartını koymak, sonra normal referansın fiziksel doğruluğunu ölçmek.
İki ailede 0.80 hâlâ hedef; mevcut denemeler bunu başarmanın garanti veya yalnız hyperparametre meselesi olduğunu göstermiyor.

## Sınırlar

Tüm sonuçlar tekrar kullanılan geliştirme validation'ına aittir; kilitli test kullanılmadı. Pressure ve flow missing 0.50, ham veri hashleri ve dış splitler aynı.
Dört eski GNN denemesi, dört önceki redesign denemesi ve dört bu-tur adayı ayrı çalışmalardır; hiçbiri sessizce uzatılmadı.
Tüm yeni adaylarda expert auditleri mevcut. Kaydedilmiş seçilmiş model calibration/validation skorlarını birebir yeniden üretiyor. 40 yazılım testi geçti.

Sayısal denetim: `thesis_v2/outputs/weak_family_results.json`.
Yeniden üretim: `thesis_v2/experiments/report_weak_families.py`.
