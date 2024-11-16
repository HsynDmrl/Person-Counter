# Person Counter

Bu proje, bir video dosyasındaki kişileri algılamak ve saymak için **YOLOv5** nesne algılama ve **Deep SORT** takip algoritmalarını kullanır. Kullanıcı, belirli bir bölgeyi (ROI) seçer ve sadece bu bölge içerisinden geçen kişiler sayılır.

## Özellikler
- **YOLOv5** ile yüksek hassasiyetli nesne algılama.
- **Deep SORT** kullanarak nesne takibi ve ID yönetimi.
- Kullanıcıdan ROI (Region of Interest) seçimi.
- Geçiş yapan kişi sayısını canlı olarak görüntüleme.
- İşlenen videonun kaydedilmesi.

## İşlenmiş Video Çıktısı

Aşağıdaki küçük resme tıklayarak işlenmiş video çıktısını izleyebilirsiniz:

[![Videoyu İzle](https://raw.githubusercontent.com/HsynDmrl/Person-Counter/main/thumbnail.png)](https://raw.githubusercontent.com/HsynDmrl/Person-Counter/main/processed_video.mp4)

> Not: Eğer video oynatılamıyorsa veya görüntülenemiyorsa, [buraya tıklayarak videoyu doğrudan izleyebilirsiniz.](https://raw.githubusercontent.com/HsynDmrl/Person-Counter/main/processed_video.mp4)

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gereklidir:
- `torch`
- `opencv-python`
- `numpy`
- `deep_sort_realtime`

Kurulum:
```bash
pip install -r requirements.txt
