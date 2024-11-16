# Person Counter

Bu proje, bir video dosyasındaki kişileri algılamak ve saymak için **YOLOv5** nesne algılama ve **Deep SORT** takip algoritmalarını kullanır. Kullanıcı, belirli bir bölgeyi (ROI) seçer ve sadece bu bölge içerisinden geçen kişiler sayılır.

## Özellikler
- **YOLOv5** ile yüksek hassasiyetli nesne algılama.
- **Deep SORT** kullanarak nesne takibi ve ID yönetimi.
- Kullanıcıdan ROI (Region of Interest) seçimi.
- Geçiş yapan kişi sayısını canlı olarak görüntüleme.
- İşlenen videonun kaydedilmesi.

## İşlenmiş Video Çıktısı

Aşağıdaki bağlantıya tıklayarak işlenmiş video çıktısını izleyebilirsiniz:

[![Videoyu İzle](https://via.placeholder.com/800x450?text=Video+Placeholder)](https://raw.githubusercontent.com/username/repository/branch/processed_video.mp4)

> Not: Yukarıdaki "Video Placeholder" yerine kendi bir küçük resim dosyanızı eklemek isterseniz, bir `.jpg` veya `.png` dosyası oluşturup `via.placeholder.com` bağlantısını değiştirin.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gereklidir:
- `torch`
- `opencv-python`
- `numpy`
- `deep_sort_realtime`

Kurulum:
```bash
pip install -r requirements.txt
