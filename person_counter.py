import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
from my_utils import select_frame

# Gelecek uyarıları devre dışı bırak
warnings.filterwarnings("ignore", category=FutureWarning)

# YOLOv5 modelini yükle ve GPU desteği mevcutsa kullan
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = model.cuda() if torch.cuda.is_available() else model.to('cpu')

# Deep SORT Tracker ayarlarını yapılandır
tracker = DeepSort(
    max_age=50,  # Nesnenin kaybolmadan önceki maksimum görünmezlik süresi (kare sayısı)
    n_init=3,  # Takip doğrulaması için gereken minimum kare sayısı
    max_iou_distance=0.6,  # IOU eşleşme eşiği
    nn_budget=150  # Bellek yönetimi için ayrılmış maksimum kapasite
)

# Video giriş ve çıkış dosyalarını ayarla
video_path = 'videoplayback.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

output_path = 'processed_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Kullanıcıdan ROI (Region of Interest) seçimini al
rect = select_frame(video_path)
if rect is None:
    print("Çerçeve seçimi başarısız. Program sonlandırılıyor.")
    cap.release()
    exit()

# ROI koordinatlarını çözümle
rect_x1, rect_y1, rect_w, rect_h = map(int, rect)
rect_x2, rect_y2 = rect_x1 + rect_w, rect_y1 + rect_h

# Başlangıç sayacı ve diğer değişkenleri tanımla
passing_count = 0
tracked_ids = set()
track_cache = {}  # Geçici olarak kaybolan ID'ler için önbellek

# ROI içinde bir nesne olup olmadığını kontrol eden yardımcı fonksiyon
def is_inside_roi(box, roi):
    """Belirli bir çerçevenin ROI içinde olup olmadığını kontrol eder."""
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    center_x, center_y = (bx1 + bx2) // 2, (by1 + by2) // 2
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ROI'yi görselleştir
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), 2)

    # YOLOv5 kullanarak nesne algılama
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    # Algılanan nesneleri takip için dönüştür
    detections = []
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0 and conf > 0.6:  # Sadece 'Person' sınıfını seç
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            # Küçük nesneleri dışla
            if w * h < 800:
                continue

            if is_inside_roi([x1, y1, x2, y2], [rect_x1, rect_y1, rect_x2, rect_y2]):
                detections.append(([x1, y1, w, h], conf, 'person'))

    # Takipçiyi güncelle ve izleme işlemlerini gerçekleştir
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Nesne ROI içindeyse sayacı artır
        if is_inside_roi([x1, y1, x2, y2], [rect_x1, rect_y1, rect_x2, rect_y2]) and track_id not in tracked_ids:
            passing_count += 1
            tracked_ids.add(track_id)
            print(f"ID {track_id} geçti! Güncel Sayım: {passing_count}")

        # Çerçeve üzerine takip bilgilerini yazdır
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Sayımı ekrana yazdır
    cv2.putText(frame, f'Count: {passing_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # İşlenen çerçeveyi göster ve kaydet
    cv2.imshow('Frame', frame)
    out.write(frame)

    # ESC tuşu ile çıkış yap
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Video kaynaklarını serbest bırak ve işlemi sonlandır
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"İşlenen video {output_path} dosyasına kaydedildi.")
