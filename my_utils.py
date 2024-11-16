import cv2

def select_frame(video_path):
    """
    Video dosyasından belirli bir kareyi seçip ROI (Region of Interest) çerçevesi oluşturur.
    """
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)

    # 2. saniyedeki kareyi yakala
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)  # 2000 ms = 2 saniye
    ret, frame = cap.read()

    if not ret:
        print("2. saniyedeki kare alınamadı.")
        cap.release()
        return None

    # Kullanıcının çerçeve seçmesini sağla
    print("Çerçeveyi seçmek için farenizle sürükleyin ve Enter'a basın.")
    rect = cv2.selectROI("Çerçeve Seçimi", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Çerçeve Seçimi")

    cap.release()
    return rect  # (x, y, w, h) koordinatlarını döndür
