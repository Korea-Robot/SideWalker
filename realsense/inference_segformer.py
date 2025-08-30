#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# 1) Ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Wczytanie modelu SegFormer z punktu kontrolnego
checkpoint = "smp-hub/segformer-b2-1024x1024-city-160k"
model = smp.Segformer.from_pretrained(checkpoint).eval().to(device)

# 3) Przetwarzanie wstępne obrazu
img_size = 512
preprocessing = A.Compose([
    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
    # POPRAWKA: Zmieniono 'constant_values' na 'value'
    A.PadIfNeeded(min_height=img_size, min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 4) Utworzenie palety kolorów
palette_base = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.int64)
colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
colors = colors.numpy().astype('uint8')  # Paleta RGB

# 5) Konfiguracja strumienia wideo z kamery RealSense
rtsp_url = "192.168.168.105:4000/fl"
cap = cv2.VideoCapture(rtsp_url)

# POPRAWKA: Poprawiono literówkę 'isOpended' na 'isOpened'
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # POPRAWKA: `frame` jest już tablicą NumPy; usunięto `.get_data()`
        img_bgr = frame
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 6) Przetwarzanie wstępne
        augmented = preprocessing(image=img_rgb)
        tensor_img = augmented['image'].unsqueeze(0).to(device)  # 1xCxHxW

        # 7) Inferencja i upsampling
        with torch.no_grad():
            logits = model(tensor_img)
        if isinstance(logits, dict):
            logits = logits['out']
        logits = F.interpolate(logits, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
        preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

        # 8) Koloryzacja maski i nałożenie na obraz
        mask_rgb = colors[preds]
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.5, mask_bgr, 0.5, 0)

        # Wyświetlanie obrazu z nałożoną maską i oryginalnego obrazu obok siebie
        # `img_rgb` jest konwertowany z powrotem do BGR, aby `hstack` i `imshow` działały poprawnie
        images = np.hstack((overlay, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)))

        # 9) Wyświetlanie
        cv2.imshow("SegFormer Segmentation", images)
        if cv2.waitKey(1) & 0xFF == 27:  # Naciśnij ESC, aby wyjść
            break

finally:
    # POPRAWKA: Zwolnienie zasobów kamery za pomocą 'cap.release()'
    cap.release()
    cv2.destroyAllWindows()
