import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # type: ignore
import os

print("="*60)
print("ÇİÇEK TAHMİN SİSTEMİ (RENK ANALİZİ İLE)")
print("="*60)

# Model ve sınıf isimlerini yükle
if not os.path.exists('basit_cicek_modeli.h5'):
    print("\n❌ HATA: Model dosyası bulunamadı!")
    print("Önce 'odev.py' dosyasını çalıştırarak modeli eğitin.")
    exit()

if not os.path.exists('class_names.npy'):
    print("\n❌ HATA: Sınıf isimleri dosyası bulunamadı!")
    print("Önce 'odev.py' dosyasını çalıştırarak modeli eğitin.")
    exit()

print("\nModel yükleniyor...")
model = tf.keras.models.load_model('basit_cicek_modeli.h5')
class_names = np.load('class_names.npy', allow_pickle=True)
print(f"✓ Model yüklendi")
print(f"✓ Tanınan çiçekler: {list(class_names)}\n")

# -------------------------------------------------------
# Renk Tespit Fonksiyonu
# -------------------------------------------------------
def detect_dominant_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_ranges = {
        'Kırmızı': [(0, 100, 100), (10, 255, 255)],
        'Kırmızı2': [(160, 100, 100), (180, 255, 255)],
        'Turuncu': [(10, 100, 100), (25, 255, 255)],
        'Sarı': [(25, 100, 100), (35, 255, 255)],
        'Yeşil': [(35, 100, 100), (85, 255, 255)],
        'Mavi': [(85, 100, 100), (130, 255, 255)],
        'Mor': [(130, 100, 100), (160, 255, 255)],
        'Pembe': [(140, 50, 100), (170, 255, 255)],
        'Beyaz': [(0, 0, 200), (180, 30, 255)],
    }

    color_scores = {}

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        score = np.sum(mask) / 255

        if 'Kırmızı' in color_name:
            if 'Kırmızı' not in color_scores:
                color_scores['Kırmızı'] = 0
            color_scores['Kırmızı'] += score
        else:
            color_scores[color_name] = score

    if color_scores:
        dominant_color = max(color_scores, key=color_scores.get)
        return dominant_color

    return "Belirsiz"

# -------------------------------------------------------
# Tahmin Fonksiyonu
# -------------------------------------------------------
def predict_flower(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    # Görüntüyü yükle
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)

    # Renk tespiti
    img_for_color = img_array.astype("uint8")
    color = detect_dominant_color(img_for_color)

    # Model için hazırla
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    predictions = model.predict(img_array, verbose=0)
    predictions = tf.nn.softmax(predictions).numpy()

    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    # 🌈 Renk grafiği için renk map'i
    color_map = {
        "Kırmızı": "#FF0000",
        "Turuncu": "#FF7F00",
        "Sarı": "#FFFF00",
        "Yeşil": "#00FF00",
        "Mavi": "#0000FF",
        "Mor": "#800080",
        "Pembe": "#FF69B4",
        "Beyaz": "#FFFFFF",
        "Belirsiz": "#888888"
    }

    # GRAFİKLER
    plt.figure(figsize=(18, 6))

    # 1) Görüntü
    plt.subplot(1, 3, 1)
    img_display = image.load_img(image_path)
    plt.imshow(img_display)
    plt.axis("off")
    plt.title("Analiz Edilen Görüntü", fontsize=14, fontweight="bold")

    # 2) Sınıf olasılıkları
    plt.subplot(1, 3, 2)
    colors_bar = ["green" if i == predicted_class_idx else "skyblue"
                  for i in range(len(class_names))]
    plt.barh(class_names, predictions[0], color=colors_bar)
    plt.xlabel("Olasılık", fontsize=12)
    plt.title("Sınıf Olasılıkları", fontsize=14, fontweight="bold")
    plt.xlim([0, 1])

    for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
        plt.text(prob + 0.02, i, f"{prob*100:.1f}%", va="center", fontsize=9)

    # 3) Renk grafiği
    plt.subplot(1, 3, 3)
    plt.bar([0], [1], color=color_map.get(color, "#888888"))
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Dominant Renk: {color}", fontsize=14, fontweight="bold")
    plt.box(False)

    plt.tight_layout()

    # Kaydet
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"prediction_{base_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {output_file}")
    plt.show()

    # Konsol çıktısı
    print("\n" + "="*60)
    print("TAHMİN SONUCU")
    print("="*60)
    print(f"🌸 Çiçek Türü: {predicted_class.upper()}")
    print(f"🎨 Renk: {color}")
    print(f"📊 Güven Skoru: {confidence:.2f}%")
    print("="*60)
    print("\nTüm Olasılıklar:")
    for i, class_name in enumerate(class_names):
        marker = "👉" if i == predicted_class_idx else "  "
        print(f"{marker} {class_name:15s}: {predictions[0][i]*100:6.2f}%")
    print("="*60 + "\n")

    return predicted_class, confidence, color

# -------------------------------------------------------
# Toplu tahmin
# -------------------------------------------------------
def predict_multiple_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Klasör bulunamadı: {folder_path}")
        return
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in extensions]
    
    if not image_files:
        print(f"❌ '{folder_path}' klasöründe görüntü bulunamadı!")
        return
    
    print(f"\n{len(image_files)} görüntü bulundu. Analiz ediliyor...\n")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        print(f"📸 {img_file}")
        print("-" * 60)
        
        pred_class, confidence, color = predict_flower(img_path)
        results.append({
            'dosya': img_file,
            'tür': pred_class,
            'renk': color,
            'güven': confidence
        })
    
    print("\n" + "="*60)
    print("TOPLU TAHMİN ÖZETİ")
    print("="*60)
    print(f"{'Dosya':<25} {'Tür':<15} {'Renk':<10} {'Güven':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['dosya']:<25} {r['tür']:<15} {r['renk']:<10} {r['güven']:>5.1f}%")
    print("="*60)

# -------------------------------------------------------
# Ana program
# -------------------------------------------------------
if __name__ == "__main__":
    print("\nKullanım Seçenekleri:")
    print("1️⃣  Tek görüntü: Dosya yolu girin")
    print("2️⃣  Toplu tahmin: Klasör yolu girin")
    print("3️⃣  Python'dan: predict_flower('resim.jpg')")
    print("\n" + "="*60)
    
    user_input = input("\nGörüntü veya klasör yolu (boş bırakarak çıkabilirsiniz): ").strip()
    
    if user_input:
        if os.path.isfile(user_input):
            predict_flower(user_input)
        elif os.path.isdir(user_input):
            predict_multiple_images(user_input)
        else:
            print(f"❌ Geçersiz yol: {user_input}")
    else:
        print("\n💡 Programdan çıkıldı.")
        print("İpucu: Python'da şöyle kullanabilirsiniz:")
        print("  >>> from predict import predict_flower")
        print("  >>> predict_flower('cicek.jpg')")
