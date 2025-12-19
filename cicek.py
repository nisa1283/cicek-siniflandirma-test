import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import pathlib
import tarfile
import urllib.request
import cv2

# --- 1. VERİYİ MANUEL VE GARANTİ YOLDAN İNDİRME ---
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
filename = "flower_photos.tgz"
extract_folder = "flower_photos"

print(f"1. Çalışma dizini: {os.getcwd()}")

if not os.path.exists(filename):
    print("Dosya indiriliyor...)")
    urllib.request.urlretrieve(url, filename)
    print("İndirme tamamlandı.")
else:
    print("Zip dosyası zaten var, indirmeye gerek yok.")

if not os.path.exists(extract_folder):
    print("Zip dosyası açılıyor...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    print("Dosyalar çıkartıldı.")
else:
    print("Klasör zaten var.")

data_dir = pathlib.Path(extract_folder)

# --- 2. KLASÖR KONTROLÜ ---
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"\n---> TESPİT EDİLEN RESİM SAYISI: {image_count}")

if image_count == 0:
    print("!!! HATA: Klasör boş! Lütfen 'flower_photos' klasörünü silip kodu tekrar çalıştır.")
    exit()

# --- 3. VERİ SETİNİ YÜKLEME ---
batch_size = 32
img_height = 180
img_width = 180

print("\n2. Veri seti yükleniyor...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(f"Sınıflar: {class_names}")

# --- RENK ANALİZİ FONKSİYONU ---
def detect_dominant_color(image):
    """HSV renk uzayında dominant rengi tespit eder"""
    img = image.numpy().astype("uint8")
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

# --- 4. KANIT: RESİMLERİ VE RENKLERİ GÖSTER ---
print("\n3. Verinin okunduğuna dair KANIT + RENK ANALİZİ...")
plt.figure(figsize=(12, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img_display = images[i].numpy().astype("uint8")
    plt.imshow(img_display)
    color = detect_dominant_color(images[i])
    plt.title(f"{class_names[labels[i]]}\nRenk: {color}", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig('sample_images_with_colors.png', dpi=150, bbox_inches='tight')
plt.show() 

# --- 5. PERFORMANS AYARLARI ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 6. GÜÇLÜ AMA HIZLI MODEL ---
num_classes = len(class_names)

model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  
  # Daha güçlü katmanlar
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.3),  # Hafif dropout
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\n📊 Model Parametreleri:")
model.summary()

# --- 7. EĞİTİM (SABİT EPOCH SAYISI) ---
print("\n4. Eğitim Başlıyor...")
epochs = 10 

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

print("\nModel Kaydediliyor...")
model.save("basit_cicek_modeli.h5")
np.save('class_names.npy', class_names)
print("Sınıf isimleri kaydedildi: class_names.npy")

# --- 8. SONUÇ GRAFİĞİ ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(14, 5))

# Sol: Accuracy
plt.subplot(1, 3, 1)
plt.plot(epochs_range, acc, label='Training', marker='o', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation', marker='s', linewidth=2)
best_epoch = np.argmax(val_acc)
plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
plt.scatter(best_epoch, val_acc[best_epoch], color='green', s=100, zorder=5)
plt.legend(loc='lower right')
plt.title('Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

# Orta: Loss
plt.subplot(1, 3, 2)
plt.plot(epochs_range, loss, label='Training', marker='o', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation', marker='s', linewidth=2)
plt.legend(loc='upper right')
plt.title('Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Sağ: Overfitting Analizi
plt.subplot(1, 3, 3)
overfitting_per_epoch = [(acc[i] - val_acc[i]) * 100 for i in range(len(acc))]
colors = ['red' if x > 15 else 'orange' if x > 8 else 'green' for x in overfitting_per_epoch]
plt.bar(epochs_range, overfitting_per_epoch, color=colors, alpha=0.7)
plt.axhline(y=8, color='orange', linestyle='--', label='Orta Risk')
plt.axhline(y=15, color='red', linestyle='--', label='Yüksek Risk')
plt.legend()
plt.title('Overfitting Analizi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Train-Val Farkı (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 9. DETAYLI PERFORMANS ÖZETİ ---
best_val_acc = max(val_acc)
best_epoch_idx = np.argmax(val_acc)
best_train_acc = acc[best_epoch_idx]
final_overfitting = (acc[-1] - val_acc[-1]) * 100
best_overfitting = (best_train_acc - best_val_acc) * 100

print("\n" + "="*60)
print("📊 EĞİTİM RAPORU")
print("="*60)
print(f"\n🎯 En İyi Performans (Epoch {best_epoch_idx+1}):")
print(f"   Training Accuracy: {best_train_acc*100:.2f}%")
print(f"   Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"   Overfitting: {best_overfitting:.1f}%", end=" ")
if best_overfitting > 15:
    print("⚠️ Yüksek")
elif best_overfitting > 8:
    print("⚡ Orta")
else:
    print("✅ Düşük")

print(f"\n📈 Final Performans (Epoch {epochs}):")
print(f"   Training Accuracy: {acc[-1]*100:.2f}%")
print(f"   Validation Accuracy: {val_acc[-1]*100:.2f}%")
print(f"   Training Loss: {loss[-1]:.4f}")
print(f"   Validation Loss: {val_loss[-1]:.4f}")
print(f"   Overfitting: {final_overfitting:.1f}%", end=" ")
if final_overfitting > 15:
    print("⚠️ Yüksek - Daha erken durun")
elif final_overfitting > 8:
    print("⚡ Orta - İdeal nokta")
else:
    print("✅ Düşük - Daha fazla eğitilebilir")

print("\n" + "="*60)
print("💡 ÖNERİLER:")
if val_acc[-1] < val_acc[best_epoch_idx] - 0.05:
    print("   ⚠️  Overfitting var! Epoch sayısını azaltın ")
elif final_overfitting < 5:
    print("   ✅ Model daha fazla öğrenebilir! Epoch artırın")
else:
    print("   ✅ Model optimal durumda!")
print("="*60)

print("\nKaydedilen dosyalar:")
print("  ✓ basit_cicek_modeli.h5")
print("  ✓ class_names.npy")
print("  ✓ sample_images_with_colors.png")
print("  ✓ training_results.png")
print("\n🧪 Şimdi test_et.py ile tahmin yapabilirsiniz!")
print("="*60)