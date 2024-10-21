import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Veri setini yükleyelim
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Eğitim ve test veri setinin boyutlarına bakalım
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# İlk resmi oluşturuyorux
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Veriler normalize ediliyor (0-1 arasına getirme)
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN için veriyi 4D hale getirme (örnek sayısı, yükseklik, genişlik, kanal)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Veriyi 2D düzleme getiriyoruz (KNN için gereklidir)
X_train_knn = X_train.reshape(-1, 28*28)
X_test_knn = X_test.reshape(-1, 28*28)

# Modeli tanımlayalım ve eğitelim
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_knn, y_train)

# Tahmin yapalım
y_pred = knn.predict(X_test_knn)

# Modelin doğruluğunu hesaplayalım
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN model accuracy: {accuracy * 100:.2f}%")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CNN modeli oluşturma
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 sınıf (0-9 rakamları)
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirelim
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")
import numpy as np

# Tek bir örneği seçip tahmin yapalım
sample_image = X_test[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)
print(f"Predicted label: {predicted_label}")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Yeni verilerle modeli eğitme
datagen.fit(X_train)
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15, validation_data=(X_test, y_test))

model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV dosyasını yükle
train_df = pd.read_csv('train.csv')

# Veri setinin ilk satırına bakalım
print(train_df.head())
# Etiketi ayıralım ve piksel verilerini alalım
labels = train_df['label']
pixels = train_df.drop(columns=['label'])

# İlk görüntüyü alalım
first_image = pixels.iloc[0].values
first_image = first_image.reshape(28, 28)  # 28x28 boyutuna getir

# Görüntüyü gösterelim
plt.imshow(first_image, cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()
# Tüm görüntüleri (pikselleri) normalize edelim (0-255 aralığındaki değerleri 0-1 aralığına getir)
X_train = pixels.values / 255.0  # Normalizasyon
y_train = labels.values  # Etiketler

# Görüntüleri 28x28 boyutunda tekrar şekillendirelim (CNN için 4D veriye ihtiyaç var: [num_images, height, width, channels])
X_train = X_train.reshape(-1, 28, 28, 1)

print(f"Veri şekli: {X_train.shape}, Etiket şekli: {y_train.shape}")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CNN modeli oluşturma
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 sınıf (0-9 rakamları)
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Test CSV'sini yükle
test_df = pd.read_csv('test.csv')

# Test veri setini 28x28 boyutuna getirelim
X_test = test_df.values / 255.0  # Normalizasyon
X_test = X_test.reshape(-1, 28, 28, 1)

# Modeli test veri seti ile tahmin yapalım
predictions = model.predict(X_test)

# Tahmin edilen sınıflar (en yüksek olasılığa sahip sınıf)
predicted_labels = np.argmax(predictions, axis=1)

print(predicted_labels[:10])  # İlk 10 tahmini göster