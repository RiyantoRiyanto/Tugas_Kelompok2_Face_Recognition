# Tugas_Kelompok2_Face_Recognition


## Pendahuluan
Deepface adalah suatu framework untuk pengenalan wajah (face recognition) yang membungkus state-of-the-art model-model pengenalan wajah yang telah dikembangkan sebelumnya seperti VGG-face, Google FaceNet, OpenFace, Facebook Deepface, DeepID, ArcFace dan Dlib. Pada tugas ini anda akan mengeksplorasi Deepface Library sehingga mendapatkan prosedur dan konfigurasi terbaik untuk task pengenalan wajah.

### Menginstal library Deepface
Untuk menggunakan Deepface terlebih dahulu menginstall library Deepface sebagai berikut:

!pip install deepface
from deepface import DeepFace

### Menyiapkan sample dataset
Menyiapkan foto-foto yang akan digunakan sebagai database referensi dan data pengujian. Data pengujian terdiri dari data tes anggota kelas dan bukan anggota kelas. Sample dataset dapat disimpan di GDrive, cara mengaksesnya sebagai berikut:

from google.colab import drive
drive.mount('/content/drive')

### Verifikasi Wajah
Untuk membandingkan kesamaan dua wajah dilakukan dengan menginput dua gambar yang akan dibandingkan, misalnya img1.jpg dan img2.jpg

img1_path = 'drive/MyDrive/dataset/test_data_anggota/Adiyasa nurfalah-2.jpeg'
img2_path = 'drive/MyDrive/dataset/database/Adiyasa nurfalah.jpeg'
df = DeepFace.verify(img1_path, img2_path)
if df['verified'] == True:
  print('Hasil: Wajah yang sama')
else:
  print('Hasil: Wajah yang beda')

### Pengenalan Wajah
Pada Deepface pengenalan wajah melakukan pencarian gambar yang tersimpan dalam folder database yang berisi kumpulan gambar referensi.

img_path = 'drive/MyDrive/dataset/test_data_anggota/Arief sartono 10.jpg'
df = DeepFace.find(img_path, db_path = img_database_path)
print('Gambar yang mirip: ' + df.iloc[0,0])

## Eksplorasi
Beberapa eksplorasi yang dapat dilakukan antara lain:
•	Model yang digunakan
•	Metrik yang digunakan
•	Detektor wajah yang digunakan

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

Berikut fungsi yang dipanggil untuk menghitung akurasi:

def hitung_akurasi(nama_model, nama_detector, nama_metrik):
  jumlah_benar = 0
  jumlah_total = 0

  for img_test in files_test_anggota:
    df = DeepFace.find(img_test, img_database_path, model_name = nama_model, detector_backend = nama_detector, distance_metric = nama_metrik, silent = True, enforce_detection = False)
    if df.empty == False:  
      img_inp = os.path.basename(img_test).split(' ');
      img_out = os.path.basename(df.iloc[0,0]).split(' ');
      if img_inp[0] == img_out[0]:  
        jumlah_benar = jumlah_benar + 1;  
    jumlah_total = jumlah_total + 1; 

  for img_test in files_test_non_anggota:
    df = DeepFace.find(img_test, img_database_path, model_name = nama_model, detector_backend = nama_detector, distance_metric = nama_metrik, silent = True, enforce_detection = False)
    if df.empty:
      jumlah_benar = jumlah_benar + 1; 
    jumlah_total = jumlah_total + 1;
    
  akurasi = 100 * jumlah_benar / jumlah_total;
  return akurasi;

### Eksplorasi Model
Berikut adalah eksplorasi model-model Deepface dengan metrics = "cosine" dan backends detektor = "opencv"

import matplotlib.pyplot as plt
nama_detektor = backends[0]
nama_metrik = metrics[0]
tabel_akurasi_model = []
for model in models:
  akurasi = hitung_akurasi(model, nama_detektor, nama_metrik);
  tabel_akurasi_model.append(akurasi)
  print('Akurasi model ' + model + '  ' + str(akurasi))
plt.bar(models, tabel_akurasi_model)
plt.ylabel('Akurasi')
plt.show()

### Eksplorasi Detektor Wajah
Detektor wajah sangat penting dalam pengenalan wajah, oleh karena itu detektor wajah perlu dieksplorasi untuk mendapatkan hasil terbaik.

idx_model_terbaik = tabel_akurasi_model.index(max(tabel_akurasi_model))
nama_model = models[idx_model_terbaik]
#nama_model = models[0]
nama_metrik = metrics[0]
tabel_akurasi_detektor = []
for nama_detektor in backends:
  akurasi = hitung_akurasi(nama_model, nama_detektor, nama_metrik);
  tabel_akurasi_detektor.append(akurasi)
  print('Akurasi menggunakan detektor ' + nama_detektor + '  ' + str(akurasi))
plt.bar(backends, tabel_akurasi_detektor)
plt.ylabel('Akurasi')
plt.show()

### Eksplorasi Metrik Jarak
Metrik jarak digunakan untuk mengukur kesamaan dari gambar test dengan gambar pada database. Berikut perbandingan hasil akurasi menggunakan metrik jarak yang berbeda:

id_model_terbaik = tabel_akurasi_model.index(max(tabel_akurasi_model))
id_detektor_terbaik = tabel_akurasi_detektor.index(max(tabel_akurasi_detektor))
nama_model = models[id_model_terbaik]
nama_detektor = backends[id_detektor_terbaik]
#nama_model = models[1]
#nama_detektor = backends[2]
tabel_akurasi_metrik = []
for nama_metrik in metrics:
  akurasi = hitung_akurasi(nama_model, nama_detektor, nama_metrik);
  tabel_akurasi_metrik.append(akurasi)
  print('Akurasi menggunakan metrik ' + nama_metrik + '  ' + str(akurasi))
plt.bar(metrics, tabel_akurasi_metrik)
plt.ylabel('Akurasi')
plt.show()

## Hasil Terbaik
Berdasarkan eksplorasi yang dilakukan diperoleh konfigurasi terbaik sebagai berikut:
•	Nama model : Facenet
•	Nama detektor : dlib
•	Nama metrik: cosine
dengan akurasi 72.8%
