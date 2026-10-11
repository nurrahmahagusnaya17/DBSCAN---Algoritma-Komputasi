# Implementasi Algoritma DBSCAN untuk Pengelompokan Pelanggan Toko

Proyek ini merupakan tugas mata kuliah Analisis Algoritma Komputasi. Algoritma Density-Based Spatial Clustering of Applications with Noise (DBSCAN) diimplementasikan secara mandiri menggunakan Python untuk mengelompokkan pelanggan berdasarkan frekuensi belanja bulanan dan rata-rata nilai transaksi. Implementasi `scikit-learn` hanya digunakan sebagai pembanding, bukan sebagai algoritma utama.

## Struktur Proyek
DBSCAN_Pelanggan_Toko/
├── eksperimen_dbscan.py
├── requirements.txt
├── README.md
├── data/
│   ├── Dataset_400.csv
│   ├── Uji_1_Satu_Cluster.csv
│   ├── Uji_2_Dua_Cluster.csv
│   └── Uji_3_Cluster_Noise.csv
└── hasil/
    ├── hasil_uji_manual.csv
    ├── label_pelanggan.csv
    ├── ringkasan_klaster.csv
    ├── hasil_waktu.csv
    ├── hasil_sensitivitas.csv
    ├── grafik_klaster.png
    └── grafik_waktu.png

Lingkungan dan Dependensi
Program dikembangkan melalui Visual Studio Code pada macOS. Pengujian menggunakan perangkat MacBook Neo dengan RAM 8 GB. Bahasa pemrograman adalah Python 3, dengan dependensi yang tercantum di `requirements.txt`:

```text
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
```

Cara Menjalankan
1. Ekstrak ZIP proyek, lalu buka folder `DBSCAN_Pelanggan_Toko` di Visual Studio Code.
2. Buka terminal pada folder proyek.
3. Instal dependensi dan jalankan program:

```bash
python3 -m pip install -r requirements.txt
python3 eksperimen_dbscan.py
```

Program membaca dataset dari folder `data/` dan membuat atau memperbarui hasil pengujian di folder `hasil/`.

Dataset dan Parameter
Dataset utama berisi 400 pelanggan sintetis. Dua fitur yang digunakan dalam perhitungan jarak adalah `Frekuensi_Belanja_Bulanan` dan `RataRata_Nilai_Transaksi_Rp`. Kedua fitur distandardisasi dengan Z-score sebelum klasterisasi. ID pelanggan hanya digunakan sebagai identitas, sedangkan total pengeluaran tidak dimasukkan sebagai fitur.
Parameter utama adalah ε = 0,30 dan MinPts = 5 dengan jarak Euclidean. MinPts menghitung titik yang diperiksa itu sendiri. Pencarian tetangga dilakukan secara brute-force dan ekspansi klaster menggunakan antrean.

## Tahapan Pengujian
1.Pengujian fungsional: tiga dataset kecil untuk satu klaster, dua klaster, dan klaster dengan noise (ε = 1,5; MinPts = 3). Hasil dibandingkan dengan label harapan menggunakan Adjusted Rand Index (ARI).
2.Pengelompokan utama: memproses 400 pelanggan, menyimpan label dan ringkasan anggota klaster, lalu memeriksa kesesuaian dengan `scikit-learn` DBSCAN (`algorithm='brute'`).
3.Pengujian waktu: ukuran input 25, 50, 100, 200, dan 400; masing-masing lima pengulangan. Waktu diukur dengan `time.perf_counter_ns()` dan dilaporkan sebagai rata-rata ± simpangan baku sampel. Grafik disandingkan dengan kurva referensi O(n²) yang diskalakan.
4.Analisis sensitivitas: menguji ε = 0,15; 0,30; 0,50 dan MinPts = 3; 5; 10.

## Contoh Keluaran
Berdasarkan hasil eksperimen yang disertakan dalam paket proyek, pengujian pada 400 pelanggan dengan ε = 0,30 dan MinPts = 5 menghasilkan 4 klaster dan 8 noise (2,00%), dengan ARI = 1,00 terhadap `scikit-learn`. Ketiga kasus uji fungsional juga menghasilkan ARI = 1,00. Pada pengukuran yang tercatat untuk 400 data, rata-rata waktu implementasi mandiri adalah 5,59 ms, sedangkan pembanding 1,14 ms. Waktu dapat berbeda ketika program dijalankan ulang pada kondisi perangkat yang berbeda.

Berkas hasil yang dibuat oleh program:

| Berkas | Isi |
|---|---|
| `hasil_uji_manual.csv` | Hasil tiga pengujian fungsional |
| `label_pelanggan.csv` | Label setiap pelanggan |
| `ringkasan_klaster.csv` | Jumlah dan persentase anggota tiap kelompok |
| `hasil_waktu.csv` | Rata-rata serta simpangan baku waktu eksekusi |
| `hasil_sensitivitas.csv` | Jumlah klaster dan noise untuk sembilan kombinasi parameter |
| `grafik_klaster.png` | Visualisasi hasil klasterisasi |
| `grafik_waktu.png` | Perbandingan waktu dan kurva O(n²) |

## Catatan Metodologis
1.Inti DBSCAN ditulis dalam fungsi `tetangga_epsilon()` dan `dbscan_dari_nol()` tanpa memanggil fungsi klasterisasi siap pakai.
2.Standardisasi dihitung dari dataset 400 pelanggan terlebih dahulu; eksperimen ukuran input menggunakan prefiks `X[:n]` dari data yang sudah distandardisasi.
3.Pengukuran waktu hanya mencakup pemanggilan fungsi DBSCAN, bukan pembacaan CSV, standardisasi, atau penyimpanan berkas.
4.Kompleksitas waktu implementasi *brute-force* untuk jumlah fitur tetap adalah Θ(n²); kurva teoretis pada grafik merupakan referensi, bukan hasil pengukuran tersendiri.
5.Dataset bersifat sintetis; hasil klasterisasi tidak langsung dapat digeneralisasikan pada pelanggan toko nyata.

## Referensi Utama

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. Proceedings of KDD-96, 226–231.
