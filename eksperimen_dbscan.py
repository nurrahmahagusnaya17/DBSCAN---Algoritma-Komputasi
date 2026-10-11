import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN as DBSCAN_PEMBANDING
from sklearn.metrics import adjusted_rand_score

plt.switch_backend('Agg')  # simpan grafik ke file tanpa jendela GUI

DIREKTORI = Path(__file__).resolve().parent
DATA = DIREKTORI / 'data'
HASIL = DIREKTORI / 'hasil'
HASIL.mkdir(exist_ok=True)

EPS = 0.30
MINPTS = 5
UKURAN = (25, 50, 100, 200, 400)
PENGULANGAN = 5
BELUM_DIKLASIFIKASIKAN = -2


def baca_csv(nama):
    """Baca dataset CSV tanpa pustaka pengelompokan."""
    with (DATA / nama).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def simpan_csv(nama, header, baris):
    with (HASIL / nama).open('w', encoding='utf-8', newline='') as f:
        penulis = csv.writer(f)
        penulis.writerow(header)
        penulis.writerows(baris)


def simpan_grafik(fig, nama):
    fig.tight_layout()
    fig.savefig(HASIL / nama, dpi=170)
    plt.close(fig)


def tetangga_epsilon(X, indeks, epsilon):
    """Seluruh indeks berjarak <= epsilon, termasuk titik itu sendiri."""
    selisih = X - X[indeks]
    jarak_kuadrat = np.sum(selisih * selisih, axis=1)
    return np.flatnonzero(jarak_kuadrat <= epsilon * epsilon)


def dbscan_dari_nol(X, epsilon, min_pts):
    """DBSCAN orisinal dengan pencarian tetangga brute-force.

    Label -1 = noise, label 0,1,... = klaster. MinPts menghitung titik itu
    sendiri. Titik noise dapat berubah menjadi border point.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError('X harus matriks dua dimensi dan tidak kosong')
    if not np.isfinite(X).all() or epsilon <= 0 or min_pts < 1:
        raise ValueError('Fitur harus finite, epsilon > 0, min_pts >= 1')

    n = len(X)
    label = np.full(n, BELUM_DIKLASIFIKASIKAN, dtype=int)
    telah_dikunjungi = np.zeros(n, dtype=bool)
    id_klaster = 0

    for i in range(n):
        if telah_dikunjungi[i]:
            continue
        telah_dikunjungi[i] = True
        tetangga = tetangga_epsilon(X, i, epsilon)

        # Kepadatan kurang: noise sementara, bisa menjadi border point nanti.
        if len(tetangga) < min_pts:
            label[i] = -1
            continue

        # Core point ditemukan: perluas klaster secara BFS.
        label[i] = id_klaster
        antrean = list(map(int, tetangga))
        sudah_diantre = set(antrean)
        posisi = 0
        while posisi < len(antrean):
            j = antrean[posisi]
            posisi += 1

            if not telah_dikunjungi[j]:
                telah_dikunjungi[j] = True
                kandidat = tetangga_epsilon(X, j, epsilon)
                if len(kandidat) >= min_pts:
                    for k in map(int, kandidat):
                        if k not in sudah_diantre:
                            antrean.append(k)
                            sudah_diantre.add(k)

            if label[j] in (-1, BELUM_DIKLASIFIKASIKAN):
                label[j] = id_klaster

        id_klaster += 1

    assert np.all(label != BELUM_DIKLASIFIKASIKAN)
    return label


def standardisasi_manual(X):
    """Z-score manual; sigma=0 diganti 1 agar tidak membagi nol."""
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - X.mean(axis=0)) / sigma


def ringkasan(label):
    """Kembalikan (jumlah klaster, jumlah noise)."""
    return len(set(label.tolist()) - {-1}), int(np.sum(label == -1))


def uji_kebenaran():
    print('\n=== 3 KASUS UJI MANUAL ===')
    hasil = []
    for nama in ('Uji_1_Satu_Cluster', 'Uji_2_Dua_Cluster', 'Uji_3_Cluster_Noise'):
        data = baca_csv(nama + '.csv')
        X = np.array([[float(r['X']), float(r['Y'])] for r in data])
        harapan = np.array([int(r['Label_Harapan']) for r in data])
        prediksi = dbscan_dari_nol(X, 1.5, 3)
        ari = adjusted_rand_score(harapan, prediksi)
        klaster, noise = ringkasan(prediksi)
        assert np.isclose(ari, 1.0), f'Uji {nama} gagal: {prediksi}'
        print(f'{nama}: label={prediksi.tolist()}, ARI={ari:.3f}, '
              f'klaster={klaster}, noise={noise}')
        hasil.append((nama, klaster, noise, ari, 'LULUS'))
    simpan_csv('hasil_uji_manual.csv',
               ['Kasus', 'Klaster', 'Noise', 'ARI', 'Status'], hasil)


def eksperimen_utama():
    print('\n=== DATA PELANGGAN ===')
    data = baca_csv('Dataset_400.csv')
    X_asli = np.array([[float(r['Frekuensi_Belanja_Bulanan']),
                        float(r['RataRata_Nilai_Transaksi_Rp'])] for r in data])
    X = standardisasi_manual(X_asli)
    print(f'Jumlah data aktual: {len(data)}; fitur=2')
    if len(data) < max(UKURAN):
        raise ValueError('Dataset tidak cukup untuk semua variasi ukuran input')

    label = dbscan_dari_nol(X, EPS, MINPTS)
    pembanding = DBSCAN_PEMBANDING(eps=EPS, min_samples=MINPTS,
                                   algorithm='brute').fit_predict(X)
    ari = adjusted_rand_score(label, pembanding)
    jumlah_klaster, noise = ringkasan(label)
    print(f'ε={EPS}, MinPts={MINPTS}: klaster={jumlah_klaster}, '
          f'noise={noise}, ARI pembanding={ari:.4f}')
    assert np.isclose(ari, 1.0), 'Partisi klaster berbeda dari pembanding'

    nilai, jumlah = np.unique(label, return_counts=True)
    simpan_csv('ringkasan_klaster.csv', ['Label', 'Jumlah', 'Persentase'],
               [(int(k), int(c), round(100 * c / len(label), 2))
                for k, c in zip(nilai, jumlah)])
    simpan_csv('label_pelanggan.csv',
               ['ID_Pelanggan', 'Frekuensi', 'Rata_Rata_Rp', 'Label_DBSCAN'],
               [(data[i]['ID_Pelanggan'], X_asli[i, 0], X_asli[i, 1], int(label[i]))
                for i in range(len(label))])

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in nilai:
        pilih = label == k
        ax.scatter(X_asli[pilih, 0], X_asli[pilih, 1] / 1000, s=22, alpha=0.75,
                   label='Noise (-1)' if k == -1 else f'Klaster {k}')
    ax.set(xlabel='Frekuensi belanja bulanan (kali)',
           ylabel='Rata-rata nilai transaksi (ribu rupiah)',
           title=f'Pengelompokan {len(data)} pelanggan (ε={EPS}, MinPts={MINPTS})')
    ax.legend(fontsize=8)
    simpan_grafik(fig, 'grafik_klaster.png')
    return X


def uji_waktu(X):
    print('\n=== EKSPERIMEN WAKTU (5 UKURAN x 5 RUN) ===')
    baris = []
    # Standardisasi dilakukan SEBELUM stopwatch agar hanya inti algoritma diukur.
    for n in UKURAN:
        subset = X[:n]
        waktu_manual, waktu_pustaka = [], []
        for _ in range(PENGULANGAN):
            mulai = time.perf_counter_ns()
            label_a = dbscan_dari_nol(subset, EPS, MINPTS)
            waktu_manual.append((time.perf_counter_ns() - mulai) / 1e6)

            mulai = time.perf_counter_ns()
            label_b = DBSCAN_PEMBANDING(eps=EPS, min_samples=MINPTS,
                                        algorithm='brute').fit_predict(subset)
            waktu_pustaka.append((time.perf_counter_ns() - mulai) / 1e6)
            assert np.isclose(adjusted_rand_score(label_a, label_b), 1.0)

        hasil = (n, np.mean(waktu_manual), np.std(waktu_manual, ddof=1),
                 np.mean(waktu_pustaka), np.std(waktu_pustaka, ddof=1))
        baris.append(hasil)
        print(f'n={n}: manual={hasil[1]:.3f} ± {hasil[2]:.3f} ms; '
              f'pustaka={hasil[3]:.3f} ± {hasil[4]:.3f} ms')

    simpan_csv('hasil_waktu.csv',
               ['n', 'Manual_Rata2_ms', 'Manual_SD_ms',
                'Pustaka_Rata2_ms', 'Pustaka_SD_ms'], baris)

    nilai = np.array(baris, dtype=float)
    teori = nilai[:, 0] ** 2
    teori = teori / teori[-1] * nilai[-1, 1]  # diskalakan, bukan pengukuran
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(nilai[:, 0], nilai[:, 1], yerr=nilai[:, 2], marker='o',
                capsize=3, label='DBSCAN buatan sendiri')
    ax.errorbar(nilai[:, 0], nilai[:, 3], yerr=nilai[:, 4], marker='s',
                capsize=3, label='DBSCAN scikit-learn (pembanding)')
    ax.plot(nilai[:, 0], teori, '--', label='Referensi O(n²), diskalakan')
    ax.set(xlabel='Jumlah data (n)', ylabel='Waktu eksekusi (ms)',
           title='Waktu eksekusi dan kurva referensi teoretis')
    ax.legend(fontsize=8)
    simpan_grafik(fig, 'grafik_waktu.png')


def uji_sensitivitas(X):
    print('\n=== SENSITIVITAS DUA PARAMETER ===')
    baris = []
    for epsilon in (0.15, 0.30, 0.50):
        for min_pts in (3, 5, 10):
            klaster, noise = ringkasan(dbscan_dari_nol(X, epsilon, min_pts))
            baris.append((epsilon, min_pts, klaster, noise))
            print(f'epsilon={epsilon:.2f}, MinPts={min_pts}: '
                  f'{klaster} klaster; {noise} noise')
    simpan_csv('hasil_sensitivitas.csv',
               ['epsilon', 'MinPts', 'Klaster', 'Noise'], baris)


def main():
    uji_kebenaran()
    X = eksperimen_utama()
    uji_waktu(X)
    uji_sensitivitas(X)
    print('\nSemua pengujian selesai. Periksa folder hasil/.')


if __name__ == '__main__':
    main()
