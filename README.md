# OptClust

Repositori dengan tujuan menhitung jumlah kluster optimal menggunakan metode Elbow dan Silhouette pada data yang diberikan. Proyek ini ditulis dalam Python dan menggunakan pustaka seperti Pandas, Matplotlib, dan Scikit-learn.

## Fitur

- Memuat data dari file CSV atau Excel.
- Memverifikasi dan memproses data.
- Menentukan jumlah kluster optimal menggunakan metode Elbow dan Silhouette.
- Menyimpan hasil dalam format JSON dan grafik.

## Instalasi

Pastikan Anda memiliki Python dan pip terinstal di sistem Anda. Kemudian, ikuti langkah-langkah di bawah ini:

1. Clone repositori ini
    ```bash
    git clone https://github.com/syahrulm98/optclust.git
    cd optclust
    ```

2. Buat virtual environment (opsional tetapi direkomendasikan)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk pengguna Unix/macOS
    .\venv\Scripts\activate   # Untuk pengguna Windows
    ```

3. Instal dependensi
    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

1. Pastikan file data yang diperlukan (`Data-Preferensi.xlsx`, `Data-Studi-Kasus.xlsx`, `Data-Preferensi.csv`, `Data-Studi-Kasus.csv`) tersedia di direktori `data/`.

2. Buat direktori `hasil` untuk menyimpan output
    ```bash
    mkdir -p hasil
    ```

3. Jalankan skrip untuk memproses data dan menentukan jumlah kluster optimal menggunakan kedua metode
    ```bash
    python scripts/verif-hitung.py
    ```

4. Jalankan skrip untuk memproses data dan menentukan jumlah kluster optimal menggunakan metode elbow
    ```bash
    python scripts/elbow.py
    ```

3. Jalankan skrip untuk memproses data dan menentukan jumlah kluster optimal menggunakan metode silhouette
    ```bash
    python scripts/silhouette.py
    ```

