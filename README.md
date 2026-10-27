# UTS Sains Data – Tindak Lanjut EDA ROV

## Isi
- `combined_log_sb(water-quality).csv`: dataset mentah asli.
- `run_uts.py`: pipeline reproducible dari data mentah, sesi, QC, agregasi hingga model HAC.
- `results/`: output CSV, JSON dan grafik.
- `requirements.txt`: dependensi.

## Menjalankan
`pip install -r requirements.txt`
`python run_uts.py --input "combined_log_sb(water-quality).csv" --out results --spline-df 8`

## Demonstrasi sensitivitas
Ubah `--spline-df 8` menjadi `--spline-df 6`, jalankan ulang pada folder output baru, lalu periksa `results/sensitivity.csv` dan `results/summary.json`. Program juga menghitung df=4,6,8,10,12 otomatis.

## Catatan metodologis
Analisis utama menggunakan sesi >180 detik, pH 6.5–8.5, TDS positif dan DO tersedia. Kriteria seleksi bersifat analitis dan dapat menyebabkan bias seleksi; raw data tidak dimodifikasi. OLS dengan model dasar efek sesi+tren waktu spline+kedalaman+kecepatan dan model lengkap yang menambahkan roll, pitch, sin/cos yaw. Uji gabungan menggunakan Wald HAC (maxlags 10), bukan identifikasi kausal. Data karbon biru belum tersedia.

## Penggunaan AI
ChatGPT (OpenAI) membantu penulisan pipeline, laporan, slide, dan pemaparan; hasil dihitung melalui eksekusi skrip pada CSV mentah dan perlu ditinjau oleh peneliti.
