# CNN_4Scenario

การปรับแต่งเพื่อใช้งานจริง
Load Audio: แก้ไขฟังก์ชัน load_audio ให้ใช้ librosa.load(path, sr=SAMPLE_RATE)

Denoise: ใน denoise_audio ให้ใช้ไลบรารี noisereduce หรือ Algorithm ที่คุณเลือก

Paths: แก้ไข generate_mock_metadata ให้อ่านไฟล์จริงจาก Folder ของคุณ

โค้ดนี้จะสร้าง Structure บน MLflow UI ที่ดูง่ายมากครับ:

Experiment: Audio_Classification_Research

Run: Baseline_LogMel

Child: Fold_1

Child: Fold_2

...

Run: Proposed_2_Raw_PCEN

Child: Fold_1

...

เมื่อรันเสร็จ คุณสามารถกดดู Artifacts ในแต่ละ Fold เพื่อดูรูป Confusion Matrix และ Training Curves ได้ทันทีครับ
