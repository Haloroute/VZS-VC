# VZS-VC Project Documentation

Tài liệu này hướng dẫn chi tiết cách cài đặt môi trường, tải các mô hình cấu hình sẵn và thực hiện quá trình suy luận (inference) cho dự án chuyển đổi giọng nói VZS-VC.

## 1. Cài đặt môi trường

Trước tiên, bạn cần cài đặt toàn bộ các thư viện và gói phụ thuộc cần thiết cho dự án. Hãy đảm bảo rằng bạn đã kích hoạt môi trường ảo (venv hoặc conda) trước khi thực hiện.

Chạy lệnh sau trong cửa sổ dòng lệnh (Terminal/Command Prompt):

```bash
pip install -r requirements.txt

```

## 2. Tải các file mô hình (Checkpoints)

Để chuẩn bị cho quá trình suy luận, bạn cần tải các file trọng số mô hình đã được huấn luyện trước.

1. Truy cập vào thư mục lưu trữ trực tuyến: [Tải file mô hình tại đây](https://drive.google.com/drive/u/0/folders/1jiGY7SYRRHXknySP_QF6OYH95HQTyI9h).
2. Tải toàn bộ các file mô hình có trong liên kết trên.
3. Tạo một thư mục có tên `checkpoints` tại thư mục gốc của dự án (nếu chưa có).
4. Di chuyển các file mô hình vừa tải vào thư mục `checkpoints/`.

Cấu trúc thư mục sau khi thiết lập sẽ có dạng:

```text
VZS-VC/
├── checkpoints/
│   ├── [các file mô hình đã tải]
├── requirements.txt
├── inference.py
└── ...

```

## 3. Hướng dẫn suy luận (Inference)

Sau khi đã hoàn tất việc cài đặt thư viện và chuẩn bị các file checkpoints, bạn có thể tiến hành chạy thử nghiệm mô hình bằng cách thực hiện lệnh sau:

```bash
python inference.py

```
