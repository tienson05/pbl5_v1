"""
Biến đổi trong giai đoạn Validation và Test
Trong giai đoạn validation và testing, một pipeline tiền xử lý xác định (deterministic preprocessing pipeline) được sử dụng để đảm bảo việc đánh giá hiệu năng nhất quán và có thể tái lập.

Pipeline này chỉ bao gồm:
    resize ảnh về 224 × 224
    chuyển sang grayscale
    chuyển sang tensor
    chuẩn hóa dữ liệu

Không áp dụng bất kỳ kỹ thuật data augmentation nào trong các giai đoạn này.
"""
name = "001_1_l"
print(name.split("_"))
is_train = name.split("_")[0] == "1"
print(is_train)
print(name)