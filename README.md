## Dự đoán ngữ nghĩa của từ theo ngữ cảnh (Word in Context - WiC)

### Chuẩn bị môi trường
- Môi trường phát triển Windows 10 (64-bit)
- Ngôn ngữ lập trình: Python 3.7:
Tải và cải đặt phiên bản python thích hợp tại địa chỉ https://www.python.org/downloads/
- Thư viện cần cài đặt: tensorflow, keras, numpy, matplotlib, allennlp, nltk.  Cài đặt các thư viện lần lượt bằng các lệnh sau:
```
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install allennlp
pip install nltk
```


### Các bước chạy chương trình

#### Xử lý dữ liệu
Chạy file "process_data.py" để nạp dữ liệu, xử lý và lưu dữ liệu đã xử lý ra thư mục "processed_data":
```
python process_data.py
```
*Thời gian ở bước này có thể rất lâu.

#### Học mô hình
Chạy file "neumf.py" để học mô hình và lưu các trọng số mô hình ra file "save_model_neumf.hdf5":
```
python neumf.py
```

#### Dự đoán
Chạy file "predict.py" để dự đoán dữ liệu trong tập kiểm thử và xuất kết quả ra file "output.txt":
```
python predict.py
```

### Quá trình đánh giá lựa chọn tham số

#### Tham số dropout
Chạy file "dropout.py" để khảo sát các giá trị dropout tiềm năng, lưu độ chính xác và giá trị lỗi ra các file "acc.npy" và "loss.npy":
```
python assessment/dropout/dropout.py
```
Chạy file "draw.py" để nạp các file "acc.npy" và "loss.npy" và vẽ đồ thị:
```
python assessment/dropout/draw.py
```
Chạy file "neumf.py" để học lại mô hình và dự đoán trên tập kiểm thử với tham số dropout vừa chọn:
```
python assessment/dropout/neumf.py
```

#### Tham số kernel_regularizer
Chạy file "kernel_regularizer.py" để khảo sát các giá trị kernel_regularizer tiềm năng, lưu độ chính xác và giá trị lỗi ra các file "acc.npy" và "loss.npy":
```
python assessment/kernel_regularizer/kernel_regularizer.py
```
Chạy file "draw.py" để nạp các file "acc.npy" và "loss.npy" và vẽ đồ thị:
```
python assessment/kernel_regularizer/draw.py
```
Chạy file "neumf.py" để học lại mô hình và dự đoán trên tập kiểm thử với tham số kernel_regularizer vừa chọn:
```
python assessment/kernel_regularizer/neumf.py
```

#### Số lượng các lớp MLP
Chạy file "number_layer.py" để khảo sát các số lượng lớp MLP tiềm năng, lưu độ chính xác và giá trị lỗi ra các file "acc.npy" và "loss.npy":
```
python assessment/number_layer/number_layer.py
```
Chạy file "draw.py" để nạp các file "acc.npy" và "loss.npy" và vẽ đồ thị:
```
python assessment/number_layer/draw.py
```
Chạy file "neumf.py" để học lại mô hình và dự đoán trên tập kiểm thử với các lớp MLP vừa chọn:
```
python assessment/number_layer/neumf.py
```

#### Kích thước vectơ tiềm ẩn
Chạy file "latent_dim.py" để khảo sát kích thước vectơ tiềm ẩn tiềm năng, lưu độ chính xác và giá trị lỗi ra các file "acc.npy" và "loss.npy":
```
python assessment/latent_dim/latent_dim.py
```
Chạy file "draw.py" để nạp các file "acc.npy" và "loss.npy" và vẽ đồ thị:
```
python assessment/latent_dim/draw.py
```


#### Sử dụng ví dụ âm tính
Chạy file "negative_sample.py" để khảo sát số lượng các ví dụ tiềm ẩn, lưu độ chính xác và giá trị lỗi ra file "results.npy":
```
python assessment/negative_sample/negative_sample.py
```
Chạy file "draw.py" để nạp file "results.npy" và vẽ đồ thị:
```
python assessment/negative_sample/draw.py
```
