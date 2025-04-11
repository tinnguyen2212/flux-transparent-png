# Hướng dẫn sử dụng Flux.1 Transparent PNG Generator

## Giới thiệu

Flux.1 Transparent PNG Generator là một giải pháp toàn diện cho việc huấn luyện và tạo ra các hình ảnh PNG trong suốt (không có nền) sử dụng mô hình Flux.1 dev. Giải pháp này dựa trên kiến trúc của dự án Microsoft ART-MSRA nhưng được điều chỉnh đặc biệt cho việc tạo ra hình ảnh PNG trong suốt một lớp.

## Cách sử dụng với Google Colab

### Bước 1: Truy cập Google Colab Notebooks

Cách đơn giản nhất để sử dụng giải pháp này là thông qua Google Colab. Bạn có thể truy cập các notebook sau:

1. [Notebook huấn luyện](https://colab.research.google.com/github/your-username/flux-transparent-png/blob/main/colab/train_transparent_png.ipynb) - Dùng để huấn luyện mô hình VAE trên dữ liệu PNG trong suốt
2. [Notebook tạo hình ảnh](https://colab.research.google.com/github/your-username/flux-transparent-png/blob/main/colab/generate_transparent_png.ipynb) - Dùng để tạo ra hình ảnh PNG trong suốt từ mô hình đã huấn luyện

### Bước 2: Kết nối Google Drive

Trong mỗi notebook, bạn cần kết nối với Google Drive để lưu trữ dữ liệu và mô hình:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 3: Cấu hình đường dẫn

Đảm bảo rằng bạn đã cấu hình đúng các đường dẫn sau:

- Dữ liệu huấn luyện: `/content/drive/MyDrive/SD-Data/TrainData/4000_PNG/TEST`
- Thư mục lưu mô hình: `/content/drive/MyDrive/VAE-DECODER`
- Thư mục lưu hình ảnh tạo ra: `/content/drive/MyDrive/VAE-DECODER/OUT`

### Bước 4: Huấn luyện mô hình

Trong notebook huấn luyện, bạn có thể điều chỉnh các tham số như:
- `BATCH_SIZE`: Kích thước batch (mặc định: 8)
- `NUM_EPOCHS`: Số epoch huấn luyện (mặc định: 100)
- `LEARNING_RATE`: Tốc độ học (mặc định: 1e-4)
- `ALPHA_WEIGHT`: Trọng số cho kênh alpha (mặc định: 2.0)
- `IMAGE_SIZE`: Kích thước hình ảnh (mặc định: 512)

Sau đó chạy các cell trong notebook để huấn luyện mô hình.

### Bước 5: Tạo hình ảnh PNG trong suốt

Trong notebook tạo hình ảnh, bạn có thể:
- Định nghĩa các prompt mô tả hình ảnh bạn muốn tạo
- Điều chỉnh các tham số như kích thước hình ảnh, guidance scale, số bước suy luận
- Tạo và hiển thị hình ảnh PNG trong suốt

## Cách sử dụng trên máy tính cá nhân

### Bước 1: Clone repository

```bash
git clone https://github.com/your-username/flux-transparent-png.git
cd flux-transparent-png
```

### Bước 2: Cài đặt dependencies

```bash
cd python
python install.py
```

### Bước 3: Huấn luyện mô hình

```bash
python main.py train \
  --data_dir "/content/drive/MyDrive/SD-Data/TrainData/4000_PNG/TEST" \
  --output_dir "/content/drive/MyDrive/VAE-DECODER" \
  --batch_size 8 \
  --num_epochs 100
```

### Bước 4: Lưu mô hình đã huấn luyện

```bash
python main.py save \
  --checkpoint_path "/content/drive/MyDrive/VAE-DECODER/checkpoints/checkpoint_epoch_100.pt" \
  --output_dir "/content/drive/MyDrive/VAE-DECODER"
```

### Bước 5: Tạo hình ảnh PNG trong suốt

```bash
python main.py generate \
  --model_path "/content/drive/MyDrive/VAE-DECODER/transparent_vae.pt" \
  --prompt "Một bông hoa đẹp trên nền trong suốt" \
  --output_dir "/content/drive/MyDrive/VAE-DECODER/OUT"
```

## Tích hợp với ComfyUI

### Bước 1: Cài đặt ComfyUI

Tải và cài đặt [ComfyUI](https://github.com/comfyanonymous/ComfyUI) theo hướng dẫn trên trang chủ.

### Bước 2: Cài đặt các node tùy chỉnh

Sao chép file `comfyui_transparent_png_nodes.py` vào thư mục `custom_nodes` trong cài đặt ComfyUI của bạn.

### Bước 3: Cài đặt mô hình

Sao chép các mô hình VAE và decoder đã huấn luyện vào thư mục `models/transparent_vae` trong cài đặt ComfyUI của bạn.

### Bước 4: Sử dụng các node trong ComfyUI

Khởi động ComfyUI và sử dụng các node sau:
- **Load Transparent VAE**: Tải mô hình VAE trong suốt đã huấn luyện
- **Load Transparent Decoder**: Tải mô hình decoder trong suốt đã huấn luyện
- **Encode with Transparent VAE**: Mã hóa hình ảnh thành không gian tiềm ẩn
- **Decode with Transparent VAE**: Giải mã không gian tiềm ẩn thành hình ảnh trong suốt
- **Decode with Transparent Decoder**: Giải mã sử dụng chỉ decoder
- **Save Transparent PNG**: Lưu hình ảnh dưới dạng PNG trong suốt
- **Generate Transparent PNG**: Tạo hình ảnh trong suốt từ prompt

## Ví dụ workflow trong ComfyUI

1. Thêm node **Load Transparent VAE** và chọn mô hình VAE đã huấn luyện
2. Thêm node **Generate Transparent PNG** và kết nối với node VAE
3. Đặt prompt để mô tả hình ảnh bạn muốn tạo
4. Thêm node **Save Transparent PNG** và kết nối với đầu ra của node Generate
5. Chạy workflow để tạo và lưu hình ảnh PNG trong suốt

## Tùy chỉnh

Bạn có thể tùy chỉnh nhiều khía cạnh của pipeline:

- **Alpha Weight**: Điều chỉnh tham số `alpha_weight` để kiểm soát tầm quan trọng của việc bảo toàn tính trong suốt
- **Kích thước hình ảnh**: Thay đổi tham số `height` và `width` để tạo hình ảnh có kích thước khác nhau
- **Guidance Scale**: Điều chỉnh tham số `guidance_scale` để kiểm soát mức độ tuân thủ prompt
- **Số bước suy luận**: Điều chỉnh tham số `num_inference_steps` để kiểm soát chất lượng và tốc độ tạo hình ảnh

## Xử lý sự cố

- **CUDA Out of Memory**: Giảm batch size hoặc kích thước hình ảnh
- **Huấn luyện chậm**: Giảm số lượng workers hoặc sử dụng tập dữ liệu nhỏ hơn
- **Tính trong suốt kém**: Tăng alpha weight trong quá trình huấn luyện
- **Vấn đề khi tạo hình ảnh**: Thử sử dụng VAE đầy đủ thay vì chỉ decoder

## Liên hệ và hỗ trợ

Nếu bạn gặp bất kỳ vấn đề nào hoặc có câu hỏi, vui lòng tạo issue trên [GitHub repository](https://github.com/your-username/flux-transparent-png/issues).
