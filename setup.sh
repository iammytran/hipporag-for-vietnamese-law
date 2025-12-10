#!/bin/bash

# Dừng script này nếu có lỗi
set -e

echo "--- Bắt đầu quá trình cài đặt môi trường cho dự án HippoRAG VN Law ---"

# Tạo môi trường conda
EVN_NAME="hipporag"
PYTHON_VER="3.10"
echo -e "\nTạo môi trường CONDA tên '$EVN_NAME'..."
if conda env list | grep -q "$EVN_NAME"; then
    echo "Môi trường '$EVN_NAME' đã tồn tài"
else 
    conda create -n $EVN_NAME python=$PYTHON_VER
    conda activate $EVN_NAME
    echo "Đã tạo môi trường '$ENV_NAME'."
fi

# Cài đặt các gói Python ---
# Chạy lệnh pip install bên trong môi trường conda vừa tạo
echo -e "\nCài đặt các gói Python vào môi trường '$ENV_NAME'..."
# eval "$(conda shell.bash hook)" # Đảm bảo conda hoạt động trong script
pip install -e .

# Tự lấy OPENAI_KEY 
echo -e "\nĐang tải biến môi trường từ file .env..."
if [ -f ".env" ]; then
    # grep -v "^#" .evn --> Bỏ qua các dòng comment #
    # grep -v "^$" .evn --> Bỏ qua các dòng trống 
    # xargs --> gom tất cả các dòng thành 1 dòng duy nhất
    # và export các biến tìm thấy.
    export $(grep -v "^#" .env | grep -v "^$" .env| xargs)
    echo "File .env đã được tải."
else
    echo "Cảnh báo: Không tìm thấy file .env. Các bước sau có thể thất bại nếu cần khóa API."
fi

# Kiểm tra xem OPENAI_API_KEY đã được set chưa
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Cảnh báo: Biến OPENAI_API_KEY chưa được thiết lập."
fi

echo -e "\n--- HOÀN TẤT CÀI ĐẶT! ---"