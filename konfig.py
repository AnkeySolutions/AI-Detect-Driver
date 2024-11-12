import os
import dlib
# Lấy đường dẫn đến thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đầy đủ đến tệp mô hình
model_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")

# Sử dụng đường dẫn này khi tạo predictor
predictor = dlib.shape_predictor(model_path)
