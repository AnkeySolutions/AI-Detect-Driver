import cv2  # Thư viện OpenCV để xử lý hình ảnh và video
import dlib  # Thư viện dlib để phát hiện khuôn mặt và nhận diện các điểm đặc trưng
import pygame  # Thư viện pygame để phát âm thanh cảnh báo
import mediapipe as mp  # Thư viện MediaPipe để nhận diện cử chỉ tay và khuôn mặt
from scipy.spatial import distance as dist  # Thư viện scipy để tính khoảng cách giữa các điểm trên khuôn mặt
from imutils import face_utils  # Tiện ích của imutils để xử lý các điểm đặc trưng của khuôn mặt
import time  # Thư viện time để theo dõi thời gian
import numpy as np  # Thư viện numpy để xử lý dữ liệu dưới dạng mảng


def list_cameras():
    index = 0
    available_cameras = []

    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            available_cameras.append(index)
        cap.release()
        index += 1

    return available_cameras
# Hiển thị danh sách các camera tìm thấy
cameras_target = list_cameras()
print("list camera hoat dong:" , cameras_target)
camera_get = int(input("Chon cam -> "))

# Hàm tính toán tỉ lệ nhắm mắt (EAR) để phát hiện ngủ gật
def tinh_ti_le_nham_mat(mat):
    A = dist.euclidean(mat[1], mat[5])  # Tính khoảng cách giữa hai điểm đầu tiên của mắt
    B = dist.euclidean(mat[2], mat[4])  # Tính khoảng cách giữa hai điểm thứ hai của mắt
    C = dist.euclidean(mat[0], mat[3])  # Tính khoảng cách ngang giữa hai điểm cuối của mắt
    ti_le_nham = (A + B) / (2.0 * C)  # Công thức tính tỉ lệ nhắm mắt EAR
    return ti_le_nham  # Trả về giá trị EAR

# Đặt ngưỡng nhắm mắt và thời gian để phát hiện ngủ gật
NGUONG_NHAM = 0.3  # Ngưỡng EAR dưới giá trị này được coi là nhắm mắt
THOI_GIAN_CANH_BAO = 1.5  # Thời gian tối thiểu để phát hiện ngủ gật (giây)

# Khởi tạo dlib để phát hiện khuôn mặt và các điểm đặc trưng trên khuôn mặt
detector = dlib.get_frontal_face_detector()  # Tạo đối tượng phát hiện khuôn mặt
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Tạo đối tượng nhận diện các điểm trên khuôn mặt
(mat_trai, mat_phai) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # Xác định vị trí các điểm trên mắt trái và mắt phải

# Khởi tạo pygame để phát âm thanh cảnh báo
pygame.mixer.init()  # Khởi tạo hệ thống âm thanh của pygame
am_thanh_1 = pygame.mixer.Sound('bdnglx.mp3')  # Âm thanh cảnh báo khi phát hiện ngủ gật
am_thanh_2 = pygame.mixer.Sound('coihu.mp3')  # Âm thanh cảnh báo tiếp theo khi vẫn ngủ gật
am_thanh_lai_mot_tay = pygame.mixer.Sound('mttlx.mp3')  # Âm thanh cảnh báo khi lái xe một tay
am_thanh_mat_tap_trung = pygame.mixer.Sound('lmt.mp3')  # Âm thanh cảnh báo khi mất tập trung

# Khởi tạo MediaPipe để nhận diện cử chỉ tay và khuôn mặt
mp_hands = mp.solutions.hands  # Đối tượng nhận diện cử chỉ tay
mp_face_mesh = mp.solutions.face_mesh  # Đối tượng nhận diện khuôn mặt
hands = mp_hands.Hands()  # Khởi tạo đối tượng nhận diện tay
face_mesh = mp_face_mesh.FaceMesh()  # Khởi tạo đối tượng nhận diện khuôn mặt
mp_drawing = mp.solutions.drawing_utils  # Công cụ để vẽ các điểm và kết nối
mp_drawing_styles = mp.solutions.drawing_styles  # Công cụ để tùy chỉnh kiểu vẽ

# Mở camera để bắt đầu quá trình giám sát
cap = cv2.VideoCapture(camera_get)  # Mở camera mặc định trên máy tính

# Tăng độ phân giải của hình ảnh từ camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Đặt chiều rộng khung hình
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Đặt chiều cao khung hình

# Biến để theo dõi trạng thái của tài xế
dem_frame_nham = 0  # Đếm số khung hình mà mắt tài xế nhắm
bat_dau_nham = 0  # Thời điểm bắt đầu nhắm mắt
dang_canh_bao = False  # Trạng thái cảnh báo hiện tại
thoi_gian_mo_mat = 0  # Thời gian mở mắt của tài xế
nhay_doi = True  # Biến để điều khiển nhấp nháy màn hình cảnh báo
bat_dau_lai_mot_tay = None  # Thời gian bắt đầu lái xe một tay
bat_dau_nghieng_dau = None  # Thời gian bắt đầu quay đầu hoặc nhìn sang bên
gioi_han_thoi_gian_mot_tay = 5  # Giới hạn thời gian lái xe một tay (giây)
gioi_han_thoi_gian_nghieng_dau = 3  # Giới hạn thời gian quay đầu hoặc nhìn sang bên (giây)
dang_lai_mot_tay = False  # Trạng thái lái xe một tay hiện tại
dang_mat_tap_trung = False  # Trạng thái mất tập trung hiện tại

# Vòng lặp để xử lý hình ảnh từ camera liên tục
while True:
    ret, frame = cap.read()  # Đọc hình ảnh từ camera
    if not ret:  # Kiểm tra nếu không có hình ảnh nào được đọc
        break  # Thoát vòng lặp nếu không có hình ảnh

    frame = cv2.flip(frame, 1)  # Đảo ngược hình ảnh để giống như nhìn qua gương

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển đổi hình ảnh sang màu xám để xử lý
    rects = detector(gray, 0)  # Phát hiện khuôn mặt trong khung hình

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi hình ảnh sang định dạng RGB
    result_hands = hands.process(rgb_frame)  # Xử lý nhận diện tay từ hình ảnh RGB
    result_face = face_mesh.process(rgb_frame)  # Xử lý nhận diện khuôn mặt từ hình ảnh RGB

    # Biến để theo dõi số lượng tay được phát hiện trên vô lăng
    so_tay_phat_hien = 0

    # Xử lý để phát hiện tay trên vô lăng
    if result_hands.multi_hand_landmarks:  # Nếu phát hiện tay trong khung hình
        so_tay_phat_hien = len(result_hands.multi_hand_landmarks)  # Đếm số lượng tay được phát hiện
        for hand_landmarks in result_hands.multi_hand_landmarks:  # Duyệt qua từng tay được phát hiện
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Vẽ các điểm và kết nối trên tay

    # Kiểm tra nếu không phát hiện tay nào trên vô lăng hoặc chỉ có một tay trên vô lăng trong thời gian dài
    if so_tay_phat_hien == 0:  # Nếu không phát hiện tay nào trên vô lăng
        if bat_dau_lai_mot_tay is None:  # Nếu chưa bắt đầu đếm thời gian
            bat_dau_lai_mot_tay = time.time()  # Bắt đầu đếm thời gian
        elif time.time() - bat_dau_lai_mot_tay > gioi_han_thoi_gian_mot_tay:  # Nếu thời gian vượt quá giới hạn
            dang_lai_mot_tay = True  # Đặt trạng thái lái xe một tay
            if not pygame.mixer.get_busy():  # Nếu không có âm thanh nào đang phát
                pygame.mixer.Sound.play(am_thanh_lai_mot_tay)  # Phát âm thanh cảnh báo
            cv2.putText(frame, "2hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        3)  # Hiển thị thông báo trên màn hình
    elif so_tay_phat_hien == 1:  # Nếu chỉ có một tay trên vô lăng
        if bat_dau_lai_mot_tay is None:  # Nếu chưa bắt đầu đếm thời gian
            bat_dau_lai_mot_tay = time.time()  # Bắt đầu đếm thời gian
        elif time.time() - bat_dau_lai_mot_tay > gioi_han_thoi_gian_mot_tay:  # Nếu thời gian vượt quá giới hạn
            dang_lai_mot_tay = True  # Đặt trạng thái lái xe một tay
            if not pygame.mixer.get_busy():  # Nếu không có âm thanh nào đang phát
                pygame.mixer.Sound.play(am_thanh_lai_mot_tay)  # Phát âm thanh cảnh báo
            cv2.putText(frame, "2hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        3)  # Hiển thị thông báo trên màn hình
    else:  # Nếu phát hiện cả hai tay trên vô lăng
        bat_dau_lai_mot_tay = None  # Đặt lại thời gian bắt đầu lái một tay
        dang_lai_mot_tay = False  # Đặt lại trạng thái lái xe một tay

    # Xử lý phát hiện khuôn mặt và phân tích sự tập trung
    if result_face.multi_face_landmarks:  # Nếu phát hiện khuôn mặt trong khung hình
        for face_landmarks in result_face.multi_face_landmarks:  # Duyệt qua từng khuôn mặt được phát hiện
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )  # Vẽ các điểm và kết nối trên khuôn mặt

            # Lấy vị trí của mũi và mắt để xác định hướng nhìn của tài xế
            nose_tip = face_landmarks.landmark[1]  # Chỉ số 1 tương ứng với đầu mũi
            left_eye_outer = face_landmarks.landmark[33]  # Chỉ số 33 cho góc ngoài mắt trái
            right_eye_outer = face_landmarks.landmark[263]  # Chỉ số 263 cho góc ngoài mắt phải

            # Tính toán khoảng cách giữa mắt và mũi theo chiều ngang
            eye_distance = abs(left_eye_outer.x - right_eye_outer.x)
            nose_to_left_eye = abs(nose_tip.x - left_eye_outer.x)
            nose_to_right_eye = abs(nose_tip.x - right_eye_outer.x)

            # Kiểm tra nếu mũi lệch quá nhiều về một phía, tức là tài xế đang quay đầu hoặc nhìn sang bên
            if nose_to_left_eye / eye_distance < 0.4 or nose_to_right_eye / eye_distance < 0.4:
                if bat_dau_nghieng_dau is None:  # Nếu chưa bắt đầu đếm thời gian quay đầu hoặc nhìn sang bên
                    bat_dau_nghieng_dau = time.time()  # Bắt đầu đếm thời gian
                elif time.time() - bat_dau_nghieng_dau > gioi_han_thoi_gian_nghieng_dau:  # Nếu thời gian vượt quá giới hạn
                    dang_mat_tap_trung = True  # Đặt trạng thái mất tập trung
                    if not pygame.mixer.get_busy():  # Nếu không có âm thanh nào đang phát
                        pygame.mixer.Sound.play(am_thanh_mat_tap_trung)  # Phát âm thanh cảnh báo mất tập trung
                    cv2.putText(frame, "un_reg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                3)  # Hiển thị thông báo trên màn hình
            else:  # Nếu tài xế đang nhìn thẳng
                bat_dau_nghieng_dau = None  # Đặt lại thời gian bắt đầu quay đầu hoặc nhìn sang bên
                dang_mat_tap_trung = False  # Đặt lại trạng thái mất tập trung

    # Xử lý phát hiện trạng thái nhắm mắt để phát hiện ngủ gật
    for rect in rects:  # Duyệt qua từng khuôn mặt được phát hiện
        shape = predictor(gray, rect)  # Dự đoán các điểm đặc trưng trên khuôn mặt
        shape = face_utils.shape_to_np(shape)  # Chuyển đổi các điểm đặc trưng sang định dạng numpy

        mat_trai_pts = shape[mat_trai[0]:mat_trai[1]]  # Lấy các điểm đặc trưng của mắt trái
        mat_phai_pts = shape[mat_phai[0]:mat_phai[1]]  # Lấy các điểm đặc trưng của mắt phải

        ti_le_mat_trai = tinh_ti_le_nham_mat(mat_trai_pts)  # Tính toán tỉ lệ nhắm mắt cho mắt trái
        ti_le_mat_phai = tinh_ti_le_nham_mat(mat_phai_pts)  # Tính toán tỉ lệ nhắm mắt cho mắt phải

        ti_le_nham_tb = (ti_le_mat_trai + ti_le_mat_phai) / 2.0  # Tính toán tỉ lệ nhắm mắt trung bình của cả hai mắt

        x, y, w, h = face_utils.rect_to_bb(rect)  # Xác định vị trí của khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hình chữ nhật bao quanh khuôn mặt

        if ti_le_nham_tb < NGUONG_NHAM:  # Nếu tỉ lệ nhắm mắt nhỏ hơn ngưỡng cho phép (ngủ gật)
            if dem_frame_nham == 0:  # Nếu đây là lần đầu tiên phát hiện nhắm mắt
                bat_dau_nham = time.time()  # Bắt đầu đếm thời gian nhắm mắt
            dem_frame_nham += 1  # Tăng số khung hình đã phát hiện nhắm mắt

            # Kiểm tra nếu mắt đã nhắm quá thời gian cho phép
            if (time.time() - bat_dau_nham) >= THOI_GIAN_CANH_BAO:
                if not dang_canh_bao:  # Nếu chưa phát cảnh báo trước đó
                    pygame.mixer.Sound.play(am_thanh_1)  # Phát âm thanh cảnh báo ngủ gật
                    dang_canh_bao = True  # Đặt trạng thái đang cảnh báo
                    bat_dau_canh_bao = time.time()  # Bắt đầu đếm thời gian cảnh báo

        else:  # Nếu mắt mở lại
            dem_frame_nham = 0  # Đặt lại đếm số khung hình nhắm mắt
            dang_canh_bao = False  # Đặt lại trạng thái cảnh báo
            pygame.mixer.Sound.stop(am_thanh_1)  # Dừng âm thanh cảnh báo
            pygame.mixer.Sound.stop(am_thanh_2)  # Dừng âm thanh cảnh báo tiếp theo

    # Chuyển sang phát âm thanh cảnh báo tiếp theo nếu tài xế vẫn nhắm mắt
    if dang_canh_bao and (time.time() - bat_dau_canh_bao) >= THOI_GIAN_CANH_BAO:
        if not pygame.mixer.get_busy():  # Nếu không có âm thanh nào đang phát
            pygame.mixer.Sound.play(am_thanh_2)  # Phát âm thanh cảnh báo tiếp theo

    # Xử lý các hiệu ứng nhấp nháy màu sắc khác nhau trên màn hình
    if dang_canh_bao:  # Nếu đang cảnh báo ngủ gật
        if nhay_doi:  # Nếu cần nhấp nháy
            #overlay = frame.copy()  # Tạo một lớp phủ từ khung hình hiện tại
            #cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255),
            #              -1)  # Vẽ hình chữ nhật đỏ trên toàn bộ màn hình
            #alpha = 0.3  # Độ trong suốt của lớp phủ
            #cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Áp dụng lớp phủ lên khung hình
            nhay_doi = False  # Chuyển trạng thái nhấp nháy
        else:
            nhay_doi = True  # Đặt lại trạng thái nhấp nháy
        # Thêm thông báo chữ lên trên lớp phủ màu đỏ
        cv2.putText(frame, "waring", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "sleepy", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "target_1", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    elif dang_lai_mot_tay:  # Nếu đang cảnh báo lái xe một tay
        if nhay_doi:  # Nếu cần nhấp nháy
            #overlay = frame.copy()  # Tạo một lớp phủ từ khung hình hiện tại
            #cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 255),
            #              -1)  # Vẽ hình chữ nhật vàng trên toàn bộ màn hình
            #alpha = 0.3  # Độ trong suốt của lớp phủ
            #cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Áp dụng lớp phủ lên khung hình
            nhay_doi = False  # Chuyển trạng thái nhấp nháy
        else:
            nhay_doi = True  # Đặt lại trạng thái nhấp nháy
    elif dang_mat_tap_trung:  # Nếu đang cảnh báo mất tập trung
        if nhay_doi:  # Nếu cần nhấp nháy
            #overlay = frame.copy()  # Tạo một lớp phủ từ khung hình hiện tại
            #cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 140, 255),
            #              -1)  # Vẽ hình chữ nhật cam đậm trên toàn bộ màn hình
            #alpha = 0.3  # Độ trong suốt của lớp phủ
            #cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Áp dụng lớp phủ lên khung hình
            nhay_doi = False  # Chuyển trạng thái nhấp nháy
        else:
            nhay_doi = True  # Đặt lại trạng thái nhấp nháy
    else:  # Khi tài xế đang lái xe tốt (không có hành vi nguy hiểm)
        overlay = frame.copy()  # Tạo một lớp phủ từ khung hình hiện tại
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0),
                      -1)  # Vẽ hình chữ nhật xanh lá trên toàn bộ màn hình
        alpha = 0.15  # Độ trong suốt của lớp phủ
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Áp dụng lớp phủ lên khung hình

    cv2.imshow("moded_by_ankey - press q to exit", frame)  # Hiển thị khung hình lên cửa sổ

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Kiểm tra nếu phím 'q' được nhấn
        break  # Thoát khỏi vòng lặp nếu nhấn 'q'

cap.release()  # Giải phóng camera khi kết thúc
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV


