# TÀI LIỆU CẤU TRÚC ĐẶC TRƯNG VÀ LUỒNG TIỀN XỬ LÝ DỮ LIỆU
**Dự án:** Store Sales - Time Series Forecasting (Kaggle)  
---

## 1. Bản Chất Luồng Xử Lý Dữ Liệu (Data Pipeline)

Hệ thống áp dụng chiến lược **Dự báo trực tiếp (Direct Forecasting)** cho toàn bộ chu kỳ 16 ngày của tập Test tương lai thay vì dự báo đệ quy từng ngày. Luồng xử lý bắt buộc phải thực hiện các bước sau:

1. **Gộp chuỗi lịch sử (`pd.concat`):** Ghép nối liền mạch tập Test (16/08/2017 - 31/08/2017) vào ngay sau tập Train. Việc này tạo một trục thời gian liên tục, cho phép các dòng của tập Test có thể nhìn ngược về doanh số (`sales`) thực tế của tập Train để làm căn cứ tính toán đặc trưng lịch sử.
2. **Đóng gói và Phân rã:** Sau khi tính toán xong xuôi toàn bộ các biến trễ và biến trượt trên bảng gộp chung, hệ thống phân rã ngược lại thành hai file độc lập. Tập Test được dọn dẹp sạch sẽ, loại bỏ hoàn toàn các cột không thể biết trước trong tương lai (`sales`, `transactions`).

---

## 2. Giải Mã Chi Tiết Các Đặc Trưng Thêm Vào (Engineered Features)

Các đặc trưng mới được trích xuất thông qua luồng biến đổi nâng cao. Để đảm bảo mô hình không bị rò rỉ dữ liệu tương lai (**Data Leakage**) khi tập Test có chu kỳ dự báo kéo dài 16 ngày liên tục hoàn toàn không có thông tin doanh số, tất cả các biến lịch sử đều sử dụng gốc dịch chuyển an toàn nhỏ nhất là **`base_lag = 16`**.

### A. Nhóm Đặc Trưng Biến Trễ (Lag Features) - Chuỗi Ngày Sát Biên Giới
Cấu trúc tên biến đã được chuẩn hóa lại từ (`lag_7`, `lag_14`) sang (`lag_2`, `lag_3`) để phản ánh chính xác bản chất tịnh tiến liên tiếp của các dòng dữ liệu ngay sát mốc biên giới của tập Train:

* **`sales_lag_1` (Ứng với lệnh dịch chuỗi `shift(16)`):** * *Ý nghĩa logic:* Doanh số thực tế của mặt hàng cách ngày hiện tại đúng **16 ngày** (tức là ngày liền trước chu kỳ dự báo).
  * *Mục tiêu:* Giúp mô hình nắm bắt phong độ và đà doanh số gần nhất của cửa hàng trước khi bước vào giai đoạn test. Đối với ngày cuối cùng của tập Test (`31/08/2017`), cột này sẽ bốc doanh số thực tế của ngày cuối cùng tập Train (`15/08/2017`) đắp sang.
* **`sales_lag_2` (Ứng với lệnh dịch chuỗi `shift(17)`):** * *Ý nghĩa logic:* Doanh số thực tế của mặt hàng cách ngày hiện tại đúng **17 ngày** (tức là cách ngày `sales_lag_1` đúng 1 ngày về phía quá khứ - tương ứng ngày **14/08/2017** ở cuối tập Train).
* **`sales_lag_3` (Ứng với lệnh dịch chuỗi `shift(18)`):** * *Ý nghĩa logic:* Doanh số thực tế của mặt hàng cách ngày hiện tại đúng **18 ngày** (tức là cách ngày `sales_lag_1` đúng 2 ngày về phía quá khứ - tương ứng ngày **13/08/2017** ở cuối tập Train).

> 🎯 **Tóm lại:** Bộ ba biến `sales_lag_1`, `sales_lag_2`, `sales_lag_3` tạo thành một kính hiển vi soi kỹ hành vi mua sắm của **3 ngày liên tiếp cuối cùng trong quá khứ** trước khi đóng sổ tập Train để làm điểm tựa dự đoán ngắn hạn.

### B. Nhóm Đặc Trưng Biến Trượt (Rolling Features) - Tầm Nhìn Trung Hạn Từ Bệ Phóng
Các biến trượt (Mean và Std) đóng vai trò làm ống nhòm tầm xa. Khung cửa sổ trượt được thiết lập cố định ở chu kỳ **7 ngày (1 tuần)** và **14 ngày (2 tuần)**, nhưng điểm bắt đầu tính toán sẽ **neo chặt vào bệ phóng `shift(16)`** (tức là từ vị trí của ngày `sales_lag_1` đổ ngược về quá khứ).

* **`rolling_mean_7`**: 
  * *Cơ chế toán học:* Giá trị trung bình trượt về doanh số của khung cửa sổ **7 ngày liên tiếp trong quá khứ** (tính từ `Lag 16` đến `Lag 22`).
  * *Ý nghĩa:* Cho mô hình biết sức mua trung bình ổn định trong vòng 1 tuần cuối cùng của quá khứ, giúp triệt tiêu các yếu tố nhiễu ngẫu nhiên.
* **`rolling_std_7`**: 
  * *Cơ chế toán học:* Độ lệch chuẩn trượt về doanh số của khung cửa sổ 7 ngày liên tiếp tính từ bệ phóng `shift(16)`.
  * *Ý nghĩa:* Đo lường mức độ biến động, trồi sụt của doanh số trong tuần sát biên giới quá khứ. Giúp mô hình biết mặt hàng bán đều đặn hay thất thường.
* **`rolling_mean_14`**: 
  * *Cơ chế toán học:* Giá trị trung bình trượt về doanh số của khung cửa sổ **14 ngày liên tiếp trong quá khứ** (tính từ `Lag 16` đến `Lag 29`).
  * *Ý nghĩa:* Đại diện cho sức mua trung bình bán nguyệt (2 tuần) gần nhất của mặt hàng, giúp nắm bắt xu hướng tiêu dùng trung hạn ổn định.
* **`rolling_std_14`**: 
  * *Cơ chế toán học:* Độ lệch chuẩn trượt về doanh số của khung cửa sổ 14 ngày liên tiếp tính từ bệ phóng `shift(16)`.
  * *Ý nghĩa:* Đo lường mức độ rủi ro và biên độ dao động doanh số trong vòng 2 tuần sát biên giới tập Train.

> 📌 *Lưu ý cấu trúc:* Giai đoạn đầu năm 2013 của tập Train sẽ bị trống dữ liệu (`NaN`) ở các cột này do chưa tích lũy đủ chuỗi lịch sử. Luồng xử lý đã thực hiện lệnh `.dropna(subset=['sales_lag_1'])` để lọc sạch các dòng khuyết thiếu này ra khỏi tập huấn luyện đầu ra.

### C. Nhóm Đặc Trưng Bối Cảnh, Thời Gian & Kinh Tế Vĩ Mô
* **`dcoilwtico`**: Giá dầu thô thế giới hàng ngày (Đã xử lý điền khuyết bằng `ffill` và `bfill`). Do nền kinh tế Ecuador phụ thuộc sống còn vào xuất khẩu dầu mỏ, biến vĩ mô này đóng vai trò cung cấp bối cảnh suy thoái hoặc hưng thịnh của thị trường qua các năm (đặc biệt là giai đoạn biến động mạnh 2015 - 2016).
* **`city` / `state`**: Thông tin địa lý cấp Thành phố và Tỉnh/Bang nơi đặt siêu thị. Đặc trưng này vô cùng quan trọng giúp mô hình học được sức mua theo vùng miền và đối chiếu để áp dụng các ngày lễ riêng biệt của từng địa phương (Local Holidays).
* **`holiday_national_type`**: Phân loại mã loại ngày lễ diễn ra ở cấp độ Quốc gia (ví dụ: *Holiday, Transfer, Event...*).
* **`year` / `month` / `day`**: Trích xuất từ trường `date` để học xu hướng dài hạn và tính chu kỳ mùa vụ trong năm.
* **`day_of_week`**: Thứ trong tuần, được mã hóa từ số `0` (Thứ Hai) đến số `6` (Chủ Nhật).
* **`is_weekend`**: Biến nhị phân (0/1): Xác định ngày cuối tuần (Thứ Bảy, Chủ Nhật).
* **`is_payday`**: Biến nhị phân (0/1): Ngày phát lương định kỳ tại Ecuador (Ngày 15 và ngày cuối cùng của tháng). Doanh số thường tăng đột biến vào những ngày này.
* **`is_month_end`**: Biến nhị phân (0/1): Xác định ngày cuối cùng của tháng.
* **`is_easter_week`**: Biến nhị phân (0/1): Xác định tuần lễ Phục Sinh (Holy Week). Đây là mùa mua sắm cao điểm dị biệt của các nhóm hàng thực phẩm, rau củ và hải sản tại Ecuador do yếu tố văn hóa tôn giáo Công giáo.

---

## 3. Bản Điều Phối Tất Cả Các Đặc Trưng Giữ Lại (26 Features)

Bảng dữ liệu sau khi kết thúc luồng xử lý tinh gọn chứa chính xác **26 cột đặc trưng** với cấu trúc phân rã nghiêm ngặt phục vụ trực tiếp cho mô hình:

| STT | Tên Đặc Trưng (Features) | Nhóm Phân Loại | Mô Tả Kỹ Thuật & Ý Nghĩa Đối Với Mô Hình |
| :---: | :--- | :--- | :--- |
| 1 | **`date`** | Thời gian | Khóa định danh thời gian thực (Sẽ drop ngay trước khi huấn luyện). |
| 2 | **`family`** | Phân loại | Mã số định danh nhóm sản phẩm (Đã mã hóa `Label Encoding` từ `0` đến `31`). |
| 3 | **`sales`** | Mục tiêu | Doanh số thực tế cần dự đoán (*Chỉ tồn tại ở tập Train*). |
| 4 | **`transactions`** | Bổ trợ | Số lượng hóa đơn trong ngày (*Chỉ có ở Train*, drop khi huấn luyện để tránh rò rỉ dữ liệu). |
| 5 | **`is_real_holiday`** | Môi trường | Biến nhị phân (0/1): Xác định ngày lễ quốc gia thực tế (Đã lọc bỏ ngày lễ bị dời lịch). |
| 6 | **`store_type`** | Thuộc tính | Mã số phân cấp quy mô/loại hình của cửa hàng (Đã mã hóa sang số từ `0` đến `4`). |
| 7 | **`cluster`** | Thuộc tính | Mã số cụm các cửa hàng có hành vi mua sắm tương đồng. |
| 8 | **`is_weekend`** | Chu kỳ | Biến nhị phân (0/1): Xác định ngày cuối tuần. |
| 9 | **`is_payday`** | Chu kỳ | Biến nhị phân (0/1): Ngày phát lương định kỳ tại Ecuador (Ngày 15 và cuối tháng). |
| 10 | **`is_month_end`** | Chu kỳ | Biến nhị phân (0/1): Xác định ngày cuối cùng của tháng. |
| 11 | **`is_easter_week`** | Chu kỳ | Biến nhị phân (0/1): Đặc trưng tuần lễ Phục Sinh biến động thực phẩm. |
| 12 | **`day_of_week`** | Chu kỳ | Thứ trong tuần, được đánh số từ `0` (Thứ Hai) đến `6` (Chủ Nhật). |
| 13 | **`year`** | Thời gian | Giá trị năm, giúp mô hình học xu hướng tăng trưởng dài hạn. |
| 14 | **`month`** | Thời gian | Giá trị tháng (1-12), giúp nắm bắt tính chu kỳ mùa vụ trong năm. |
| 15 | **`day`** | Thời gian | Giá trị ngày trong tháng (1-31), giúp học biến động tiêu dùng đầu/cuối tháng. |
| 16 | **`sales_lag_1`** | Biến trễ | Doanh số lịch sử cách ngày hiện tại 16 ngày (`shift(16)`). |
| 17 | **`sales_lag_2`** | Biến trễ | Doanh số lịch sử cách ngày hiện tại 17 ngày (`shift(17)`). |
| 18 | **`sales_lag_3`** | Biến trễ | Doanh số lịch sử cách ngày hiện tại 18 ngày (`shift(18)`). |
| 19 | **`rolling_mean_7`** | Biến trượt | Trung bình trượt doanh số chu kỳ 7 ngày tính từ mốc bệ phóng `shift(16)`. |
| 20 | **`rolling_std_7`** | Biến trượt | Độ lệch chuẩn trượt doanh số chu kỳ 7 ngày tính từ mốc bệ phóng `shift(16)`. |
| 21 | **`rolling_mean_14`** | Biến trượt | Trung bình trượt doanh số chu kỳ 14 ngày tính từ mốc bệ phóng `shift(16)`. |
| 22 | **`rolling_std_14`** | Biến trượt | Độ lệch chuẩn trượt doanh số chu kỳ 14 ngày tính từ mốc bệ phóng `shift(16)`. |
| 23 | **`dcoilwtico`** | Kinh tế vĩ mô | Biến chỉ số giá dầu thô đại diện cho sức khỏe nền kinh tế Ecuador. |
| 24 | **`city`** | Địa lý | Thành phố nơi đặt siêu thị (Đã mã hóa `Label Encoding`). |
| 25 | **`state`** | Địa lý | Bang/Tỉnh nơi đặt siêu thị (Đã mã hóa `Label Encoding`). |
| 26 | **`holiday_national_type`**| Môi trường | Loại ngày lễ cấp quốc gia diễn ra (Đã mã hóa `Label Encoding`). |

---

## 4. Các Bảng Ánh Xạ Mã Hóa Dữ Liệu (Categorical Encoding Tables)

Toàn bộ các biến định danh dạng chữ (Object/String) đã được mã hóa tự động sang dạng số nguyên liên tục bắt đầu từ số `0` theo thứ tự bảng chữ cái alphabet (`Label Encoding`).

### A. Bảng Ánh Xạ Mã Hóa Loại Cửa Hàng (`store_type`)
Cột `type` thô trong bảng `stores.csv` đại diện cho phân cấp quy mô cửa hàng (A, B, C, D, E) được đổi tên thành cột `store_type` và mã hóa thành số:
| Mã Số Sau Mã Hóa (`codes`) | Loại Cửa Hàng Gốc (`store_type`) |
| :---: | :--- |
| **0** | Phân loại cửa hàng nhóm A |
| **1** | Phân loại cửa hàng nhóm B |
| **2** | Phân loại cửa hàng nhóm C |
| **3** | Phân loại cửa hàng nhóm D |
| **4** | Phân loại cửa hàng nhóm E |

*(Lưu ý: Các trường `family`, `city`, `state`, và `holiday_national_type` cũng được tự động áp dụng cơ chế Label Encoding tương tự theo bảng chữ cái từ số 0 của thư viện Pandas).*

---

## 5. Tổng Kết Cấu Trúc File Đầu Ra (Dataset Shape Specification)

Bảng dữ liệu sau khi kết thúc luồng xử lý tinh gọn chứa cấu trúc phân rã nghiêm ngặt phục vụ trực tiếp cho mô hình:

1. **`train_ready_to_model.csv`**: Chứa đầy đủ **26 cột** (Bao gồm cả các tính năng bối cảnh, biến trễ lịch sử `lag_1, lag_2, lag_3`, biến trượt và hai biến mục tiêu `sales`, `transactions`).
2. **`test_ready_to_model.csv`**: Chỉ giữ lại chính xác **24 cột**. Toàn bộ hai cột tương lai không thể biết trước là `sales` và `transactions` đã được drop bỏ hoàn toàn để đảm bảo ma trận tính năng đầu vào (`X_test`) trùng khớp 100% về mặt số lượng và thứ tự cột với `X_train` lúc đưa vào pha dự báo.

---

## 6. Chiến Lược Phân Chia Tập Kiểm Thử Nội Bộ (Validation Strategy)

Tuyệt đối không sử dụng phương pháp chia ngẫu nhiên (Random K-Fold) vì sẽ phá vỡ tính liên tục của chuỗi thời gian. Kỹ thuật áp dụng là **Time-based Hold-out (Cắt phân đoạn theo thời gian cuối chuỗi)**:

* **Tập Huấn Luyện (Train Split):** Toàn bộ dữ liệu sạch từ năm 2013 đến trước ngày **01/08/2017**.
* **Tập Kiểm Thử Nội Bộ (Validation Split):** Lấy chính xác **16 ngày cuối cùng** của tập huấn luyện gốc (từ ngày **01/08/2017** đến ngày **15/08/2017**). Quãng thời gian này có độ dài tương ứng 100% với tập Test thực tế, giúp ước lượng chính xác sai số RMSLE tiệm cận thực tế nhất trước khi submit bài lên hệ thống.