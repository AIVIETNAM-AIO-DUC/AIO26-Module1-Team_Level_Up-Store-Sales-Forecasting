# TÀI LIỆU CẤU TRÚC ĐẶC TRƯNG VÀ LUỒNG TIỀN XỬ LÝ DỮ LIỆU
**Dự án:** Store Sales - Time Series Forecasting (Kaggle)  
---

## 1. Danh Sách Đặc Trưng Giữ Lại Từ Dữ Liệu Thô (Raw Features)

Đây là các trường thông tin được giữ lại nguyên bản hoặc kế thừa trực tiếp từ các file dữ liệu gốc (`train.csv`, `test.csv`, `stores.csv`), chỉ thực hiện thay đổi kiểu dữ liệu hoặc đổi tên cột cho đồng bộ:

* **`date`**: Định dạng ngày tháng năm (Được đồng bộ về kiểu dữ liệu `datetime64` ở tất cả các bảng để làm khóa kết nối và phân chia tập dữ liệu).
* **`cluster`**: Nhóm các cửa hàng có đặc tính hoặc hành vi mua sắm tương đồng (Giữ nguyên dạng số nguyên từ bảng cửa hàng).
* **`sales`**: Doanh số thực tế của mặt hàng (Biến mục tiêu - *chỉ có trong tập Train*).
* **`transactions`**: Số lượng hóa đơn giao dịch phát sinh trong ngày tại cửa hàng (*chỉ có trong tập Train* - dùng cho phân tích tương quan và sẽ bị loại bỏ trước khi fit model).

---

## 2. Danh Sách Đặc Trưng Được Thêm Vào (Engineered Features)

Các đặc trưng mới được trích xuất và tính toán thông qua luồng biến đổi nâng cao, chia làm 3 nhóm chính:

### A. Nhóm Đặc Trưng Thời Gian & Chu Kỳ (Time-based Features)
* **`year`**: Định dạng năm (Trích xuất từ trường `date`) giúp mô hình học xu hướng dài hạn qua các năm.
* **`month`**: Định dạng tháng (1 đến 12) giúp nắm bắt tính mùa vụ theo các tháng trong năm.
* **`day`**: Định dạng ngày trong tháng (1 đến 31) giúp mô hình học biến động trong một tháng.
* **`day_of_week`**: Thứ trong tuần, được mã hóa từ số `0` (Thứ Hai) đến số `6` (Chủ Nhật).
* **`is_weekend`**: Biến nhị phân nhận giá trị `1` nếu ngày đó là Thứ Bảy hoặc Chủ Nhật, và `0` cho các ngày thường.
* **`is_payday`**: Biến nhị phân nhận giá trị `1` vào các ngày phát lương định kỳ của thị trường Ecuador (Ngày 15 và ngày cuối cùng của tháng). 
* **`is_month_end`**: Biến nhị phân nhận giá trị `1` nếu đó là ngày cuối cùng của tháng.
* **`is_easter_week`**: Biến đặc trưng cho tuần lễ Phục Sinh (Hiện tại đang được khởi tạo mặc định bằng giá trị `0`).

### B. Nhóm Đặc Trưng Môi Trường (Contextual Features)
* **`is_real_holiday`**: Biến nhị phân xác định ngày lễ Quốc gia thực tế. Đặc trưng này được liên kết từ bảng `holidays_events.csv`, chỉ lọc các ngày lễ thuộc cấp độ địa lý `National` và có trạng thái `transferred = False` (Ngày lễ không bị dời lịch).

### C. Nhóm Đặc Trưng Biến Trễ & Biến Trượt Lịch Sử (Lag & Rolling Features)
Để đảm bảo mô hình không bị rò rỉ dữ liệu tương lai (**Data Leakage**) khi tập Test có chu kỳ dự báo kéo dài 16 ngày liên tục, tất cả các biến lịch sử đều sử dụng gốc dịch chuyển an toàn nhỏ nhất là **`base_lag = 16`**:
* **`sales_lag_1`**: Doanh số thực tế của chính cặp cửa hàng - mặt hàng đó cách đây đúng **16 ngày**.
* **`sales_lag_7`**: Doanh số thực tế của chính cặp cửa hàng - mặt hàng đó cách đây đúng **17 ngày** (`16 + 1`).
* **`sales_lag_14`**: Doanh số thực tế của chính cặp cửa hàng - mặt hàng đó cách đây đúng **18 ngày** (`16 + 2`).
* **`rolling_mean_7`**: Giá trị trung bình trượt về doanh số trong chu kỳ **7 ngày**, tính từ mốc `sales_lag_1` lùi về quá khứ.
* **`rolling_std_7`**: Độ lệch chuẩn trượt về doanh số trong chu kỳ **7 ngày**, tính từ mốc `sales_lag_1` lùi về quá khứ.
* **`rolling_mean_14`**: Giá trị trung bình trượt về doanh số trong chu kỳ **14 ngày**, tính từ mốc `sales_lag_1` lùi về quá khứ.
* **`rolling_std_14`**: Độ lệch chuẩn trượt về doanh số trong chu kỳ **14 ngày**, tính từ mốc `sales_lag_1` lùi về quá khứ.

> 📌 *Lưu ý quan trọng:* Đã xóa bỏ hoàn toàn tất cả các thuộc tính biến trượt liên quan đến chu kỳ **30 ngày** nhằm tinh gọn dữ liệu theo yêu cầu.

---

## 3. Các Bảng Ánh Xạ Mã Hóa Dữ Liệu (Categorical Encoding Tables)

Để các thuật toán Machine Learning dạng bảng (LightGBM, XGBoost, CatBoost) có thể tính toán, toàn bộ các biến định danh dạng chữ (Object/String) đã được mã hóa tự động sang dạng số nguyên liên tục bắt đầu từ số `0` theo thứ tự bảng chữ cái alphabet (`Label Encoding`).

### A. Bảng Ánh Xạ Mã Hóa Nhóm Hàng (Family Mapping Table)

Cột `family` thô ban đầu gồm các chuỗi ký tự mô tả nhóm sản phẩm, được chuyển đổi thành mã số định danh cụ thể:

| Mã Số Sau Mã Hóa (`codes`) | Tên Nhóm Hàng Gốc Lịch Sử (`family`) |
| :---: | :--- |
| 0 | AUTOMOTIVE |
| 1 | BABY CARE |
| 2 | BEVERAGES |
| 3 | BOOKS |
| 4 | BREAD/BAKERY |
| 5 | CELEBRATIONS |
| 6 | CLEANING |
| 7 | DAIRY |
| 8 | DELI |
| 9 | EGGS |
| 10 | FROZEN FOODS |
| 11 | GROCERY I |
| 12 | GROCERY II |
| 13 | HARDWARE |
| 14 | HOME AND KITCHEN I |
| 15 | HOME AND KITCHEN II |
| 16 | HOME APPLIANCES |
| 17 | HOME CARE |
| 18 | LADIESWEAR |
| 19 | LAWN AND GARDEN |
| 20 | LINGERIE |
| 21 | LIQUOR,WINE,BEER |
| 22 | MAGAZINES |
| 23 | MEATS |
| 24 | PERSONAL CARE |
| 25 | PET SUPPLIES |
| 26 | PLAYERS AND ELECTRONICS |
| 27 | POULTRY |
| 28 | PREPARED FOODS |
| 29 | PRODUCE |
| 30 | SCHOOL AND OFFICE SUPPLIES |
| 31 | SEAFOOD |

### B. Bảng Ánh Xạ Mã Hóa Loại Cửa Hàng (Store Type Mapping Table)

Cột `type` thô trong bảng `stores.csv` đại diện cho phân cấp quy mô cửa hàng (A, B, C, D, E) được đổi tên thành cột `store_type` và mã hóa thành số:

| Mã Số Sau Mã Hóa (`codes`) | Loại Cửa Hàng Gốc (`store_type`) |
| :---: | :--- |
| 0 | Phân loại loại A |
| 1 | Phân loại loại B |
| 2 | Phân loại loại C |
| 3 | Phân loại loại D |
| 4 | Phân loại loại E |

### C. Bảng Ánh Xạ Mã Hóa Loại Ngày Lễ Quốc Gia (Holiday National Type)

Biến phân loại tính chất ngày lễ quốc gia (`holiday_national_type`) được trích xuất từ cột `type` của bảng ngày lễ để mô hình học thêm trọng số riêng biệt cho từng kiểu sự kiện:

| Tên Loại Ngày Lễ Gốc | Trạng Thái / Ý Nghĩa Sau Mã Hóa Số Nguyên |
| :--- | :--- |
| `None` | Ngày thường, hoàn toàn không có sự kiện hoặc ngày lễ quốc gia |
| `Holiday` | Ngày nghỉ lễ chính thức, được nghỉ làm việc |
| `Transfer` | Ngày nghỉ bù hoặc ngày lễ được dịch chuyển theo quy định |
| `Additional` | Ngày lễ bổ sung, lễ hội đặc thù |
| `Bridge` | Ngày nghỉ cầu nối (Bắc cầu giữa ngày lễ và ngày cuối tuần) |
| `Event` | Sự kiện đặc biệt tầm quốc gia (Ví dụ: Các trận thi đấu World Cup,...) |

---

## 4. Tổng Kết Cấu Trúc File Đầu Ra (Dataset Shape Specification)

Sau khi xử lý xong thông qua luồng code kết hợp, hai tệp dữ liệu được lưu xuống ổ đĩa với cấu trúc cột chuẩn chỉnh:

1.  **`train_ready_to_model.csv`**: Chứa đầy đủ **22 cột** (Bao gồm cả tính năng thời gian, biến trễ và biến mục tiêu `sales`, `transactions`). Các dòng trống lịch sử của những ngày đầu năm 2013 đã được xử lý triệt để bằng lệnh loại bỏ hàng khuyết (`dropna`).
2.  **`test_ready_to_model.csv`**: Chỉ giữ lại chính xác **20 cột** (Đã drop bỏ 2 cột `sales` và `transactions` để tránh lỗi cấu trúc, chỉ giữ lại các cột tính năng trễ lịch sử tương ứng để phục vụ pha dự báo `predict`).