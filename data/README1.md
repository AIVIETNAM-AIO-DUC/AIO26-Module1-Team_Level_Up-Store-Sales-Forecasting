# TÀI LIỆU CẤU TRÚC ĐẶC TRƯNG VÀ LUỒNG TIỀN XỬ LÝ DỮ LIỆU
**Dự án:** Store Sales - Time Series Forecasting (Kaggle)  
**Mục đích:** Tài liệu bàn giao và giải thích thuật toán Feature Engineering dành cho AI Engineer (AIE)

---

## 1. Bản Chất Luồng Xử Lý Dữ Liệu (Data Pipeline)

Hệ thống áp dụng chiến lược **Dự báo trực tiếp (Direct Forecasting)** cho toàn bộ chu kỳ 16 ngày của tập Test tương lai thay vì dự báo đệ quy từng ngày. Luồng xử lý bắt buộc phải thực hiện các bước sau:

1. **Gộp chuỗi lịch sử (`pd.concat`):** Ghép nối liền mạch tập Test (16/08/2017 - 31/08/2017) vào ngay sau tập Train. Việc này tạo một trục thời gian liên tục, cho phép các dòng của tập Test có thể nhìn ngược về doanh số (`sales`) thực tế của tập Train để làm căn cứ tính toán đặc trưng lịch sử.
2. **Đóng gói và Phân rã:** Sau khi tính toán xong xuôi toàn bộ các biến trễ và biến trượt trên bảng gộp chung, hệ thống phân rã ngược lại thành hai file độc lập. Tập Test được dọn dẹp sạch sẽ, loại bỏ hoàn toàn các cột không thể biết trước trong tương lai (`sales`, `transactions`).

---

## 2. Giải Mã Chi Tiết Các Đặc Trưng Thêm Vào (Engineered Features)

Các đặc trưng mới được trích xuất thông qua luồng biến đổi nâng cao. Để đảm bảo mô hình không bị rò rỉ dữ liệu tương lai (**Data Leakage**) khi tập Test có chu kỳ dự báo kéo dài 16 ngày liên tục hoàn toàn không có thông tin doanh số, tất cả các biến lịch sử đều sử dụng gốc dịch chuyển an toàn nhỏ nhất là **`base_lag = 16`**.

### A. Nhóm Đặc Trưng Biến Trễ (Lag Features) - Giải nghĩa logic thực tế
Trong cấu trúc dữ liệu hiện tại, tên gọi của các biến được đặt theo chu kỳ tương đối của bệ đỡ lịch sử. Tuy nhiên, một AI Engineer cần hiểu rõ mốc thời gian tịnh tiến thực tế so với ngày hiện tại (`date`) để tránh hiểu lầm:

* **`sales_lag_1` (Ứng với lệnh dịch chuỗi `shift(16)`):** * *Ý nghĩa logic:* Doanh số thực tế của mặt hàng cách ngày hiện tại đúng **16 ngày** (tức là ngày liền trước chu kỳ dự báo). 
  * *Mục tiêu:* Giúp mô hình nắm bắt phong độ và đà doanh số gần nhất của cửa hàng trước khi bước vào giai đoạn test. Đối với ngày cuối cùng của tập Test (`31/08/2017`), cột này sẽ bốc doanh số thực tế của ngày cuối cùng tập Train (`15/08/2017`) đắp sang.
* **`sales_lag_7` (Ứng với lệnh dịch chuỗi `shift(17)`):** * *Ý nghĩa logic:* Doanh số thực tế của mặt hàng cách ngày hiện tại đúng **17 ngày** (tức là cách ngày `sales_lag_1` 1 hôm về phía quá khứ).
  * *Đính chính kỹ thuật:* Biến này thực chất đang lấy dữ liệu của **ngày hôm trước nữa** của bệ đỡ lịch sử, chứ không phải là đúng thứ này tuần trước (Lag 7 ngày tự nhiên phải là `shift(23)`). It kết hợp với `sales_lag_1` tạo thành một chuỗi ngày liên tục sát biên giới tập Train.
* **`sales_lag_14` (Ứng với lệnh dịch chuỗi `shift(18)`):** * *Ý nghĩa logic:* Doanh số thực tế của mặt hàng cách ngày hiện tại đúng **18 ngày** (tức là cách ngày `sales_lag_1` 2 hôm về phía quá khứ).
  * *Đính chính kỹ thuật:* Biến này lấy dữ liệu của **ngày hôm trước nữa nữa**, tạo thành một bộ ba ngày liên tiếp (`shift(16)`, `shift(17)`, `shift(18)`) đại diện cho bức tranh doanh số 3 ngày cuối cùng của tập Train để mô hình học hành vi ngắn hạn.

### B. Nhóm Đặc Trưng Biến Trượt (Rolling Features) - Cơ chế tính toán từ bệ đỡ
Hệ thống đã loại bỏ hoàn toàn chu kỳ 30 ngày cồng kềnh để tránh làm nặng bộ nhớ và giảm nhiễu, chỉ giữ lại hai chu kỳ tinh gọn là **7 ngày** và **14 ngày**.

* **`rolling_mean_7`**: 
  * *Cơ chế toán học:* Giá trị trung bình trượt về doanh số của khung cửa sổ **7 ngày liên tiếp trong quá khứ**.
  * *Logic thực thi:* Khung cửa sổ này được tính toán **bắt đầu từ mốc `sales_lag_1` đổ về trước**. Nói cách khác, mô hình sẽ tính trung bình cộng doanh số của dải ngày từ `Lag 16` đến `Lag 22` so với ngày hiện tại. Nó đại diện cho mức tiêu thụ trung bình tuần gần nhất trước chu kỳ dự báo.
* **`rolling_std_7`**: 
  * *Cơ chế toán học:* Độ lệch chuẩn trượt về doanh số của khung cửa sổ **7 ngày liên tiếp trong quá khứ** (tính từ mốc `sales_lag_1` đổ về trước).
  * *Logic thực thi:* Đo lường mức độ biến động, trồi sụt của doanh số trong tuần sát biên giới tập Train. Chỉ số này giúp mô hình biết mặt hàng này có doanh số ổn định hay thường xuyên trồi sụt thất thường.
* **`rolling_mean_14`**: 
  * *Cơ chế toán học:* Giá trị trung bình trượt về doanh số của khung cửa sổ **14 ngày liên tiếp trong quá khứ** (tính từ mốc `sales_lag_1` đổ về trước, tức là dải ngày từ `Lag 16` đến `Lag 29`).
  * *Logic thực thi:* Đại diện cho sức mua trung bình bán nguyệt (2 tuần) gần nhất của mặt hàng, giúp mô hình nhận diện xu hướng tiêu dùng trung hạn.
* **`rolling_std_14`**: 
  * *Cơ chế toán học:* Độ lệch chuẩn trượt về doanh số của khung cửa sổ **14 ngày liên tiếp trong quá khứ** (tính từ mốc `sales_lag_1` đổ về trước).
  * *Logic thực thi:* Đo lường mức độ rủi ro và biến động doanh số trong vòng 2 tuần sát biên giới tập Train.

> 📌 *Lưu ý cấu trúc:* Giai đoạn đầu năm 2013 của tập Train sẽ bị trống dữ liệu (`NaN`) ở các cột Rolling và Lag này do chưa tích lũy đủ chuỗi lịch sử lịch sử. Luồng xử lý dữ liệu đã thực hiện lệnh `.dropna(subset=['sales_lag_1'])` để loại bỏ các dòng thiếu bệ đỡ này, giúp tập huấn luyện đầu ra cực kỳ sạch và không dính số 0 nhiễu.

### C. Nhóm Đặc Trưng Thời Gian & Môi Trường
* **`year` / `month` / `day`**: Trích xuất từ trường `date` để học xu hướng dài hạn và tính chu kỳ mùa vụ trong năm.
* **`day_of_week`**: Thứ trong tuần, được mã hóa từ số `0` (Thứ Hai) đến số `6` (Chủ Nhật).
* **`is_weekend`**: Biến nhị phân nhận giá trị `1` nếu là Thứ Bảy/Chủ Nhật, và `0` cho ngày thường.
* **`is_payday`**: Biến nhị phân nhận giá trị `1` vào các ngày phát lương định kỳ của Ecuador (Ngày 15 và ngày cuối cùng của tháng). Doanh số thường tăng đột biến vào những ngày này.
* **`is_month_end`**: Biến nhị phân nhận giá trị `1` nếu đó là ngày cuối cùng của tháng.
* **`is_easter_week`**: Biến đặc trưng cho tuần lễ Phục Sinh (Mặc định bằng `0`).
* **`is_real_holiday`**: Biến nhị phân (0/1) xác định ngày lễ cấp Quốc gia thực tế (Đã lọc bỏ các ngày lễ bị dời lịch - `transferred = True`).

---

## 3. Các Bảng Ánh Xạ Mã Hóa Dữ Liệu (Categorical Encoding Tables)

Toàn bộ các biến định danh dạng chữ (Object/String) đã được mã hóa tự động sang dạng số nguyên liên tục bắt đầu từ số `0` theo thứ tự bảng chữ cái alphabet (`Label Encoding`).

### A. Bảng Ánh Xạ Mã Hóa Nhóm Hàng (`family`)
| Mã Số (`codes`) | Tên Nhóm Hàng Gốc Lịch Sử (`family`) | | Mã Số (`codes`) | Tên Nhóm Hàng Gốc Lịch Sử (`family`) |
| :---: | :--- | :--- | :---: | :--- |
| **0** | AUTOMOTIVE | | **16** | HOME APPLIANCES |
| **1** | BABY CARE | | **17** | HOME CARE |
| **2** | BEVERAGES | | **18** | LADIESWEAR |
| **3** | BOOKS | | **19** | LAWN AND GARDEN |
| **4** | BREAD/BAKERY | | **20** | LINGERIE |
| **5** | CELEBRATIONS | | **21** | LIQUOR,WINE,BEER |
| **6** | CLEANING | | **22** | MAGAZINES |
| **7** | DAIRY | | **23** | MEATS |
| **8** | DELI | | **24** | PERSONAL CARE |
| **9** | EGGS | | **25** | PET SUPPLIES |
| **10** | FROZEN FOODS | | **26** | PLAYERS AND ELECTRONICS |
| **11** | GROCERY I | | **27** | POULTRY |
| **12** | GROCERY II | | **28** | PREPARED FOODS |
| **13** | HARDWARE | | **29** | PRODUCE |
| **14** | HOME AND KITCHEN I | | **30** | SCHOOL AND OFFICE SUPPLIES |
| **15** | HOME AND KITCHEN II | | **31** | SEAFOOD |

### B. Bảng Ánh Xạ Mã Hóa Loại Cửa Hàng (`store_type`)
Cột `type` thô trong bảng `stores.csv` đại diện cho phân cấp quy mô cửa hàng (A, B, C, D, E) được đổi tên thành cột `store_type` và mã hóa thành số:
| Mã Số Sau Mã Hóa (`codes`) | Loại Cửa Hàng Gốc (`store_type`) |
| :---: | :--- |
| **0** | Phân loại cửa hàng nhóm A |
| **1** | Phân loại cửa hàng nhóm B |
| **2** | Phân loại cửa hàng nhóm C |
| **3** | Phân loại cửa hàng nhóm D |
| **4** | Phân loại cửa hàng nhóm E |

---

## 4. Tổng Kết Cấu Trúc File Đầu Ra (Dataset Shape Specification)

Bảng dữ liệu sau khi kết thúc luồng xử lý tinh gọn chứa chính xác **22 cột đặc trưng** với cấu trúc phân rã nghiêm ngặt phục vụ trực tiếp cho mô hình:

1. **`train_ready_to_model.csv`**: Chứa đầy đủ **22 cột** (Bao gồm cả các tính năng thời gian, biến trễ lịch sử, biến trượt và biến mục tiêu `sales`, `transactions`).
2. **`test_ready_to_model.csv`**: Chỉ giữ lại chính xác **20 cột**. Toàn bộ hai cột tương lai không thể biết trước là `sales` và `transactions` đã được drop bỏ hoàn toàn để đảm bảo ma trận tính năng đầu vào (`X_test`) trùng khớp 100% về mặt số lượng và thứ tự cột với `X_train` lúc dự báo.

---

## 5. Chiến Lược Phân Chia Tập Kiểm Thử Nội Bộ (Validation Strategy)

Tuyệt đối không sử dụng phương pháp chia ngẫu nhiên (Random K-Fold) vì sẽ phá vỡ tính liên tục của chuỗi thời gian. Kỹ thuật áp dụng là **Time-based Hold-out (Cắt phân đoạn theo thời gian cuối chuỗi)**:

* **Tập Huấn Luyện (Train Split):** Toàn bộ dữ liệu sạch từ năm 2013 đến trước ngày **01/08/2017**.
* **Tập Kiểm Thử Nội Bộ (Validation Split):** Lấy chính xác **16 ngày cuối cùng** của tập huấn luyện gốc (từ ngày **01/08/2017** đến ngày **15/08/2017**). Quãng thời gian này có độ dài tương ứng 100% với tập Test thực tế, giúp ước lượng chính xác sai số RMSLE tiệm cận thực tế nhất trước khi submit bài lên hệ thống.
