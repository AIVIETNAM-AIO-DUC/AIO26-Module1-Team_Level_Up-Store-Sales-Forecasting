# Store Sales Forecasting - Mô Tả Dataset Cuối Cùng

## Tổng Quan

Dataset cuối cùng được xây dựng nhằm phục vụ bài toán dự báo doanh thu theo tháng của từng nhóm sản phẩm (Product Family).

Dữ liệu gốc ở mức ngày đã được xử lý, làm sạch, kết hợp với các nguồn dữ liệu khác và tổng hợp lên mức tháng. Đồng thời, nhiều đặc trưng thời gian (time series features) được tạo thêm để giúp mô hình học được xu hướng và tính mùa vụ của doanh thu.

Dataset này phù hợp cho các mô hình dự báo như XGBoost, LightGBM, Random Forest, LSTM và các mô hình dự báo chuỗi thời gian khác.

---

## Mức Độ Chi Tiết Của Dữ Liệu

Mỗi dòng dữ liệu đại diện cho:

> Một nhóm sản phẩm (Family) trong một tháng cụ thể.

Ví dụ:

| date | family | sales |
|------|---------|---------|
| 2015-01-01 | GROCERY I | 1,250,000 |
| 2015-02-01 | GROCERY I | 1,320,000 |
| 2015-01-01 | BEVERAGES | 850,000 |

Mỗi nhóm sản phẩm được xem như một chuỗi thời gian độc lập.

---

## Biến Mục Tiêu (Target)

| Feature | Ý nghĩa |
|----------|----------|
| sales | Tổng doanh thu của nhóm sản phẩm trong tháng |

Đây là biến mà mô hình sẽ dự đoán.

---

## Các Đặc Trưng Được Sử Dụng

### 1. Đặc Trưng Thời Gian

| Feature | Ý nghĩa |
|----------|----------|
| year | Năm |
| month | Tháng |
| quarter | Quý |

Nhóm đặc trưng này giúp mô hình học được xu hướng tăng trưởng và tính mùa vụ của doanh thu.

---

### 2. Đặc Trưng Giao Dịch

| Feature | Ý nghĩa |
|----------|----------|
| transactions | Tổng số giao dịch phát sinh trong tháng |

Số lượng giao dịch phản ánh mức độ hoạt động mua sắm của khách hàng và có tương quan mạnh với doanh thu.

---

### 3. Đặc Trưng Ngày Lễ

Thông tin ngày lễ được lấy từ bảng `holidays_events.csv`.

| Feature | Ý nghĩa |
|----------|----------|
| is_real_holiday | Tổng số ngày nghỉ thực tế trong tháng |
| holiday_Holiday | Số ngày Holiday trong tháng |
| holiday_Event | Số ngày Event trong tháng |
| holiday_Additional | Số ngày Additional trong tháng |
| holiday_Bridge | Số ngày Bridge trong tháng |
| holiday_Transfer | Số ngày Transfer trong tháng |

Các đặc trưng này giúp mô hình nhận biết ảnh hưởng của các dịp đặc biệt tới doanh thu.

---

### 4. Đặc Trưng Lịch Sử Doanh Thu (Lag Features)

Lag Features thể hiện doanh thu trong các tháng trước đó.

| Feature | Ý nghĩa |
|----------|----------|
| sales_lag_1 | Doanh thu của tháng trước |
| sales_lag_2 | Doanh thu của 2 tháng trước |
| sales_lag_3 | Doanh thu của 3 tháng trước |
| sales_lag_6 | Doanh thu của 6 tháng trước |
| sales_lag_12 | Doanh thu cùng kỳ năm trước |

Đây là nhóm đặc trưng quan trọng nhất trong các bài toán dự báo chuỗi thời gian.

---

### 5. Đặc Trưng Thống Kê Trượt (Rolling Features)

Rolling Features mô tả xu hướng doanh thu gần đây.

| Feature | Ý nghĩa |
|----------|----------|
| rolling_mean_3 | Doanh thu trung bình 3 tháng gần nhất |
| rolling_mean_6 | Doanh thu trung bình 6 tháng gần nhất |
| rolling_mean_12 | Doanh thu trung bình 12 tháng gần nhất |
| rolling_std_3 | Độ biến động doanh thu trong 3 tháng gần nhất |
| rolling_std_6 | Độ biến động doanh thu trong 6 tháng gần nhất |

Các đặc trưng này giúp mô hình nắm bắt được xu hướng tăng trưởng và mức độ ổn định của doanh thu.

---

## Đặc Điểm Của Dataset

Dataset cuối cùng là một bộ dữ liệu chuỗi thời gian đa biến (Multivariate Time Series Dataset), bao gồm:

- Nhiều nhóm sản phẩm khác nhau.
- Dữ liệu được tổng hợp theo tháng.
- Thông tin doanh thu lịch sử.
- Thông tin giao dịch.
- Thông tin ngày lễ.
- Các đặc trưng thời gian.

Mỗi nhóm sản phẩm tạo thành một chuỗi thời gian riêng nhưng được huấn luyện bằng cùng một mô hình.

---

## Mục Tiêu Dự Báo

Mục tiêu của bài toán là:

> Dự báo doanh thu của từng nhóm sản phẩm trong tháng kế tiếp.

Đầu vào của mô hình gồm:

- Doanh thu lịch sử.
- Thông tin thời gian.
- Thông tin ngày lễ.
- Thông tin giao dịch.

Đầu ra:

- Doanh thu dự báo (`sales`) của tháng tiếp theo.

---

## Các Mô Hình Phù Hợp

### Machine Learning

- XGBoost
- LightGBM
- Random Forest
- CatBoost

### Deep Learning

- LSTM
- GRU
- Temporal Fusion Transformer (TFT)

Trong phạm vi đồ án, XGBoost hoặc LightGBM là lựa chọn phù hợp vì tận dụng tốt các đặc trưng Lag và Rolling đã được xây dựng.

---

## Danh Sách Feature Cuối Cùng

### Target

```text
sales
```

### Time Features

```text
year
month
quarter
```

### Transaction Features

```text
transactions
```

### Holiday Features

```text
is_real_holiday
holiday_Holiday
holiday_Event
holiday_Additional
holiday_Bridge
holiday_Transfer
```

### Lag Features

```text
sales_lag_1
sales_lag_2
sales_lag_3
sales_lag_6
sales_lag_12
```

### Rolling Features

```text
rolling_mean_3
rolling_mean_6
rolling_mean_12

rolling_std_3
rolling_std_6
```