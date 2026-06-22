# Scoring metric: RMSLE

**Phạm vi:** metric mà cuộc thi dùng để chấm điểm chúng ta (Root Mean Squared *Logarithmic*
Error), nó thưởng cho điều gì, và hai thói quen nó *bắt buộc* model phải tuân theo (train trong
log space, clip prediction ≥ 0).

---

## Đọc tên *ngược lại* — nó là một chuỗi các bước

**RMSLE** = *Root Mean Squared **Logarithmic** Error*, mỗi từ là một bước:

| Từ | Bước |
|------|------|
| **Error** | mỗi prediction sai bao nhiêu — nhưng đo *sau* khi đã lấy log ở bước dưới |
| **Logarithmic** | lấy **log** của cả hai số trước, *rồi* mới trừ |
| **Squared** | bình phương từng error (sai cao và sai thấp đều cộng dương) |
| **Mean** | lấy trung bình tất cả các bình phương error |
| **Root** | căn bậc hai ở bước cuối |

Viết dưới dạng công thức — trước hết là **dạng gốc**, đúng như cuộc thi định nghĩa:

```
                   1    n
RMSLE  =  sqrt(   ───   Σ   ( ln(1 + pᵢ) − ln(1 + aᵢ) )²   )
                   n   i=1

   n  = số lượng prediction          pᵢ = prediction thứ i
   ln = log tự nhiên (natural log)   aᵢ = giá trị actual thứ i
```

Đó chính là chuỗi các bước ở trên, chỉ viết lại bằng ký hiệu. Với mỗi cặp gồm một
prediction `pᵢ` và actual `aᵢ` tương ứng:

1. **Logarithmic** — lấy log của cả hai số: `ln(1 + pᵢ)` và `ln(1 + aᵢ)`.
2. **Error** — lấy hiệu của hai log đó; hiệu này chính là sai số, nhưng đo trong log space.
3. **Squared** — bình phương sai số đó.
4. **Mean** — làm vậy cho cả `n` cặp rồi lấy trung bình các bình phương.
5. **Root** — cuối cùng lấy căn bậc hai.

## `log1p` nghĩa là gì, từng bước một

Code — và công thức gọn mà ta dùng bên dưới — viết `log1p(x)` thay vì `ln(1 + x)`.
Chúng là **cùng một hàm**; `log1p` chỉ đọc rõ ra thành "**log** của **1** cộng x":

```
log1p(x) = ln(1 + x)
```

Đi từng bước một cho một con số, ví dụ `x = 15`:

```
1. bắt đầu với input         x      = 15
2. cộng một                  1 + x  = 16
3. lấy log tự nhiên          ln(16) ≈ 2.7726     →  log1p(15) ≈ 2.7726
```

Và với `x = 0` (ngày đóng cửa hoặc sản phẩm không bán được — rất hay gặp ở đây):

```
1. x      = 0
2. 1 + x  = 1
3. ln(1)  = 0                                    →  log1p(0) = 0
```

Cái `+1` chính là toàn bộ mánh khóe: `log(0)` thuần không xác định và sẽ làm crash,
nhưng `log1p(0)` cho ra một số `0` sạch sẽ. (*Vì sao* phải `+1`, và hàm ngược `expm1`
của nó, sẽ có section riêng bên dưới.)

Khi đã định nghĩa `log1p`, công thức gốc rút gọn lại thành dạng bạn sẽ gặp trong code —
cùng phép toán, viết ngắn hơn:

```
RMSLE = sqrt( mean( (log1p(prediction) − log1p(actual))² ) )
```

## Tại sao phải lấy log? Vì nó đo *tỷ lệ*, không phải khoảng cách thô

Lấy log **trước khi** đo error làm cho metric quan tâm đến **tỷ lệ** của cú miss
("lệch bao nhiêu lần?") thay vì khoảng cách thô giữa hai số. Cái identity làm điều này hoạt động
rất ngắn:

```
log(a) − log(b) = log(a / b)      ← trừ log chính là chia
```

Vậy nên một khi đã lấy log, "khoảng cách" không còn phụ thuộc vào *scale* nữa mà chỉ phụ thuộc
vào **tỷ lệ** `a / b`. Phần tiếp theo sẽ làm rõ điều này với hai family sản phẩm thật trong dataset.

## Unpack — vì sao điều này quan trọng

Các family sản phẩm sống trên những scale cực kỳ khác nhau: `GROCERY I` bán vài **nghìn**
unit/ngày, trong khi `BABY CARE` chỉ bán đếm trên đầu ngón tay. Với RMSE thông thường (đo khoảng
cách thô), cùng một cú miss *theo tỷ lệ* lại bị phạt rất khác nhau:

| Family | Predicted | Actual | Khoảng cách thô | Tỷ lệ |
|--------|-----------|--------|---------|-------|
| GROCERY I | 3,000 | 1,000 | **2,000** | 3× |
| BABY CARE |    15 |     5 |     **10** | 3× |

RMSE phạt hàng GROCERY **gấp 200×** hàng BABY CARE — dù cả hai forecast đều "cao gấp 3×". Một
model train theo RMSE sẽ học cách dồn hết công sức vào family lớn và *bỏ rơi* family nhỏ.

**Log sửa chuyện đó như thế nào.** Áp dụng identity ở phần trên cho cả hai cặp:

- `log(3000) − log(1000) = log(3000 / 1000) = log(3)`
- `log(15)   − log(5)    = log(15 / 5)     = log(3)`

Cả hai cặp đều thu về **cùng một giá trị `log(3)`**. RMSLE chấm chúng *bằng nhau*.

**Tóm tắt một dòng:** log chuyển trục số từ "gap scale" sang "ratio scale" → forecast của
family nhỏ và family lớn được chấm công bằng.

## `log1p` và `expm1` — miếng vá cho zero-sales

Công thức bên trên viết `log1p` chứ không phải `log` thuần vì một lý do: `log(0)` không xác
định, mà sales rất hay bằng **0** (ngày đóng cửa, sản phẩm bán chậm). Cách sửa là dịch input
thêm 1 trước khi lấy log:

```
log1p(x) = log(1 + x)        →   log1p(0) = log(1) = 0     ← không crash
expm1(x) = eˣ − 1            →   inverse của log1p, trả về sales thật
```

`log1p` đi *vào* log-space; `expm1` đi *ra*. Chúng đi thành cặp.

## Hai rule mà metric này ép lên *mọi* model

**Rule 1 — Train trong log space.**
- **Vì sao:** training tối thiểu hóa bất cứ loss nào bạn đưa cho nó. Nếu leaderboard đo *log*
  error, bạn muốn model cũng tối ưu *log* error — không thì bạn đang tối ưu một thứ và bị chấm
  bằng một thứ khác.
- **Làm sao:** fit trên `log1p(sales)`, predict trong log space, rồi `expm1` ngược về sales
  thật cho submission.

**Rule 2 — Clip prediction về ≥ 0.**
- **Vì sao:** một regression model có thể output ra số âm, mà `log1p(−2)` không xác định —
  metric sẽ crash. Sales âm cũng là điều bất khả thi về mặt vật lý.
- **Làm sao:** `clip_nonneg(pred)` trước khi score hoặc submit.

## Ví dụ chạy tay — clipping in action

Hai cặp (actual, prediction) chạy qua pipeline. Cặp thứ hai có prediction âm bị **clip về 0**
trước khi bất kỳ phép log nào được lấy:

| actual | pred | pred clipped | log1p(actual) | log1p(clipped) | log-diff² |
|--------|------|--------------|---------------|----------------|-----------|
|   5    |  15  |     15       |   1.7918      |    2.7726      |   0.962   |
|   0    |  −2  |      0       |   0.0000      |    0.0000      |   0.000   |

```
RMSLE = sqrt( (0.962 + 0.000) / 2 ) ≈ 0.694
```

Dòng 2 chính là lý do tồn tại của clipping: nếu không có nó, `log1p(−2)` đã làm sập cả phép
tính. Clip về 0 thì nó khớp với actual `0` và đóng góp một error sạch sẽ bằng zero.

**Ở đâu:** `src/validation.py :: rmsle()` và `clip_nonneg()`.

**Liên quan:** [validation-holdout.md](validation-holdout.md) · [baselines.md](baselines.md)
