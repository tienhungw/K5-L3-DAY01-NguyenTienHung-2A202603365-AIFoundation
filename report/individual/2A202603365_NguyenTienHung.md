# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **2A202603365**
- **Nguyễn Tiến Hưng**
- **2934**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.758      0.862        0.947   
2. BoW + MLP (deep learning)         0.978   0.909      0.942        0.974   
3a. Encoder (MiniLM) + kNN           0.876   0.929      0.902        0.927   
3b. Encoder (MiniLM) + LogReg        0.957   0.899      0.927        0.947   
3c. Encoder (MiniLM) + MLP head      0.967   0.899      0.932        0.947   
4a. LLM nhỏ zero-shot                0.783   0.900      0.837        0.837   
4b. LLM lớn zero-shot                0.864   0.950      0.905        0.905   
4c. LLM nhỏ few-shot                 0.857   0.900      0.878        0.878   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.388      14791.097       0.053   
2. BoW + MLP (deep learning)        14.816      28548.233       0.318   
3a. Encoder (MiniLM) + kNN         103.082         16.036      91.591   
3b. Encoder (MiniLM) + LogReg      104.866         35.938      34.395   
3c. Encoder (MiniLM) + MLP head    104.289         48.944      32.338   
4a. LLM nhỏ zero-shot                0.000          0.934     546.627   
4b. LLM lớn zero-shot                0.000          1.163     351.169   
4c. LLM nhỏ few-shot                 0.000          0.618    2299.781   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.142           0.0  
2. BoW + MLP (deep learning)          2.810           0.0  
3a. Encoder (MiniLM) + kNN          142.289           0.0  
3b. Encoder (MiniLM) + LogReg        50.703           0.0  
3c. Encoder (MiniLM) + MLP head      58.982           0.0  
4a. LLM nhỏ zero-shot              2613.518           0.0  
4b. LLM lớn zero-shot              2510.604           0.0  
4c. LLM nhỏ few-shot               2782.790           0.0 *

## 3. Dự đoán và thực tế

*
predict("tradeoff", "BoW + LogReg và BoW + MLP dự kiến nằm trên đường Pareto vì có tốc độ xử lý rất nhanh và F1 cao. Trong đó, BoW + MLP có thể là lựa chọn cân bằng tốt giữa độ chính xác và tốc độ. Các mô hình MiniLM có F1 khá tốt nhưng độ trễ cao hơn, còn LLM có độ trễ lớn nhất nên khó nằm trên đường Pareto. Theo số liệu đã đo, cả 3 mô hình BoW đều nằm bên trái vạch SLA 50 ms, trong khi các mô hình MiniLM kNN, LogReg, MLP và cả 3 mô hình LLM đều nằm bên phải vạch này. Dự đoán rằng BoW + MLP sẽ là lựa chọn nổi bật vì vừa đạt F1_SUB cao nhất (0.974), vừa có p50 chỉ 0.271 ms/tin.")
predict("ml_f1", "F1 khoảng 0.85, precision cao hơn recall vì mô hình có thể dự đoán spam khá chính xác nhưng vẫn bỏ sót một số tin nhắn spam.") 
predict(
    "ml_lr",
    "Với lr=0.001, loss sẽ giảm rất chậm và đường loss thoải, gần như phẳng. Với lr=50, loss có thể dao động mạnh hoặc tăng giảm thất thường do learning rate quá lớn, khiến mô hình cập nhật trọng số quá mức và khó hội tụ."
)
.*

## 4. Phân tích 50 case khó

*
Model BoW + MLP tụt nhiều nhất, từ accuracy 97,4% trên tập test thông thường xuống còn 56% trên 50 case khó, giảm 41,4 điểm phần trăm. Model hoạt động kém nhất ở nhóm spam giọng tự nhiên (chỉ đúng 10%) và nhóm spam né từ khóa (0%). Ví dụ, với tin nhắn giả danh người quen, sử dụng cách nói chuyện thân mật và không có từ khóa spam quen thuộc, model dễ phân loại nhầm thành ham vì chỉ dựa vào các từ xuất hiện trong tin nhắn mà không hiểu được ngữ cảnh và ý định lừa đảo. Một trường hợp khác là tin nhắn chứa từ C0ngr4tulat10ns thay cho Congratulations, khiến model không nhận diện được từ khóa quen thuộc do cách viết đã bị biến đổi. Nguyên nhân là mô hình Bag of Words chủ yếu học dựa trên tần suất và sự xuất hiện của từ ngữ, chưa có khả năng hiểu sâu ngữ nghĩa. Qua đó, có thể thấy accuracy cao trên tập test thông thường chưa chứng minh model thực sự hiểu spam. Trong thử nghiệm này, LLM nhỏ kết hợp prompt phòng thủ đạt 92% accuracy trên 50 case khó, cho thấy khả năng nhận diện ngữ cảnh và các hình thức lừa đảo tốt hơn.
*

## 5. Thử thách cá nhân

*
Em đã tiến hành thay đổi ngưỡng quyết định (Decision Threshold) của mô hình Logistic Regression từ 0.5 thành 0.2 và 0.8 để đánh giá ảnh hưởng đến Precision, Recall và F1-score.Kết quả thực nghiệm:

Threshold = 0.2: Precision = 0.9062, Recall = 0.8788, F1-score = 0.8923.
Threshold = 0.5: Precision = 1.0000, Recall = 0.7576, F1-score = 0.8621.
Threshold = 0.8: Precision = 1.0000, Recall = 0.6061, F1-score = 0.7547. Nhận xét: Khi giảm ngưỡng xuống 0.2, mô hình phát hiện được nhiều tin nhắn spam hơn, Recall tăng lên 87.88%, nhưng Precision giảm xuống 90.62%, dẫn đến một số tin nhắn bình thường bị phân loại nhầm là spam. Khi tăng ngưỡng lên 0.8, Precision đạt 100% nhưng Recall giảm còn 60.61%, nghĩa là mô hình bỏ sót nhiều tin nhắn spam hơn.
.*

## 6. Phương án cho sản phẩm thật

*
Dựa trên bảng benchmark, BoW MLP là phương án phù hợp nhất vì đạt F1 = 0.974, tốc độ khoảng 31.869 tin/giây và p95 = 0.401 ms, đáp ứng yêu cầu độ trễ dưới 50 ms và tốc độ xử lý 116 tin/giây. Các mô hình LLM có độ trễ khoảng 2.4 giây nên không phù hợp. Nếu xử lý 300 triệu tin/tháng, chi phí LLM được tính bằng 300.000 × C USD, trong đó C là chi phí cho mỗi 1.000 tin nhắn.
*

## 7. AI log và điều đã học

*
Em hiểu rằng accuracy cao trên tập test thông thường chưa chắc chứng minh mô hình thực sự hiểu nội dung tin nhắn spam. Những mô hình dựa trên Bag of Words dễ bị đánh lừa khi từ khóa bị thay đổi hoặc tin nhắn được viết tự nhiên như người quen. Tôi cũng hiểu rằng LLM có khả năng phân tích ngữ cảnh tốt hơn trong thử nghiệm này, nhưng vẫn cần đánh giá trên nhiều loại dữ liệu khác nhau để kiểm chứng độ tin cậy.
*
