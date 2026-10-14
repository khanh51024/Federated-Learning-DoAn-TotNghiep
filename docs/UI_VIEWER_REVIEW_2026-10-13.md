# Review UI dưới góc nhìn người xem và hội đồng — 13/10/2026

URL đã xem trực tiếp: `http://127.0.0.1:8123/#tong-quan`.

## Nhận định chính

Giao diện có màu sắc nhẹ, bố cục sạch và trình tự sáu bước hợp lý. Người lần đầu xem có thể hiểu ý tưởng “cùng huấn luyện, ảnh ở lại từng nơi” ở phần mở đầu. Tuy nhiên, để dùng trong buổi bảo vệ, nội dung hiện còn dày ở trang Dữ liệu, các tab thống kê và Bằng chứng. Các đoạn giải thích thường đưa nhiều số, thuật ngữ và điều kiện vào cùng một khối; người nghe phải đọc trong lúc nghe thuyết trình.

Vấn đề chính là thứ tự và lượng thông tin mở mặc định. Nhiều chỗ lặp lại cùng một ý; một số thông tin cần cho câu hỏi sâu xuất hiện trước kết luận chính. Nên thiết kế mỗi bước có một câu kết luận, một hình hoặc phép so sánh chính, rồi cho mở chi tiết khi cần. Giữ phạm vi và hạn chế quan trọng ở ngay cạnh kết quả, với câu ngắn; phần chứng minh dài đặt trong mục mở rộng.

Đây là nhận định của reviewer khi mô phỏng lần xem đầu tiên, chưa phải kết quả khảo sát giảng viên thật. Các câu hỏi hội đồng dưới đây là câu hỏi dự kiến từ nội dung UI.

## Bằng chứng đã kiểm

- Mở cả sáu bước, đọc các nội dung đang hiển thị và xem ảnh chụp ở viewport 1280×720.
- Mở đủ sáu tab của Kết quả; xem riêng màn đầu của tab Tập giữ riêng.
- Mở ví dụ Tomato — Bacterial spot ở trang Dữ liệu.
- Chạy chẩn đoán ảnh mẫu #12: Centralized và FedAvg đúng, Local-only đúng 1/5; tóm tắt là 3/7 mô hình đúng. Đây là quan sát trên một ảnh, không dùng để kết luận chất lượng chung.
- Bật chế độ Trình chiếu ở trang Bằng chứng. Chế độ này tăng cỡ chữ và ẩn một số chi tiết kỹ thuật, nhưng các đoạn dài vẫn mở mặc định.
- Đối chiếu mã nguồn đang làm việc. SHA-256 nội dung JavaScript phục vụ từ cổng 8123 khớp bảy file cục bộ: Overview, Results, Evidence, NonIID, Diagnosis, FLNet và Store.
- Chỉ tạo báo cáo và ảnh chứng minh trong lần review này; chưa áp dụng các đề xuất vào UI.

Ảnh chụp: [Tổng quan](ui-review-2026-10-13/overview.png), [Tập giữ riêng](ui-review-2026-10-13/holdout.png), [Bằng chứng khi trình chiếu](ui-review-2026-10-13/evidence-present.png), [Chẩn đoán mẫu #12](ui-review-2026-10-13/diagnosis-sample12.png). Nội dung DOM đã đọc lưu ở [observations.json](ui-review-2026-10-13/observations.json).

## Cảm nhận theo từng bước

| Bước | Người xem hiểu được gì | Chỗ gây nặng hoặc khó hiểu | Đề xuất cụ thể |
| --- | --- | --- | --- |
| 1. Tổng quan | Hiểu ý tưởng học chung và không gửi ảnh gốc. Sơ đồ giúp giải thích FedAvg. | Hero chiếm phần lớn màn đầu; kết quả chính bắt đầu gần mép dưới. Sơ đồ, chú thích, ba bước và các thẻ lặp lại ý ảnh ở client/gửi trọng số. Lợi ích nghiên cứu và phần nhóm tự thực hiện chưa nổi bật ở màn đầu. | Rút chiều cao hero, giữ một sơ đồ và một câu mô tả. Đưa so sánh ba cách học và kết luận chính lên sớm hơn. Thu gọn phần ba bước nếu sơ đồ đã giải thích cùng nội dung. Thêm mục ngắn “Phần nhóm thực hiện” với những đóng góp đã xác nhận được, nêu rõ phần dùng Flower/MobileNet. |
| 2. Dữ liệu phân tán | Hiểu dữ liệu lệch; số 18/38 nhóm tập trung ở một client là thông điệp hữu ích. | Năm thẻ client có nhiều số và hai loại accuracy; bảng 38 nhóm × 5 client mở mặc định. Ví dụ cụ thể dễ hiểu hơn lại đang đóng. Tên cây/bệnh dài bằng tiếng Anh làm bảng khó đọc nhanh. | Mở ví dụ một bệnh làm nội dung chính: biểu đồ năm cột số ảnh + kết quả Local/FedAvg. Giữ chỉ số 18/38. Đưa bảng đầy đủ, năm thẻ chi tiết và 12 cách chia vào phần khám phá mở rộng. |
| 3. Kết quả | Tab Tổng hợp hiện khá gọn. Ba số lớn dễ nhìn, các tab giúp truy vấn sâu. | Ba thẻ đang trộn hai phép so sánh: Centralized/FedAvg trên PlantVillage và FedAvg trên PlantDoc. Chưa có câu chốt trả lời “FedAvg có ích gì nếu Centralized cao hơn?”. Biểu đồ lịch sử chỉ Centralized dễ kéo sự chú ý khỏi đóng góp FL. Tab Tập giữ riêng có khối văn dài trước bảng; nhiều thuật ngữ thống kê xuất hiện cùng lúc. | Phần Tổng hợp dùng một phép so sánh thống nhất: Local-only → FedAvg → Centralized trên cùng nguồn ảnh. Thêm một câu kết luận về lợi ích học chung và một câu phạm vi. Cho chuyển nguồn ảnh rõ ràng. Đặt lịch sử hội tụ và phương pháp thống kê trong phần mở rộng; tab giữ riêng hiển thị kết quả chính trước, giữ cảnh báo ngắn ngay cạnh bảng. |
| 4. Chẩn đoán | Đây là phần dễ hiểu và thuyết phục nhất khi có ảnh mẫu: một ảnh, các mô hình dự đoán khác nhau, đúng/sai rõ. Mẫu #12 minh họa được học chung. | Trước khi chạy đã có nhiều lựa chọn: bộ lọc, nhãn thật, 7 mô hình/2 mô hình, Grad-CAM. Sau khi chạy, tên bệnh lặp trong preview, tóm tắt và từng hàng; hai tỷ lệ thô/hiệu chuẩn cùng xuất hiện. “Vì sao?” có sáu dòng theo client; người xem có thể tưởng đây là bằng chứng nguyên nhân từ mô hình. | Đưa ảnh và tóm tắt ba nhóm học thành nội dung chính. Mặc định ba nhóm: Centralized, FedAvg, Local-only 1/5; mở “Xem 5 mô hình học riêng” để xem từng hàng. Giữ cấu hình nhãn thật/hiệu chuẩn/Grad-CAM trong tùy chọn có chú thích. Rút giải thích còn hai ý và đổi tên thành “Liên hệ với dữ liệu huấn luyện”. |
| 5. Huấn luyện thử | Nút Bắt đầu rõ; sơ đồ trực tiếp hợp với demo. Cảnh báo ảnh tổng hợp là cần thiết. | Cùng thông tin 5 client, 2 vòng, 1 epoch, localhost được nhắc ở subtitle, callout, chip và mô tả. Địa chỉ IP, α, RTT, Macro-F1 dễ gây phân tâm khi mục tiêu chỉ là chứng minh quy trình. | Một dòng cố định “Demo trên một máy · 5 client · 2 vòng · ảnh tổng hợp”. Giữ số vòng và trạng thái client ở vùng theo dõi. Chuyển IP, α, RTT, bảng chỉ số và cấu hình nâng cao vào chi tiết kỹ thuật. Giữ cảnh báo ngắn rằng kết quả này không phải accuracy đồ án. |
| 6. Bằng chứng & hạn chế | Thể hiện sự minh bạch; có chấm lại, kiểm tra trùng ảnh và công việc còn lại. | Ba card đều dài ngay màn đầu. “10/14” thiếu nghĩa trực quan cho người mới; “11.015” có thể khiến người xem bỏ qua việc chỉ có 172 ảnh PlantDoc. Năm hạn chế mở mặc định chứa nhiều đoạn số liệu. “Phần kỹ thuật và các con số đã kiểm chứng xong” nghe rộng hơn phạm vi checklist. Trình chiếu chỉ phóng chữ nên card càng dài. | Mặc định ba dòng kiểm chứng có phạm vi, ba hạn chế chính và ba bước tiếp theo. Chi tiết số liệu/đối soát/thống kê để mở. Dòng tập giữ riêng cần nêu riêng 172 ảnh ngoài đồng. Đổi tuyên bố checklist thành “19/22 hạng mục trong checklist dự án đã hoàn thành”. |

Độ dài trang mặc định quan sát tại 1280×720: Tổng quan khoảng 2,8 màn; Dữ liệu 2,7; Tổng hợp kết quả 1,6; Chẩn đoán trước chạy 1,7 và sau mẫu #12 khoảng 2,9; Huấn luyện 2,0; Bằng chứng 2,8. Đây là chiều cao tài liệu chia cho viewport, gồm khoảng đệm và vùng điều hướng; không phải số từ hoặc thời gian đọc. Bảng/vùng cuộn nội bộ còn có nội dung ngoài phần đang nhìn thấy.

## “Hardcode” ở đâu và có đáng lo không?

### Số liệu chính được lấy từ dữ liệu

`web/static/js/core/store.js:20-27` lấy evidence/summary từ API. `web/static/js/pages/overview.js:159-171` lấy metrics và tính hiệu số. Các số 93,12%, 99,22%, khoảng Local-only, +37,7 điểm và −45,9 điểm được render từ dữ liệu/tính toán. `web/static/js/pages/noniid.js:238` lấy 18/38 từ báo cáo shard. Số đúng/sai ở chẩn đoán được tính từ dự đoán hiện tại và nhãn thật. Vì vậy không có bằng chứng rằng các headline số này chỉ là chữ số trang trí viết sẵn.

### Lời giới thiệu là câu viết sẵn

Hero, nhãn, mô tả ba cách học, CTA và nhiều cảnh báo là template HTML/JavaScript viết sẵn. Đây là nội dung biên tập của sản phẩm; vấn đề cần xử lý là tính rõ ràng, độ dài và tính đúng theo phạm vi, chứ không phải mọi câu viết sẵn đều cần AI sinh lại.

Các nhận định cần viết chính xác hơn:

- `overview.js:205`: “Chính xác nhất” nên thành “Cao nhất trong các mô hình đang so sánh trên nguồn ảnh này”, hoặc tạo câu theo dữ liệu nếu bộ số liệu thay đổi.
- `overview.js:211`: “không riêng tư” quá tuyệt đối. Nên chỉ mô tả dữ kiện “Ảnh gốc được gửi về máy chủ”. Cơ chế bảo vệ dữ liệu của một hệ centralized phụ thuộc thiết kế cụ thể.
- `evidence.js:8`: “Chưa phải ảnh ngoài thực địa” dễ mâu thuẫn với PlantDoc là ảnh ngoài đồng. Nên đổi thành “Chưa có kiểm thử thực địa độc lập”.
- `evidence.js:121-125`: kết luận hoàn tất phần kỹ thuật dựa trên trạng thái checklist của dự án. Nên nêu rõ phạm vi checklist để tránh bị hiểu là chứng nhận mọi yêu cầu kỹ thuật/UI.

### “Vì sao?” là diễn giải theo quy tắc

`diagnosis.js:316-355` phân nhóm dự đoán đúng/sai, tra số ảnh của lớp trong shard rồi ghép câu mẫu. Khi client có 0 ảnh và dự đoán sai, UI viết “chưa từng thấy nhóm này”; khi có ít ảnh và sai, UI viết “chỉ có … ảnh … và vẫn sai”. FedAvg đúng thì ghép câu về tổng ảnh của năm client.

Các số này có nguồn dữ liệu và hữu ích để giải thích sự liên hệ. Tuy nhiên, template này chưa chứng minh nguyên nhân của một dự đoán cụ thể. Đề xuất nhãn **“Liên hệ với dữ liệu huấn luyện”**, kèm câu “Sự thiếu hụt dữ liệu có thể góp phần vào lỗi; đây chưa phải kiểm chứng nguyên nhân”. Grad-CAM là phần khác, cho xem vùng ảnh ảnh hưởng tới mô hình; heatmap cũng không tự chứng minh quan hệ nhân quả.

### Dễ/Khó là nhãn catalog có sẵn

`scripts/build_demo_gallery.py:137` đọc cột `do_kho` từ `data/demo_images/labels.csv`; UI đọc catalog để hiển thị. Catalog mô tả việc xếp dễ/khó theo recall của **lớp** của FedAvg trên validation cùng nguồn. Đây không phải độ khó đo mới cho từng ảnh hoặc đánh giá của cả bảy mô hình.

Nên gọi “Nhóm FedAvg thường nhận tốt / khó nhận” hoặc giữ Dễ/Khó và thêm chú thích “Theo tỉ lệ nhận đúng của lớp, không phải đánh giá riêng ảnh này”. Không nên khiến hội đồng hiểu mức khó là kết quả suy luận trực tiếp.

### Có metadata thực nghiệm viết cố định

Số lớp 38, năm client và α=0,1 xuất hiện trong nhiều template; `evidence.js:66` viết cố định Centralized 83 epoch/AdamW và FedAvg 32 vòng/SGD/chọn vòng 27; `results.js:109,413` viết cố định thông tin 27/32 vòng. Các con số này đang mô tả bundle hiện tại, nhưng sẽ dễ lệch khi thay bundle.

Nên lấy số lớp/client từ taxonomy/shards, và lịch sử/thiết lập thực nghiệm từ metadata chung. Với demo cố định 2 vòng/1 epoch, vẫn có thể giữ cấu hình cố định nhưng UI đọc cùng cấu hình với backend thay vì lặp giá trị ở nhiều nơi.

## Những câu hội đồng có thể hỏi mà UI nên trả lời sớm

1. **Tại sao cần FedAvg nếu Centralized đạt cao hơn?** Hiện có số liệu nhưng thiếu câu chốt nổi bật. Nên giải thích mục tiêu học chung khi ảnh phân tán, nêu chênh accuracy quan sát được và điều kiện huấn luyện chưa đồng nhất.
2. **Năm client có phải năm máy/nông hộ thật không?** Cần nêu ở Tổng quan rằng thực nghiệm dùng năm phân vùng giả lập; demo Flower chạy trên một máy. Hiện thông tin này nằm sâu ở hạn chế.
3. **Bảy mô hình là những mô hình nào?** Nên có nhãn ngắn “1 mô hình gom ảnh + 1 mô hình học liên kết + 5 mô hình học riêng”.
4. **93,12% có dùng được ngoài đồng không?** Phải giữ nhãn nguồn ảnh/tập đánh giá và hạn chế ngắn bên cạnh số; hiển thị kết quả PlantDoc là nguồn khác, không đồng nhất với kiểm thử thực địa độc lập.
5. **Phần nào do nhóm tự làm?** Cần nêu đóng góp đã xác nhận và vai trò của Flower, backbone MobileNet; không nên chỉ giới thiệu khái niệm FL.
6. **Giải thích “Vì sao?” có phải XAI không?** Phân biệt rõ template dựa vào phân bố dữ liệu và Grad-CAM.

Một điểm gây nhầm cụ thể: sơ đồ Tổng quan/Flower đánh số Client 01–05, còn Dữ liệu và mô hình Local dùng 00–04 (`core/flnet.js:52` và `noniid.js:45`). Nên dùng một quy ước hiển thị trên toàn web và có ánh xạ với ID nội bộ.

## Bản câu chữ đề xuất

Đây là nội dung đề xuất biên tập; các số khi triển khai tiếp tục lấy từ API.

| Vị trí | Câu gợi ý |
| --- | --- |
| Hero | **Học chung để nhận diện bệnh lá. Ảnh gốc ở lại từng nơi.** |
| Mô tả hero | “Năm client trao đổi trọng số để tạo mô hình chung nhận diện 38 nhóm lá khỏe/bệnh.” Thêm nhãn phạm vi “Thực nghiệm: 5 phân vùng dữ liệu giả lập”. |
| Kết luận kết quả | “Trong lần chạy này, FedAvg đạt 93,12% trên PlantVillage, cao hơn mô hình học riêng tốt nhất 37,7 điểm phần trăm.” |
| Phạm vi ngay dưới kết quả | “Kết quả trên bộ dữ liệu hiện có; các mô hình chưa huấn luyện cùng điều kiện. Chưa có kiểm thử thực địa độc lập.” |
| Ví dụ dữ liệu | “97% ảnh của nhóm này nằm ở một client. Bốn nơi còn lại có rất ít hoặc không có ảnh để học.” Kèm biểu đồ và kết quả, mở chi tiết để xem từng count. |
| Demo mẫu #12 | “FedAvg và Centralized nhận đúng; chỉ 1/5 mô hình học riêng nhận đúng.” Dòng phụ: “Minh họa trên một ảnh mẫu có nhãn thật.” |
| Giải thích chẩn đoán | “Liên hệ với dữ liệu: client nhận sai có ít hoặc không có ảnh của nhóm này. Điều này có thể góp phần vào lỗi.” |
| Huấn luyện | “Demo trên một máy · 5 client · 2 vòng · ảnh tổng hợp.” Dòng phạm vi: “Dùng để kiểm tra quy trình; accuracy của đồ án xem ở Kết quả.” |
| Bằng chứng | “Centralized và FedAvg được chấm lại khớp số công bố.” “Không phát hiện trùng ảnh trong các split đã kiểm.” “Tập giữ riêng có 172 ảnh ngoài đồng; kết quả còn dao động rộng.” Các ngoại lệ và phạm vi chi tiết vẫn cần ở mục mở rộng. |

## Thứ tự sửa đề xuất

1. **Sửa câu có thể gây hiểu sai:** nhãn “Vì sao?”, ý nghĩa Dễ/Khó, “không riêng tư”, “chưa phải ảnh ngoài thực địa”, quy ước Client 00/01 và phạm vi checklist hoàn tất.
2. **Rút nội dung mở mặc định:** Bằng chứng chỉ ba hạn chế lớn; Tập giữ riêng đưa kết quả lên trước khối phương pháp; Dữ liệu lấy một ví dụ làm trung tâm, bảng đầy đủ mở theo yêu cầu. Giữ caveat thiết yếu cạnh metric để không làm người xem hiểu quá mức.
3. **Làm rõ câu chuyện đồ án:** Tổng quan nêu mục tiêu, đóng góp đã xác nhận và lợi ích học chung; Tổng hợp kết quả dùng ba nhóm học trên cùng nguồn ảnh; Chẩn đoán lấy kết luận ba nhóm làm trọng tâm.
4. **Tối ưu trình chiếu theo nội dung:** chữ lớn đi cùng ít nội dung mở. Hiện chỉ phóng chữ làm card dài hơn. Mỗi bước nên có một kết luận và một hình chính phù hợp màn 1280×720, phần sâu mở khi hỏi.
5. **Chuẩn hóa dữ liệu và cách gọi:** dùng nguồn metadata chung cho số lớp/client/cấu hình; tên cây tiếng Việt ngắn, tên bệnh khoa học chính xác có thể để dòng phụ. Các bản dịch bệnh cần được đối chiếu trước khi dùng.
6. **Hoàn thiện chuyển động sau khi rút nội dung:** hoạt ảnh hiện có ở sơ đồ phục vụ minh họa tốt. Dùng chuyển trạng thái ngắn và nhấn đúng bước gửi/gộp trọng số; độ mượt cần đo riêng nếu cam kết 60 FPS. Không có phép đo FPS trong review này.

## Điều kiện kiểm lại sau khi sửa

Cho một người chưa biết dự án xem Tổng quan khoảng 30 giây và hỏi họ nêu mục tiêu, cách làm, lợi ích, phạm vi. Ở mỗi bước họ cần nói được một kết luận chính mà không đọc toàn bộ bảng. Thử trình chiếu ở 1280×720, kiểm tra chữ và metric từ khoảng cách phòng bảo vệ. Kiểm tra mọi số/tuyên bố vẫn khớp nguồn dữ liệu sau khi rút gọn; nội dung sâu còn truy cập được để trả lời câu hỏi hội đồng. Đây là phép thử đề xuất cho vòng sửa, chưa được thực hiện với người dùng thật.
