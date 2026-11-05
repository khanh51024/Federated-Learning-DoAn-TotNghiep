# Hướng dẫn demo

Mọi thứ chạy từ một file: nhấp đúp **`DEMO_TUNG_MUC.bat`** ở thư mục dự án. Chọn **A** cho 9 mục của thầy, **B** cho file Word, rồi gõ mã mục và Enter.

**Từ cần biết**

| Từ | Nghĩa |
|---|---|
| Client | Một máy tham gia học, giữ ảnh của riêng mình |
| Trọng số | Phần "kiến thức" của mô hình, thứ duy nhất được gửi qua mạng |
| Vòng | Máy chủ gửi mô hình xuống, các client học, gửi lại, máy chủ gộp thành mô hình chung |
| Dấu vân tay (SHA-256, "băm") | Chuỗi ký tự tính từ trọng số. Khác một chút là chuỗi khác hẳn |
| Độ trễ (RTT) | Thời gian gửi tín hiệu đi và nhận lại |
| Loss | Mức sai của mô hình, càng thấp càng tốt |
| α | Mức lệch dữ liệu. α càng nhỏ, mỗi máy càng chỉ có vài loại bệnh |
| Flower | Phần mềm điều phối việc học phân tán |

## 1. Chuẩn bị trước giờ demo

1. Chép `tmp\LAPTOP_2_10_CLIENT.zip` sang laptop 2, giải nén ra một thư mục. Không phải cài gì. Windows chặn file `.bat` thì xem bảng lỗi ở mục 5.
2. Oracle Cloud còn rule mở `18089-18090` cho `0.0.0.0/0`. Xóa rule này sau khi demo xong.
3. Cả 2 laptop: tắt game và ứng dụng nặng, cắm sạc, tắt WARP hoặc VPN nếu được.
4. Diễn tập trọn mục 2 một lần với 2 laptop, bấm giờ.
5. Mở sẵn file này để nhìn khi nói.

## 2. Kịch bản demo 15 phút (bản mới, dùng web)

**Mỗi phần chứng minh gì:**

| Phần | Trả lời câu hỏi | Kết quả cần thấy | Có phải số của đề tài? |
|---|---|---|---|
| Trang Đo qua mạng | Phân tán có chạy được qua mạng thật không? | Máy chủ riêng, đủ client, mã băm khớp, độ chính xác tăng dần | Không |
| Chương 4, mục 4.7 | Phân tán thua tập trung bao nhiêu? | Tập trung 97,69%, phân tán 96,97 đến 93,93%, tự học 46,55% | **Có** |
| Trang Cộng đồng tham gia | Người dùng thường có đóng góp được không? | Chỉ mở khi thầy hỏi, phiên đang đóng | Không |

Mở sẵn 2 tab: `https://datn.tranngochung.id.vn/#flower` và chương 4, tab **Phân tán và tập trung**. Kể theo các câu hỏi dưới, không đi theo danh sách nhiệm vụ.

| Phút | Câu hỏi | Làm gì | Câu chốt |
|---|---|---|---|
| 1 | Đề tài chứng minh gì? | Chỉ nói | "Dữ liệu nằm riêng ở từng nơi, chỉ gửi trọng số, mà vẫn tiệm cận học tập trung." |
| 4 | Có thật sự gửi qua mạng không? | Chạy 7 client, 3 vòng (xem dưới). Xem `#flower` tự chạy. Bấm ô **Trọng số gửi đúng**, rồi bấm một client | "Máy chủ riêng, 2 laptop. Client không gửi độ chính xác, loss, thời gian: máy chủ tự tính. Mã băm hai đầu phải giống nhau." |
| 2 | Chạy lâu hơn thì sao? | Chọn lần đo 10 client, kéo thanh chọn vòng | "20 vòng, mã băm khớp 200/200, trình kiểm đạt 17/17. Độ chính xác tăng từ 13% lên 90% chứng minh mô hình gộp thật sự học được. Đây là số thật trên tập nhỏ 190 ảnh, số so sánh của đề tài ở chương 4." |
| 1 | Lỡ kiểm tra lúc nào cũng ra đúng thì sao? | Bấm ô **Thử cố tình sai** | "Khi em cố tình đổi trọng số trên đường truyền, máy chủ phát hiện và loại client đó." |
| 3 | Phân tán thua tập trung bao nhiêu? | Chương 4, biểu đồ theo mức lệch | "Tập trung 97,69%. Phân tán 96,97% khi gần như chia đều, 93,93% khi lệch mạnh. Tự học một mình 46,55%. Lệch càng mạnh, khoảng cách càng rộng." |
| 2 | Ai chịu thiệt khi lệch? | Bảng từng client, bảng client trôi xa mô hình chung | "Lệch mạnh thì client yếu nhất 84,29%, mạnh nhất 94,59%, độ lệch chuẩn từ 0,53 lên 3,60. Client cũng trôi xa mô hình chung hơn: từ 0,038 lên 0,147." |
| 2 | Còn gì chưa xong? | Chỉ nói, chủ động nói trước | "Tự học một mình mới có α=0,1. Lệch số lượng, lệch miền mới chia dữ liệu. 4 thuật toán có mã và test, chưa có kết quả. Tập niêm phong chưa mở, đúng quy tắc." |
| 1 | Kế hoạch tiếp | Chỉ nói | "Khi có GPU: xong Đợt 2 trước, rồi FedProx, FedBN, SCAFFOLD, MOON." |

**Chạy trực tiếp bằng nút trên web (không cần file `.bat` lúc demo):**

Chuẩn bị một lần: mỗi laptop nhấp đúp `CAI_NUT_WEB.bat` (laptop 1 ở `deploy\lan_client`, laptop 2 trong thư mục đã giải nén). Đăng nhập ở `/quan-tri` trên laptop 1.

1. Laptop 1, trang `#flower`: chọn `7 client, 3 vòng`, bấm **Bắt đầu lần đo**. Máy chủ mở trên VM, nhật ký máy chủ hiện ngay trên trang.
2. Trên từng laptop, mở trang `#flower`, bấm **Mở client trên máy này**. Trình duyệt hỏi thì chọn Mở.
3. Đủ client thì tự học. Muốn dừng sớm, bấm **Dừng lần đo** trên laptop đã bấm Bắt đầu.

Sau demo, mỗi laptop nhấp đúp `GO_NUT_WEB.bat` để gỡ.

**Laptop chưa có gì:** mở trang `#flower`, bấm **Tải bộ cài cho máy này**, mở file vừa tải. File tự tải gói, kiểm mã băm và cài nút mở client, không cần chép gì sang máy đó. Máy cài kiểu này đóng vai laptop 2 (client 5 và 6). Gói dựng bằng `python scripts/build_client_package.py`, nằm ở `~/client-package` trên VM.

**Máy tự nối với trang:** cài xong, máy hiện ở mục **Máy đã cài bộ client** trên `#flower` sau vài giây, kèm mã 6 ký tự do trình cài in ra, và tự nối lại mỗi lần đăng nhập Windows. Chọn lần đo bộ ảnh nhẹ, đánh dấu các máy, bấm **Bắt đầu lần đo**: client chia đều cho các máy đã chọn và tự mở, không cần bấm gì trên máy đó. Trong lần đo, các máy hiện thành Máy A, Máy B, Máy C. Lần đo bộ thường vẫn bấm tay vì mỗi máy cài chỉ có ảnh của client 5 và 6. Laptop 1 chạy lại `deploy\lan_client\CAI_NUT_WEB.bat` thì cũng hiện trên trang. Tắt hẳn: `GO_NUT_WEB.bat`. Nhật ký của máy ở `%LOCALAPPDATA%\CropFedClient\agent.log`.

**Chỉ có 1 laptop, muốn thấy độ chính xác tăng:** chọn `10 client, 10 vòng, bộ ảnh nhẹ`, bấm **Bắt đầu lần đo**, rồi bấm **Mở client trên máy này** trên laptop đó. Bộ này có 10 trong 38 loại bệnh, mỗi client 40 ảnh, nên đây chỉ là bằng chứng mô hình gộp học được, không phải kết quả của đề tài. Trang tự ghi chú điều này và đặt tên lần đo là `bộ ảnh nhẹ`.

Chuẩn bị bộ nhẹ một lần: `python scripts/prepare_network_speedtest.py --out tmp/mini10 --clients 10 --per-class 4 --val-per-class 3 --test-per-class 5 --class-step 4`, chép `partition` và `images_eval` lên `~/mini10-data` của VM, rồi chạy lại `CAI_NUT_WEB.bat` trên laptop 1 để có thêm nút.

**Dự phòng khi nút trên web lỗi, chạy bằng file:**

1. Laptop 1: cắm ổ F chứa khóa SSH, nhấp đúp `DEMO_NHANH.bat`. File tự mở máy chủ trên VM, mở trang web và mở 5 client.
2. Khi file báo `BÂY GIỜ`, laptop 2 nhấp đúp `LAPTOP_2_DEMO_NHANH.bat` trong thư mục đã giải nén.
3. Muốn cho thầy thấy client gửi thật: mở cửa sổ một client, so `gửi trọng số mã băm ...` với cột mã băm trên trang web ở cùng vòng. Hai mã phải giống nhau.

Không có ổ F: đặt biến `DEMO_SSH_KEY` trỏ tới khóa, hoặc bỏ phần chạy trực tiếp.

Trước đó tắt ứng dụng nặng, cắm sạc cả 2 máy. Nếu mạng hoặc laptop 2 lỗi thì bỏ phần chạy trực tiếp, dùng lần đo 10 client đã lưu và thanh chọn vòng; số đã lưu không bị ảnh hưởng.

**Về 2 bộ số:** số trên web là bộ seed 42, 123, 2026 trong repo này. Báo cáo Đợt 2 của bạn cùng nhóm là bộ chạy riêng (seed 42, 43, 44, có thêm PD-463, EXT-TEST). Không đọc số của bộ này như số của bộ kia.

Nếu thầy hỏi theo file Word thì chuyển sang phần B ở mục 4.

## 3. Phần A: 9 mục của thầy

**A → 1. Gửi trọng số có bị sai không**
- **Thấy:** 6 dòng `400/400 lượt khớp`.
- **Cách làm:** client tính dấu vân tay ngay trước khi gửi, máy chủ tính lại ngay sau khi nhận rồi so. Làm cả hai chiều, mọi client, mọi vòng. Lệch một lần là dừng cả lần chạy.
- **Nói:** "6 lần chạy, mỗi lần 40 vòng, 5 client, 2 chiều, tổng 2.400 lần kiểm tra, khớp hết."

**A → 2. So với tập trung có lệch không, kiểm tra trọng số** (khoảng 1,5 phút)
- **Thấy:** `KHỚP`; dòng A rất nhỏ (khoảng 0,00000000000006); dòng B lớn hơn nhiều (0,13); rồi `94.72%`.
- **Cách làm, 3 bước:** (1) So dấu vân tay mô hình xuất phát của hai bên. (2) Cho mỗi client học đúng 1 bước rồi gộp, về toán học phải trùng tập trung; sau đó cho học nhiều bước. (3) So hai mô hình đã học xong.
- **Nói:** "Một bước thì trùng, nên gửi và gộp không gây lệch. Nhiều bước mới lệch. Hai mô hình học xong cho cùng đáp án ở 94,72% ảnh."
- **Lưu ý:** bước 2 ở đây là bản nhanh trên ít ảnh. Bản đầy đủ trên GPU ra 0,00000000005, cùng kết luận.

**A → 3. Phương pháp kiểm tra**
- **Thấy:** `PASSED`, `1 passed`.
- **Nói:** "Em kiểm tra 5 thứ: dấu vân tay khi gửi; tính lại phép gộp bằng công cụ khác; cùng mô hình xuất phát; một bước phải trùng tập trung; so mô hình đã học xong. Test này cố tình đổi 2.0 thành 2.0000002, hệ thống phải dừng lại, để chứng minh phép kiểm tra thật sự bắt được lỗi."

**A → 4. Nhiều client**
- **Thấy:** client 0 đến 4 ở vòng đầu và vòng cuối.
- **Nói:** "Thí nghiệm chính có 5 client mô phỏng trên một GPU. Phần đo tốc độ là 2 máy thật."

**A → 5. Thông số Flower** (khoảng 1,5 phút)
- **Thấy:** 3 cửa sổ mở ra, rồi bảng từng vòng: số ảnh, thời gian học, độ trễ, tốc độ, `KHỚP`, và mức sai giảm dần (3,27 → 2,82 → 2,30).
- **Nói:** "1 máy chủ và 2 client chạy thật, mỗi client chỉ học ảnh của mình."
- **Lưu ý:** độ trễ chỉ 0,1 ms vì 3 chương trình cùng trên một máy. Khởi đầu từ mô hình có sẵn để mức sai giảm rõ, chỉ để trình diễn, không phải số báo cáo. Chạy lại phải đóng các cửa sổ cũ.

**A → 6. Mô hình tổng và mô hình client có sai số không**
- **Thấy:** `sai số gộp` rất nhỏ; `cách mô hình chung` từ khoảng 3 giảm dần về 0,02.
- **Nói:** "Có 2 chuyện khác nhau. Sai số phép gộp chỉ ở mức làm tròn số, lớn nhất 0,0000002 qua 720 vòng. Còn client khác mô hình tổng là do mỗi máy học ảnh khác nhau, đó không phải lỗi."

**A → 7. Code backend** (mở VS Code)
- **Tập trung:** `train_v4.py` dòng 189, lấy toàn bộ ảnh.
- **Phân tán:** `flower_app.py` dòng 206, gửi trọng số, nhận về, kiểm tra.
- **Gộp:** `strategy.py` dòng 133, trung bình theo số ảnh.
- **Hàm học chung:** `loop.py` dòng 67.
- **Nói:** "Hai cách dùng chung một hàm học, chỉ khác cách dùng dữ liệu và cách trao đổi trọng số."
- **Phím T:** chạy thật cả hai cách với 200 ảnh, khoảng 2 phút. Độ chính xác rất thấp vì quá ít ảnh, có thể hiện dòng chữ đỏ lúc tắt, không phải lỗi. Chỉ dùng khi thầy muốn xem chạy thật.

**A → L hoặc A → 8. Đo tốc độ 2 máy**
- **L:** đo trực tiếp 2 laptop, xem mục 5.
- **8:** số đã đo. Ngày 02/11 laptop 18,0 giây, máy GPU 0,9 giây, độ trễ 1.425 ms qua đường hầm SSH. Ngày 20/10 laptop 20,9 giây, máy Oracle 5,2 giây, độ trễ 47 ms qua Internet.
- **Cách làm:** mỗi vòng máy chủ gửi một gói đúng cỡ mô hình (khoảng 6 MB) rồi bấm giờ, ra độ trễ và tốc độ. Thời gian học do mỗi máy tự đo.

**A → 9. Khác biệt phân tán và tập trung**

| | Tập trung | Phân tán |
|---|---|---|
| Ảnh | Gom về một chỗ | Ở nguyên từng máy |
| Gửi qua mạng | Ảnh, 1.524 MB một lần | Trọng số, khoảng 2.510 MB (40 vòng) |
| Độ chính xác | 97,69% | 96,97% (gần đều) đến 93,93% (rất lệch) |

- **Nói:** "Phân tán không tiết kiệm mạng trong cấu hình này. Lợi ích là ảnh không rời máy."

## 4. Phần B: file Word

Ký hiệu: ✅ làm trực tiếp được, 📊 xem số đã đo, ⚠️ làm được một phần, ❌ chưa có.

**Đợt 1: Nền tảng so sánh**

| Mã | Mức | Thấy | Nói |
|---|---|---|---|
| 1.1 | ✅ | Bảng độ chính xác | "SGD 97,69%, AdamW 97,80%, chênh rất ít. Em chọn SGD làm trần vì cùng cách học với client." |
| 1.2 | ✅ | Bảng, thêm phím T | "Flower 94,03%, tự cài 93,93%, chênh 0,11 điểm. Riêng ảnh ngoài đồng chênh 0,96 điểm, nên em không kết luận tương đương trên mọi nhóm." |
| 1.3 | ⚠️ | Nhật ký từng vòng | "Phân tán đủ từng vòng. Local-only là đợt cũ, chưa có số byte và độ lớn cập nhật." |
| 1.4 | ✅ | Tên run, `2 passed` | "Em niêm phong 7.005 ảnh, chưa mở. Test chứng minh chỉ chạy được một lần." |

**Đợt 2: Nhiều mức lệch dữ liệu**

| Mã | Mức | Thấy | Nói |
|---|---|---|---|
| 2.1, 2.4 | ⚠️ | Bảng, 2 biểu đồ | "Phân tán đủ 5 mức α, mỗi mức 3 lần chạy. Khoảng cách từ 0,73 điểm lên 3,77 điểm khi dữ liệu càng lệch. Local-only mới có ở α=0,1." |
| 2.2 | ⚠️ | Số ảnh từng client | "Đã chia xong dữ liệu: lệch số lượng gấp 7,7 lần, lệch nguồn ảnh từ 0% đến 83% ảnh ngoài đồng. Chưa train vì máy thuê mất kết nối." |
| 2.3 | ✅ | Bảng theo client, bảng client trôi xa mô hình chung | "Rất lệch: trung bình 90,79%, mạnh hơn yếu 10,30 điểm, client ít ảnh nhất 84,29%. Client cũng trôi xa mô hình chung hơn: từ 0,038 lên 0,147." |

- **Nếu hỏi vì sao Local-only ở bảng 2.3 cao tới 96,57%:** bảng này đo trên loại bệnh mà chính client đó có. Trên ảnh chung thì Local-only chỉ 46,55%.
- **Nếu hỏi nguyên nhân suy giảm:** "Em đo được 2 cơ chế tạo lệch: học nhiều bước trên dữ liệu riêng và BatchNorm. Em chưa tách được mỗi nguyên nhân chiếm bao nhiêu."

**Đợt 3: Thuật toán chịu dữ liệu lệch**

| Mã | Mức | Thấy | Nói |
|---|---|---|---|
| 3.1 | ⚠️ | `30 passed`, code | "Đã cài FedProx, FedBN, SCAFFOLD, MOON và có test. Chưa có thí nghiệm đầy đủ." |
| 3.1T | ✅ | FedProx, FedBN chạy trên Flower, khoảng 2,5 phút | "Chạy với rất ít ảnh để thấy thuật toán chạy được, không dùng số này." |
| 3.3 | ⚠️ | `2 passed` | "Có 2 hướng: kết hợp FedBN với FedProx (mới có code), và tinh chỉnh riêng cho từng client (mới chạy 1 lần: 97,01% trên dữ liệu của client, 82,36% trên ảnh chung). Chưa đủ 3 lần chạy nên chưa kết luận." |
| 3.2, 3.4 | ❌ | | "Chưa dò tham số, chưa có so sánh thu hẹp khoảng cách." |

**Đợt 4: Tối ưu và riêng tư**

| Mã | Mức | Thấy | Nói |
|---|---|---|---|
| 4.1 | 📊 | Bảng nén | "Mô hình int8 còn 2,12 MB so với 6,24 MB, độ chính xác 93,81% so với 93,88%, 5,14 ms một ảnh trên laptop. Chưa đo trên điện thoại." |
| 4.2 | 📊 | Bảng nén | "Nén 8-bit còn 26,4% dung lượng, độ chính xác gần như không đổi. Em đo bằng cách nén một lần sau khi học xong, chưa học với nén ở mỗi vòng. Số 661 MB cho 40 vòng chỉ là ước tính." |
| 4.3 | ⚠️ | `4 passed` | "Phần này là tùy chọn. Đã có code thêm nhiễu và tính ε, nhưng chưa chạy đường ε – độ chính xác vì chưa đủ thời gian GPU." |
| 4.4 | ✅ | Trang Flower, lần đo 10 client | "10 client trên 2 laptop, máy chủ riêng trên VM, qua Internet thật." |

- **Nếu hỏi vì sao không dùng bản int8 nén toàn bộ:** bản đó độ chính xác tụt còn 9,98%, nên em chỉ nén một phần các lớp.

**Đợt 5: Tổng hợp**

| Mã | Mức | Thấy | Nói |
|---|---|---|---|
| 5.1 | ⚠️ | Bảng, biểu đồ | "Có ba mốc và phân tán nhiều mức lệch. Còn thiếu thuật toán cải tiến và Local-only nhiều mức." |
| 5.2 | ⚠️ | 3 bảng | "Có đủ độ chính xác, độ bền khi lệch, truyền thông và công bằng. Riêng tư mới dừng ở phân tích: ảnh không rời máy, chưa có DP." |
| 5.4 | ✅ | Cấu hình một lần chạy, `KHỚP` | "Mỗi lần chạy lưu cấu hình, seed, phiên bản mã, mã băm dữ liệu." |
| 5.3 | ❌ | | "Chạy tập niêm phong sau khi chốt cấu hình cuối, nên chưa mở." |
| 5.5 | ❌ | | "Mới có báo cáo kiểm chứng, chưa có quyển báo cáo và bản thảo bài báo." |

**Câu nói về tiến độ:** "Em chưa đạt đủ mức Tối thiểu. Để đủ, còn phải train 2 dạng lệch cho phân tán (6 lần chạy, khoảng 1 giờ GPU) và Local-only ở các mức còn lại (90 lần chạy). Máy thuê bị mất kết nối giữa chừng nên phần này chưa có."

## 5. Đo trực tiếp giữa 2 laptop

**Cách cũ, chỉ dùng khi không nối được VM:** máy chủ nằm chung laptop chính nên không phải máy chủ tách riêng. Khi demo chính dùng `DEMO_NHANH.bat` ở mục 2.

Laptop chính là máy chủ và client 0, laptop 2 là client 1. Mất khoảng 2 phút.

1. Cho 2 laptop vào **cùng một mạng**. Chắc nhất là cùng bắt điểm phát từ điện thoại, vì wifi trường hay chặn 2 máy nói chuyện với nhau.
2. **Laptop chính:** menu A → L. Windows hỏi quyền thì chọn **Yes**. Nếu Windows hỏi cho Python qua tường lửa thì chọn **Allow**. Màn hình sẽ hiện một địa chỉ IP, ví dụ `192.168.1.20`.
3. **Laptop 2:** nhấp đúp `CHAY_CLIENT.bat` trong thư mục đã giải nén, nhập IP đó, Enter.
4. Chờ xong, laptop chính in bảng: thời gian học, độ trễ, tốc độ gửi và nhận, số lần khớp.

**Đọc kết quả:** độ trễ và tốc độ mạng thật là dòng **Laptop 2**. Laptop 1 cùng máy với máy chủ nên số mạng rất lớn.

**Không nối được:**

| Hiện tượng | Xử lý |
|---|---|
| Laptop 2 báo `KHÔNG NỐI ĐƯỢC` | Kiểm tra IP. Đổi sang điểm phát điện thoại, rồi chạy lại vì IP sẽ đổi |
| Laptop chính báo chưa mở được cổng | Chạy lại và chọn Yes |
| Laptop chính báo cổng đang bận | Đóng các cửa sổ "Máy chủ Flower", "Client" cũ |
| Windows chặn file `.bat` ("Windows đã bảo vệ PC của bạn") | Chuột phải trong thư mục đã giải nén, chọn Mở trong Terminal, gõ `Get-ChildItem -Recurse \| Unblock-File`, rồi nhấp đúp lại. Hoặc bấm "Thông tin thêm" rồi "Vẫn chạy" |
| Laptop 2 báo thiếu DLL hoặc không chạy được Python | Chụp màn hình gửi lại. Khi đang demo thì dùng A → 8 |
| Vẫn không được khi đang demo | Dùng A → 8 và nói: "Mạng ở đây chặn, em cho xem số đã đo." |

Cổng tường lửa chỉ nhận máy cùng mạng và tự đóng sau khi đo xong, lâu nhất 30 phút.

## 5b. Đo 10 client, máy chủ tách riêng (đúng 8 ý thầy dặn ngày 03/11)

Máy chủ chạy trên VM Oracle `149.118.51.118`, **không có client nào trên VM**. Mỗi laptop chạy 5 client, tổng 10, học 20 vòng. Máy chủ chỉ bắt đầu khi đủ 10 client nối vào, nên không có client vào trễ.

| Ý của thầy | Đã làm ở đâu |
|---|---|
| 1. Từ 5 client trở lên, có số tối thiểu | `--num-clients 10 --min-clients 10` ở máy chủ. Muốn thử "tổng 10, chỉ cần 5 là chạy" thì đặt `--min-clients 5` (số vòng đầu có thể thiếu client nên **không dùng để lấy số**) |
| 2. Máy chủ tách riêng | Máy chủ ở VM, client ở 2 laptop. Độ trễ đo qua Internet thật, không qua đường hầm |
| 3. Nhiều vòng | 20 vòng |
| 4. File .bat mở client cùng lúc | `deploy\lan_client\CHAY_CLIENT.bat <client đầu> <số client> <IP máy chủ>` |
| 5. Client không gửi độ chính xác, thời gian | Client **không** còn gửi độ chính xác, loss, thời gian, độ trễ. Máy chủ tự tính, và tự tính lại độ biến thiên từ trọng số nhận được để đối chiếu với số client khai (lệch thì loại client đó) |
| 6. Máy chủ tính trung bình, độ chính xác, độ trễ | Trung bình theo số mẫu; độ chính xác tính trên ảnh giữ riêng của máy chủ sau mỗi vòng; độ trễ và tốc độ do máy chủ tự đo. Gửi đúng trọng số: băm SHA-256 khớp từng lượt, byte đúng 6.307.176 mỗi chiều |

**Chuẩn bị một lần (chủ tài khoản Oracle làm):** thêm quy tắc cho 2 laptop nối vào VM, đo xong thì xóa.

1. Mỗi laptop tắt Cloudflare WARP hoặc VPN (bật thì IP công khai đổi theo lần nối và số mạng đo cả đường vòng qua VPN), rồi mở trình duyệt vào `https://api.ipify.org` để biết IP công khai (nếu cùng một wifi thì hai máy cùng IP).
2. Oracle Cloud Console, Networking, Virtual Cloud Networks, chọn mạng của VM, Security Lists, danh sách mặc định, **Add Ingress Rules**: Source CIDR `<IP laptop>/32`, IP Protocol `TCP`, Destination Port Range `18089-18090`. Thêm một dòng cho mỗi IP.
3. Báo lại IP của 2 laptop. Phần mở cổng trên chính VM do script tự làm và tự đóng.

**Ngày đo:**

1. **VM** (Claude chạy giúp): `bash <thư mục bản web>/scripts/remote/run_network_server.sh --ip <IP laptop 1> --ip <IP laptop 2> --clients 10 --rounds 20`. Script mở cổng riêng cho 2 IP đó, tự đóng khi xong, khi lỗi, hoặc sau 3 giờ. Không muốn lấy IP thì thay các `--ip` bằng `--public` và để rule Oracle là `0.0.0.0/0`: trong lúc đo mọi IP đều nối được, vì máy chủ Flower không có mật khẩu.
   Mở sẵn `https://datn.tranngochung.id.vn/#flower` để chiếu: trang tự cập nhật, không cần tải lại.
2. **Laptop 1:** mở PowerShell trong thư mục dự án, gõ `deploy\lan_client\CHAY_CLIENT.bat 0 5 149.118.51.118`. Ảnh lấy từ `tmp\net10`.
3. **Laptop 2:** giải nén `LAPTOP_2_10_CLIENT.zip` (có sẵn Python, không cần cài), nhấp đúp `LAPTOP_2_BAM_DAY.bat`. Đừng nhấp đúp `CHAY_CLIENT.bat`, file đó không có tham số thì chỉ mở client 1.
4. Mỗi laptop mở 5 cửa sổ client. Máy chủ chờ đủ 10 rồi mới học. Cửa sổ chính báo `XONG` khi 5 client của máy đó học hết 20 vòng.
5. Kết quả nằm ở `artifacts/network_live/<giờ>_10client/` của bản web trên VM. Trang `#flower` và lệnh `python scripts/demo_show.py flower` (chạy trên VM) cùng đọc thư mục này bằng cùng một hàm tính, nên số trên web và terminal trùng nhau.

**Trên trang `#flower`:**

- Dải trên cùng: đang chờ mấy client, đang học vòng mấy, hoặc đã xong, kèm thanh tiến độ. Nếu máy chủ chết thì dải báo mất liên lạc sau khoảng 90 giây; nếu một client rớt thì báo lỗi ở vòng đó.
- Khung client: mỗi ô là một client, xếp theo máy. Viền cam là đang học, nền xanh là đã gửi trọng số về.
- 6 ô bằng chứng: mã băm khớp, độ biến thiên khớp, vòng đủ client, số client chung máy chủ (phải là 0), số MiB đã truyền, độ chính xác do máy chủ tính.
- 2 biểu đồ: độ chính xác và loss theo vòng; thời gian mỗi vòng dùng vào đâu.
- Bấm vào một client, một máy, máy chủ hoặc một ô bằng chứng: hiện ngay bên dưới cách làm và terminal chứa lệnh kèm kết quả. Lệnh đó gõ trên VM ra đúng các dòng này, ví dụ `python scripts/demo_show.py flower --detail client:7 --round 20`. Bấm lại hoặc nhấn Esc để đóng. Phần client là những gì máy chủ nhận và đo được; muốn cho xem màn hình client thì mở cửa sổ thật trên laptop.
- Bảng từng client, có thanh kéo để xem lại bất kỳ vòng nào. Không chạy trực tiếp được thì vẫn dùng thanh kéo này để trình bày lần đo đã lưu.

So sánh phân tán và tập trung, kiểm trọng số, các lần đo tốc độ cũ: chương 4, tab **Phân tán và tập trung**. Số lấy từ cùng tệp với mục 9 trên terminal.

**Đọc kết quả trên terminal:** mỗi vòng có độ chính xác do máy chủ tính, mỗi client có dòng "học (máy chủ ước lượng)", "phản hồi", độ trễ, tốc độ lên xuống, dòng "độ biến thiên ... KHỚP" và dòng băm KHỚP.

**Nói rõ khi trả lời thầy:**
- Ảnh là PlantVillage thật, mỗi client 380 ảnh, mô hình khởi tạo từ ImageNet và chỉ học trong ít vòng, nên độ chính xác ở phép đo này **chỉ chứng minh hệ thống chạy đúng**. Kết quả độ chính xác chính vẫn là 18 run 40 vòng.
- "Thời gian học" là **ước lượng của máy chủ** (thời gian phản hồi trừ thời gian đo mạng và thời gian truyền dự đoán), vì client không còn gửi thời gian của nó.
- Trọng số đi qua mạng không mã hóa; vì vậy cổng chỉ mở cho IP của 2 laptop trong lúc đo. Không có ảnh nào rời khỏi client.

## 6. Câu hỏi khó

| Thầy hỏi | Trả lời |
|---|---|
| File Word ghi SGD lr 0,01, sao em dùng 0,02? | "Em dùng batch 128 nên chọn lr 0,02 qua chạy thử, và dùng chung cho cả tập trung lẫn client nên so sánh vẫn công bằng." |
| Seed sao không phải 42, 43, 44? | "File Word ghi đó là ví dụ. Em dùng 42, 123, 2026, vẫn là 3 seed." |
| Dòng float32 trong báo cáo trùng dự đoán chỉ 76,47%? | "Em làm đối chứng chỉ đổi thứ tự ảnh, về toán phải trùng mà float32 vẫn lệch cùng cỡ. Nên không quy cho phân tán. Nguyên nhân cụ thể em chưa chứng minh được." |
| Flower có tương đương bản tự cài không? | "Trên toàn bộ ảnh chênh 0,11 điểm, nhưng ảnh ngoài đồng chênh 0,96 điểm, lớn hơn dao động giữa các lần chạy. Nên em chưa kết luận tương đương trên mọi nhóm." |
| 5 client có phải 5 máy thật không? | "Không, 5 client là mô phỏng trên một GPU. Máy thật là phần đo 2 laptop." |
| Tab 4.1 ghi 96,71%, tab 4.7 ghi 96,85%, số nào đúng? | "Cả hai đúng, khác tập đo. Tab 4.1 là một hạt giống trên tập kiểm tra. Tab 4.7 là trung bình 3 hạt giống trên tập thử riêng, em dùng tab 4.7 để so sánh." |
| Seed là gì? | "Số cố định phần ngẫu nhiên. Em chạy 3 seed để xem kết quả có do may rủi không, dấu ± là độ dao động." |
| Client gửi những gì? | "Trọng số, số ảnh, độ biến thiên và các mã để máy chủ đối chiếu. Không gửi ảnh, độ chính xác, loss, thời gian." |
| Kiểm tra này chống được giả mạo không? | "Không. Nó bắt được trọng số bị đổi trên đường truyền hoặc client khai sai. Ai sửa cả trọng số lẫn mã băm thì không bắt được, vì máy chủ đo chưa có mã hóa." |
| Số này sao khác báo cáo Đợt 2? | "Đó là hai bộ chạy riêng, khác seed và tập đánh giá. Tụi em không trộn hai bộ số." |
| Trên PlantDoc thì sao? | "172 ảnh: tập trung 57,56%, phân tán lệch mạnh 40,50%. Tập nhỏ nên phải đọc kèm khoảng tin cậy, và cả tập trung cũng yếu trên ảnh ngoài đồng." |
| Có tiết kiệm truyền thông không? | "Không. 40 vòng gửi 2.510 MB, gấp 1,65 lần gửi thẳng ảnh. Lợi ích chính là ảnh không rời máy." |
| Sao laptop 2 chậm vậy? | "Chạy 5 client thì học chậm hơn laptop 1 khoảng 9 lần, nguyên nhân chưa kiểm tra. Nên demo trực tiếp em giảm còn 2 client." |
| Phân tán có đảm bảo riêng tư tuyệt đối không? | "Không. Ảnh không rời máy, nhưng em chưa có DP hay mã hóa nên chưa có bảo đảm toán học." |

## 7. Lỗi thường gặp

Cửa sổ `.bat` hiện `?` thay chữ có dấu: chỉ do phông chữ, vẫn chạy đúng. Muốn đẹp thì chuột phải thanh tiêu đề cửa sổ, Properties, Font, chọn Consolas.

| Hiện tượng | Xử lý |
|---|---|
| Mục A → 5 báo cổng bận | Đóng các cửa sổ cũ, chạy lại |
| Máy chậm hoặc treo | Đóng Chrome, game. Cần khoảng 3 GB RAM trống |
| Chữ đỏ của Ray khi kết thúc phím T | Không phải lỗi nếu cuối cùng có dòng `xong` |
| Bấm mã không có trong menu | Menu tự hiện lại, gõ lại mã |

---

# Phụ lục: lệnh gõ tay

Mở PowerShell tại `D:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old`. Mọi lệnh dưới đây đã chạy thử trên laptop này, trừ Cách A của mục 8 (lệnh đó cần máy thuê còn bật, xem chú thích ở mục 8). Số trong "Sẽ thấy" là số của lần chạy thử; số của bạn có thể khác chút ở thời gian, các kết quả KHỚP thì phải giống. Mỗi cửa sổ PowerShell mới phải vào đúng thư mục và chạy dòng này một lần (mỗi cửa sổ giữ riêng, cửa sổ mới phải chạy lại):

```powershell
cd D:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old; $env:PYTHONUTF8 = 1; $env:OMP_NUM_THREADS = 2
```

Đóng bớt Chrome và game: máy cần khoảng 3 GB RAM trống. Số báo cáo đầy đủ nằm ở `docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md`.

**Các lệnh này chỉ để trình diễn, số độ chính xác của chúng không dùng để báo cáo:** chạy trên CPU với rất ít ảnh, ít vòng. Số báo cáo lấy từ các run 40 vòng trên GPU trong `runs/v4_sweep` và `runs/v4b`.

## Mục 4 và 5: Chạy Flower trên nhiều client, xem thông số

Cần 3 cửa sổ PowerShell, chạy theo đúng thứ tự, mỗi vòng khoảng 15–20 giây.

**Cửa sổ 1, máy chủ.** Chờ dòng `Starting Flower Server` rồi mới mở client:

```powershell
.\.venv\Scripts\python.exe -m src.federated.server_app --mode deployment --host 127.0.0.1 --port 18089 --rounds 3 --num-clients 2 --local-epochs 1 --seed 42 --pretrained-init --require-real-clients --reject-suspicious-updates --max-update-norm 500 --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 --metrics-file tmp/demo/history.json --checkpoint-file tmp/demo/global.pt
```

**Cửa sổ 2, client 0:**

```powershell
.\.venv\Scripts\python.exe -m src.federated.client_app --client-id 0 --server 127.0.0.1:18089 --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 1 --seed 42
```

**Cửa sổ 3, client 1:**

```powershell
.\.venv\Scripts\python.exe -m src.federated.client_app --client-id 1 --server 127.0.0.1:18089 --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_01 --require-real-data --local-epochs 1 --seed 42
```

`--pretrained-init` cho mô hình bắt đầu từ trọng số ImageNet để loss giảm thấy rõ sau vài vòng (đã thử: 3,27, 2,82, 2,30; vòng cuối dao động nhẹ giữa các lần chạy); chỉ dùng cho trình diễn, được ghi vào nhật ký. Thông số Flower hiện ngay ở cửa sổ 1 khi kết thúc: loss từng vòng, `round_fit_elapsed_ms`, `mean_probe_rtt_ms`, `rejected_clients`... Xem cho dễ đọc:

```powershell
.\.venv\Scripts\python.exe scripts/demo_show.py flower --history tmp/demo/history.json
```

Sẽ thấy:

```text
Vòng 1: 2 client được nhận, 0 bị từ chối; cả vòng 18.59 giây; gộp 51.1 ms; truyền ước tính 0.21 giây
  client 0: 380 ảnh, học 18.13 giây, RTT 0.1 ms, lên 1515.17 MiB/s, xuống 215.37 MiB/s, độ lệch cập nhật 0.7705
            băm client gửi 82350c8c8a6a / máy chủ tính lại 82350c8c8a6a -> KHỚP
```

**Nói khi demo:** đây là Flower thật qua gRPC, mỗi client là một tiến trình riêng, học trên ảnh của mình. Đây là 2 client trên một máy; mục 8 mới là hai máy thật.

Thông số của run thật 5 client, 40 vòng trên GPU:

```powershell
.\.venv\Scripts\python.exe scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 20
```

## Mục 1: Gửi trọng số có sai lệch không

**Cách kiểm:** client tính SHA-256 của trọng số ngay trước khi gửi, máy chủ tính lại ngay sau khi nhận rồi so. SHA-256 đổi hoàn toàn nếu chỉ một bit khác, nên khớp nghĩa là không sai bit nào.

Xem kết quả: các dòng `KHỚP` của lệnh `demo_show flower` ở trên, và dòng cuối của lệnh `demo_show run`:

```text
Băm hai chiều: 400/400 lượt khớp.
```

**Chứng minh phép kiểm thật sự bắt được lỗi**, tự đổi một bit thì phải báo sai:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_flower_app.py -k changed_weights -v
```

Sẽ thấy: `PASSED`. Test này đổi trọng số từ `2.0` thành `2.0000002` và yêu cầu run phải dừng ở cả hai chiều.

## Mục 2: Có sai lệch so với tập trung không

**Bước 1, hai bên xuất phát cùng một mô hình (W0):**

```powershell
.\.venv\Scripts\python.exe scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower
```

Sẽ thấy: `-> KHỚP`.

**Bước 2, chứng minh phép gộp không gây sai lệch** (khoảng 1 phút): cho mỗi client học đúng 1 bước rồi gộp, về toán học phải trùng học tập trung. Nếu truyền hoặc gộp có lỗi thì sẽ lệch.

```powershell
.\.venv\Scripts\python.exe scripts/verify_fedsgd_equivalence.py --quick --device cpu
```

Sẽ thấy:

```text
A_fedsgd label_alpha_0_1 float64 K=1: lệch L2 5.892e-14 ...      <- 1 bước: trùng, chỉ còn sai số làm tròn
B_local_steps label_alpha_0_1 float64 K=5: lệch L2 1.318e-01 ... <- 5 bước cục bộ: lệch thật
```

**Nói khi demo:** gửi và gộp không làm lệch. Lệch xuất hiện khi client học nhiều bước trên dữ liệu riêng, đó là bản chất của học phân tán.

**Bước 3, so trọng số hai mô hình đã train xong:**

```powershell
.\.venv\Scripts\python.exe scripts/demo_show.py compare --a runs/v4b/centralized_s42 --b runs/v4b/fedavg_s42
```

Sẽ thấy: L2 tham số 29,17; 94,72% dự đoán trùng trên 13.101 ảnh. **Đây là khác biệt do cách học, không phải lỗi truyền**, vì mục 1 đã chứng minh truyền đúng.

## Mục 3: Trình bày phương pháp kiểm tra

| Kiểm cái gì | Cách làm | Lệnh | Đạt khi |
|---|---|---|---|
| Truyền trọng số | SHA-256 hai đầu, mọi client, mọi vòng | `demo_show run` | khớp hết |
| Phép gộp | Tính lại trung bình theo n_k bằng NumPy, độc lập với mã gộp | `demo_show run`, cột sai số gộp | chỉ còn mức làm tròn |
| Xuất phát giống nhau | SHA-256 của W0 | `demo_show w0` | KHỚP |
| Công thức | FedSGD một bước so với tập trung | `verify_fedsgd_equivalence` | lệch dưới 1e-10 |
| Phép kiểm có nhạy không | Cố tình đổi một bit | `pytest ... -k changed_weights` | test đạt |
| Sau khi train | Khoảng cách trọng số, tỉ lệ dự đoán trùng | `demo_show compare` | đọc số, không có ngưỡng đạt |

## Mục 6: Huấn luyện tổng so với huấn luyện client

```powershell
.\.venv\Scripts\python.exe scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 20
```

Sẽ thấy:

```text
vòng  21: lr 0.0104, val top-1 90.17%, 14.8 giây, sai số gộp 7.45e-09, ...
    client 0: loss 0.821, 103 bước, học 2.9 giây, đã đi 1.048, cách mô hình chung 0.966
```

Cách đọc:
- **Sai số gộp:** sai số của phép tính gộp, chỉ ở mức làm tròn.
- **Đã đi:** client đi xa bao nhiêu so với mô hình nhận đầu vòng.
- **Cách mô hình chung:** sau khi gộp, client vẫn cách mô hình chung bao nhiêu. Đây là độ trôi của client, là khác biệt thật giữa mô hình client và mô hình chung, không phải lỗi.

## Mục 7: Code backend, một phần tập trung, một phần phân tán

**Chạy thật hai đường cùng một lệnh train, chỉ khác cờ** (khoảng 1,5 phút):

```powershell
.\.venv\Scripts\python.exe scripts/train_v4.py --method centralized --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs
.\.venv\Scripts\python.exe scripts/train_v4.py --method fedavg --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag flower
```

Chạy lại phải xóa `tmp/demo/runs` trước, nếu không lệnh sẽ báo đã xong và bỏ qua.

**Mở code để show:**

| Phần | Vị trí | Việc |
|---|---|---|
| Tập trung | `scripts/train_v4.py`, nhánh `else` trong `training_loop` | học trên toàn bộ ảnh bằng `train_epoch` |
| Phân tán, máy chủ | `src/federated/flower_app.py`: `make_round_fn` | gửi trọng số xuống, nhận về, kiểm băm |
| Phân tán, client | `src/federated/flower_app.py`: `client_step` | nhận trọng số, học, gửi lên |
| Một client học | `src/training/client_round.py`: `train_one_client` | gọi đúng `train_epoch` của nhánh tập trung |
| Gộp | `src/federated/strategy.py`: `aggregate_fedavg` | trung bình có trọng số theo số ảnh |
| Flower qua mạng (mục 4, 8) | `src/federated/server_app.py`, `src/federated/client_app.py` | máy chủ và client gRPC |

**Nói khi demo:** hai đường dùng chung một hàm học `train_epoch`, nên khác biệt chỉ nằm ở cách dùng dữ liệu và cách trao đổi trọng số. Đoạn code đã trích sẵn ở mục 7 của `docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md`.

## Mục 8: Đo tốc độ giữa 2 máy khác nhau

Đo được bằng cách cho server gửi một khối byte đúng cỡ mô hình (`--probe-payload-bytes 6275944`) mỗi vòng: RTT là thời gian khứ hồi, tốc độ lên và xuống tính từ thời gian gửi khối đó. Thời gian học do chính mỗi máy tự đo.

**Cách A, máy thuê còn bật:** laptop là client 0, máy thuê là server và client 1. Đo khi máy thuê đã train xong, vì GPU bận làm sai số thời gian học. Lệnh này ghi đè kết quả cũ trên máy thuê; bản đã lưu ở laptop không ảnh hưởng. Các bước này chưa được chạy lại trong lúc viết hướng dẫn: đang dùng lại đúng cấu hình của lần đo ngày 02/11.

1. Cửa sổ 1, mở đường hầm, giữ nguyên cửa sổ này (mật khẩu máy thuê):
   ```powershell
   ssh -N -p 3000 -L 18089:127.0.0.1:18089 -L 18090:127.0.0.1:18090 root@n2.ckey.vn
   ```
2. Cửa sổ 2: `ssh -p 3000 root@n2.ckey.vn`, rồi `bash ~/cropfed/speedtest_server.sh`.
3. Cửa sổ 3: `ssh -p 3000 root@n2.ckey.vn`, rồi `bash ~/cropfed/speedtest_client1.sh`.
4. Cửa sổ 4, trên laptop:
   ```powershell
   .\.venv\Scripts\python.exe -m src.federated.client_app --client-id 0 --server 127.0.0.1:18089 --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 2 --seed 42
   ```
5. Xong thì kéo kết quả về và xem:
   ```powershell
   scp -P 3000 root@n2.ckey.vn:cropfed/speedtest/out/wan_history.json tmp/demo/wan_history.json
   .\.venv\Scripts\python.exe scripts/demo_show.py flower --history tmp/demo/wan_history.json
   ```

**Cách B, không còn máy thuê:** xem lại kết quả đã đo thật ngày 02/11:

```powershell
.\.venv\Scripts\python.exe scripts/demo_show.py flower --history artifacts/network_speedtest/20261102T143913Z_wan_may_thue/wan_history.json
```

Kết quả đã đo, 3 vòng:

| Máy | Học 1 vòng | Ảnh/giây | RTT | Lên | Xuống |
|---|---:|---:|---:|---:|---:|
| Laptop i5-12500H CPU | 18,0 giây | 42 | 1.425 ms | 2,5 MiB/s | 0,45 MiB/s |
| Máy thuê RTX 3090 | 0,9 giây | 845 | 0,1 ms | không phải đường truyền thật | không phải đường truyền thật |

**Nói rõ khi demo:** kết nối đi qua SSH qua relay nên RTT cao và không đại diện cho mạng thường; client trên máy thuê chạy cùng máy với server. Muốn đo hai máy cùng mạng LAN: `docs/HUONG_DAN_DO_TOC_DO_LAN.md`.

## Mục 9: Khác biệt giữa phân tán và tập trung

Bảng đầy đủ ở mục 9 của `docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md`, nguồn `artifacts/sweep_v4/table_main.csv`.

| | Tập trung | Phân tán |
|---|---|---|
| Dữ liệu | Gom toàn bộ ảnh về một nơi | Ảnh ở lại client, chỉ gửi trọng số |
| Bước gộp | Không có | Máy chủ gộp theo số ảnh mỗi vòng |
| Top-1 toàn tập thử (3 seed) | 97,69% | 96,80% khi đồng nhất; 94,03% khi lệch mạnh (α = 0,1) |
| Ảnh ngoài đồng | 92,23% | 90,09% khi đồng nhất; 84,08% khi lệch mạnh |
| Truyền | 1.524 MB ảnh, một lần | khoảng 2.523 MB trọng số, hai chiều, 40 vòng |

**Khoảng cách FedAvg đến Tập trung theo mức lệch** (điểm %, 3 seed; biểu đồ `artifacts/sweep_v4/gap_by_alpha.png` và `curves_by_alpha.png`):

| α | Toàn tập thử | PlantVillage | Ngoài đồng | PlantDoc |
|---:|---:|---:|---:|---:|
| 100 (gần đồng nhất) | 0,73 | 0,46 | 2,00 | 2,13 |
| 10 | 0,74 | 0,47 | 2,05 | 1,94 |
| 1 | 1,03 | 0,64 | 2,96 | 2,71 |
| 0,5 | 1,36 | 0,93 | 3,32 | 4,65 |
| 0,1 (lệch mạnh) | 3,77 | 2,53 | 9,11 | 17,05 |

**Ý chính:** gửi và gộp trọng số không gây sai lệch (mục 1, 2). Khoảng cách với tập trung rất nhỏ khi dữ liệu gần đồng nhất và tăng nhanh khi lệch mạnh; PlantDoc chỉ 172 ảnh nên dao động lớn. Trên biểu đồ, điểm IID là run Flower, các điểm α là run một tiến trình; chưa có Local-only ở các α khác 0,1. Phân tán không tiết kiệm băng thông trong cấu hình này; lợi ích là dữ liệu không rời client.

## Nếu lỗi khi demo

| Hiện tượng | Xử lý |
|---|---|
| `The expression after '&' ... not valid` | Lệnh bản cũ dùng biến `$py` chưa đặt; dùng lệnh `.\.venv\Scripts\python.exe` trong hướng dẫn này |
| Client báo không nối được | Chưa thấy `Starting Flower Server`, hoặc cổng đang bị chiếm: đóng hết cửa sổ cũ rồi chạy lại từ máy chủ |
| `tmp/speedtest` không có | `.\.venv\Scripts\python.exe -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10` |
| Lệnh train báo "đã xong, bỏ qua" | Xóa `tmp/demo/runs` |
| Hết RAM, máy treo | Đóng Chrome, game; 2 client cần khoảng 3 GB |
