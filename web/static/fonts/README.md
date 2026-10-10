# Phông tự host

Giao diện web không phụ thuộc Google Fonts lúc chạy; các tệp ở đây được tải một lần từ Google Fonts (CSS API v2, subset `latin`, `latin-ext`, `vietnamese`, và `greek` cho JetBrains Mono) và đặt cạnh mã nguồn.

| Họ phông | Trọng số | Dùng cho | Giấy phép |
|---|---|---|---|
| Be Vietnam Pro | 400, 500, 600, 700 | Toàn bộ chữ giao diện (có bộ glyph tiếng Việt đầy đủ) | SIL Open Font License 1.1 |
| JetBrains Mono | 400–600 (biến thiên) | Số KPI, mã/hash, chip thiết bị | SIL Open Font License 1.1 |

- `fonts.css` chứa các khối `@font-face` (kèm `unicode-range`), được nạp bởi `web/static/index.html`.
- Trình duyệt chỉ tải subset nào có ký tự được dùng; trang tiếng Việt thường dùng `latin` + `vietnamese`.
- Máy chủ trả các tệp này với `Cache-Control: public, max-age=86400` (xem `web/app.py`), các tệp khác của `/static/` vẫn `no-store`.
- Thay phông: tải lại woff2 từ Google Fonts, cập nhật tên tệp trong `fonts.css`, và nhớ cập nhật `<link rel="preload">` ở `index.html`.
