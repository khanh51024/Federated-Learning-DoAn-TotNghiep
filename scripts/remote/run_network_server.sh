#!/usr/bin/env bash
# Máy chủ Flower TÁCH RIÊNG cho phép đo nhiều client qua Internet (chạy trên VM, KHÔNG có client nào trên VM).
#
# Cách dùng (trên VM):
#   bash scripts/remote/run_network_server.sh --ip 1.2.3.4 --ip 5.6.7.8 [--clients 10] [--min-clients 10] [--rounds 20] [--data ~/net10-data]
#   bash scripts/remote/run_network_server.sh --public ...   (mở cho mọi IP trong lúc đo: không cần biết IP laptop, nhưng máy chủ Flower không có mật khẩu nên ai dò ra cổng cũng nối được)
#   bash scripts/remote/run_network_server.sh --public --light --clients 5 --rounds 10   (bộ ảnh nhẹ ở ~/mini10-data, chỉ để thấy mô hình học; lần đo có đuôi _nhe và ghi chú trên trang)
#
# - Dữ liệu (partition/ + images_eval/) lấy từ tmp/net10 của máy dev (scripts/prepare_network_speedtest.py --clients 10 ...), chép lên --data.
#   Máy chủ chỉ giữ ảnh kiểm tra, KHÔNG giữ ảnh của client nào.
# - Mở cổng Flower và cổng đo mạng (cổng + 1) cho các IP khai bằng --ip, hoặc mọi IP nếu có --public, và tự gỡ khi xong, lỗi, hoặc sau --max-hours (mặc định 3).
#   Quy tắc Oracle Security List (ngoài VM) phải do chủ tài khoản thêm và xóa tay.
# - Không đụng dịch vụ web đang chạy: dùng chung mã/venv của bản đang phục vụ nhưng chạy tiến trình riêng, ưu tiên thấp.
# - Kết quả ghi vào artifacts/network_live/<giờ UTC> của bản web đó: history.json, global.pt, progress.json, server.log. Trang #flower và
#   scripts/demo_show.py cùng đọc thư mục này (src/federated/run_view.py), nên web tự cập nhật theo từng vòng trong lúc chạy.
set -euo pipefail

RELEASE="${RELEASE:-$(systemctl show cropfed-demo -p WorkingDirectory --value)}"   # bản web đang phục vụ: chỉ đọc mã và .venv, không đổi gì của dịch vụ
DATA="$HOME/net10-data"
OUT=""
PORT=18089
CLIENTS=10
MIN_CLIENTS=""
ROUNDS=20
MAX_HOURS=3
IPS=()
PUBLIC=0
LIGHT=0
LIGHT_NOTE="Bộ ảnh nhẹ: chỉ lấy 10 trong 38 loại bệnh, rất ít ảnh, để thấy mô hình gộp học được qua từng vòng. Không dùng số này làm kết quả của đề tài."

while [ $# -gt 0 ]; do
  case "$1" in
    --ip) IPS+=("$2"); shift 2 ;;
    --public) PUBLIC=1; shift ;;
    --light) LIGHT=1; DATA="$HOME/mini10-data"; shift ;;
    --clients) CLIENTS="$2"; shift 2 ;;
    --min-clients) MIN_CLIENTS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    *) echo "Tham số lạ: $1" >&2; exit 2 ;;
  esac
done
MIN_CLIENTS="${MIN_CLIENTS:-$CLIENTS}"
SUFFIX=""
[ "$LIGHT" = 0 ] || SUFFIX="_nhe"
OUT="${OUT:-$RELEASE/artifacts/network_live/$(date -u +%Y%m%dT%H%M%SZ)_${CLIENTS}client$SUFFIX}"

[ "$PUBLIC" = 0 ] || [ ${#IPS[@]} -eq 0 ] || { echo "Chỉ dùng một trong hai: --public hoặc --ip." >&2; exit 2; }
[ "$PUBLIC" = 1 ] || [ ${#IPS[@]} -gt 0 ] || { echo "Cần --ip (IP công khai của laptop) hoặc --public (mở cho mọi IP trong lúc đo)." >&2; exit 2; }
for ip in "${IPS[@]:-}"; do
  [ "$PUBLIC" = 1 ] || [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "IP không hợp lệ: $ip" >&2; exit 2; }
done
[ "$PUBLIC" = 0 ] || IPS=("0.0.0.0/0")
[ -d "$DATA/partition" ] && [ -d "$DATA/images_eval" ] || { echo "Thiếu $DATA/partition hoặc $DATA/images_eval" >&2; exit 2; }
[ -x "$RELEASE/.venv/bin/python" ] || { echo "Không thấy $RELEASE/.venv/bin/python" >&2; exit 2; }
if ss -ltn | awk '{print $4}' | grep -qE ":($PORT|$((PORT + 1)))$"; then
  echo "Cổng $PORT hoặc $((PORT + 1)) đang bận: còn máy chủ Flower cũ?" >&2; exit 2
fi
mkdir -p "$OUT"
[ "$LIGHT" = 0 ] || printf '%s\n' "$LIGHT_NOTE" > "$OUT/note.txt"

RULES=()
cleanup() {
  for ip in "${RULES[@]:-}"; do
    [ -n "$ip" ] && sudo iptables -D INPUT -p tcp -s "$ip" --dport "$PORT:$((PORT + 1))" -j ACCEPT 2>/dev/null || true
  done
  echo "[server] đã gỡ quy tắc tường lửa cho: ${RULES[*]:-không có}"
}
trap cleanup EXIT
trap 'exit 143' INT TERM HUP    # dừng bằng Ctrl+C, kill, hoặc đóng cửa sổ SSH thì thoát và để trap EXIT dọn tường lửa đúng một lần

for ip in "${IPS[@]}"; do
  sudo iptables -I INPUT 1 -p tcp -s "$ip" --dport "$PORT:$((PORT + 1))" -j ACCEPT
  RULES+=("$ip")
done
echo "[server] đã mở cổng $PORT-$((PORT + 1)) cho: ${IPS[*]} (tự đóng khi xong hoặc sau ${MAX_HOURS} giờ)"
echo "[server] $CLIENTS client (tối thiểu $MIN_CLIENTS để bắt đầu), $ROUNDS vòng. Nhật ký: $OUT/server.log"

cd "$RELEASE"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 timeout --foreground "$((MAX_HOURS * 3600))" .venv/bin/python -m src.federated.server_app \
  --mode deployment --host 0.0.0.0 --port "$PORT" --rounds "$ROUNDS" \
  --num-clients "$CLIENTS" --min-clients "$MIN_CLIENTS" --local-epochs 1 --seed 42 --pretrained-init \
  --require-real-clients --reject-suspicious-updates --max-update-norm 500 \
  --evaluation-partition-dir "$DATA/partition" --evaluation-dataset-root "$DATA/images_eval" --require-real-evaluation \
  --probe-payload-bytes 6275944 \
  --metrics-file "$OUT/history.json" --checkpoint-file "$OUT/global.pt" --progress-file "$OUT/progress.json" 2>&1 | tee "$OUT/server.log"
# Đo xong thì chương trình riêng đọc lại tệp kết quả và tệp mô hình; web hiện kết quả này. Chưa đạt không làm script báo lỗi: kết quả nằm trong tệp.
.venv/bin/python -X utf8 scripts/validate_network_artifact.py --metrics "$OUT/history.json" --checkpoint "$OUT/global.pt" --expected-clients "$CLIENTS" --expected-rounds "$ROUNDS" --output "$OUT/validation.json" > /dev/null || echo "[server] trình kiểm báo chưa đạt, xem $OUT/validation.json"
