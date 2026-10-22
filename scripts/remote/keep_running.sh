#!/usr/bin/env bash
# Chạy ở MÁY NÀY: canh máy GPU thuê, nếu máy tự khởi động lại giữa chừng (mất các phiên tmux) mà đợt chưa xong thì chạy lại
# scripts/remote/run_parallel.sh trên máy đó; run đã xong được bỏ qua, run dở chạy tiếp từ checkpoint_last.pt.
# (Đã gặp: máy thuê khởi động lại sau khoảng 13 phút, mọi tiến trình mất, ổ đĩa còn nguyên.)
# Dùng: SSH_TARGET=root@host SSH_PORT=... SSH_KEY=... RELEASE_SHA=<sha> RECIPE="..." REMOTE_OUT=/root/cropfed/runs/v4b bash scripts/remote/keep_running.sh
# Biến tùy chọn: SEEDS (mặc định "42 123 2026"), INTERVAL (giây, mặc định 120), WORK (mặc định /root/cropfed),
#   WORKERS (worker mỗi phiên khi chạy lại một phần hạt giống, mặc định 9: tránh một phiên chiếm cả CPU khi các phiên khác vẫn chạy).
set -uo pipefail
: "${SSH_TARGET:?}" "${SSH_PORT:?}" "${SSH_KEY:?}" "${RELEASE_SHA:?}" "${REMOTE_OUT:?}"
SEEDS="${SEEDS:-42 123 2026}"
INTERVAL="${INTERVAL:-120}"
WORK="${WORK:-/root/cropfed}"
read -r -a seed_list <<< "$SEEDS"
# SSH_HOSTKEY_ALIAS: khi gọi máy bằng IP (tên miền của nơi thuê có lúc không phân giải được) vẫn kiểm khóa máy chủ theo mục đã lưu, ví dụ "[n2.ckey.vn]:1766"
SSH=(ssh -i "$SSH_KEY" -p "$SSH_PORT" -o IdentitiesOnly=yes -o PasswordAuthentication=no -o ConnectTimeout=20 -o ServerAliveInterval=30 ${SSH_HOSTKEY_ALIAS:+-o "HostKeyAlias=$SSH_HOSTKEY_ALIAS"} "$SSH_TARGET")
while true; do
  missing=""
  if ! state=$("${SSH[@]}" "for s in $SEEDS; do if [ -f '$REMOTE_OUT'/FINISHED_seeds_\$s ]; then echo \"\$s xong\"; elif tmux has-session -t train_s\$s 2>/dev/null; then echo \"\$s chạy\"; else echo \"\$s mất\"; fi; done; cut -d' ' -f1 /proc/uptime"); then
    echo "$(date '+%F %T') không kết nối được máy GPU"
    sleep 60
    continue
  fi
  finished=$(grep -c " xong" <<< "$state")
  missing=$(grep " mất" <<< "$state" | cut -d' ' -f1 | tr '\n' ' ')
  if [ "$finished" -ge "${#seed_list[@]}" ]; then
    echo "$(date '+%F %T') đủ ${#seed_list[@]} hạt giống, dừng canh"
    exit 0
  fi
  if [ -n "${missing// /}" ]; then
    echo "$(date '+%F %T') hạt giống không có phiên chạy: $missing (uptime $(tail -n 1 <<< "$state") s) -> chạy lại"
    "${SSH[@]}" "cd '$WORK' && env RELEASE_SHA='$RELEASE_SHA' RECIPE='${RECIPE:-}' OUT_DIR='$REMOTE_OUT' SEEDS='$missing' WORKERS='${WORKERS:-9}' \
      bash code/scripts/remote/run_parallel.sh '$WORK' > /dev/null 2>&1 < /dev/null" \
      || echo "$(date '+%F %T') chạy lại thất bại, thử ở vòng sau"
  fi
  sleep "$INTERVAL"
done
