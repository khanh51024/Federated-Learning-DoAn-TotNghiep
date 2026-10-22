#!/usr/bin/env bash
# Chạy ở MÁY NÀY (Git Bash/Linux): định kỳ kéo kết quả từ máy GPU thuê về, để máy thuê có bị thu hồi giữa chừng cũng không mất run đã xong
# (đã xảy ra: máy thuê đầu tiên bị cấp cho người khác, mất run mốc và ablation chỉ nằm trên đó).
# Kéo: run đã xong (có DONE, trừ checkpoint_last.pt) chưa có ở máy này, cùng logs/, history.jsonl, config.json của mọi run (nhỏ, để theo dõi).
# Dùng: SSH_TARGET=root@host SSH_PORT=22 SSH_KEY=~/.ssh/key REMOTE_OUT=cropfed/runs/v4b LOCAL_OUT=runs/v4b bash scripts/remote/pull_runs.sh
# Biến tùy chọn: INTERVAL (giây, mặc định 600), EXPECTED (số run, mặc định 21; đủ thì dừng).
set -uo pipefail
: "${SSH_TARGET:?}" "${SSH_PORT:?}" "${SSH_KEY:?}" "${REMOTE_OUT:?}" "${LOCAL_OUT:?}"
INTERVAL="${INTERVAL:-600}"
EXPECTED="${EXPECTED:-21}"
# SSH_HOSTKEY_ALIAS: khi gọi máy bằng IP (tên miền của nơi thuê có lúc không phân giải được) vẫn kiểm khóa máy chủ theo mục đã lưu, ví dụ "[n2.ckey.vn]:1766"
SSH=(ssh -i "$SSH_KEY" -p "$SSH_PORT" -o IdentitiesOnly=yes -o PasswordAuthentication=no -o ConnectTimeout=20 -o ServerAliveInterval=30 ${SSH_HOSTKEY_ALIAS:+-o "HostKeyAlias=$SSH_HOSTKEY_ALIAS"} "$SSH_TARGET")
mkdir -p "$LOCAL_OUT"
while true; do
  # `exit 0` ở cuối: chưa có run nào xong thì vòng lặp trả mã 1, không được nhầm thành mất kết nối (ssh lỗi trả 255)
  if ! remote_done=$("${SSH[@]}" "cd '$REMOTE_OUT' 2>/dev/null && for d in */DONE; do [ -f \"\$d\" ] && echo \"\${d%/DONE}\"; done; exit 0"); then
    echo "$(date '+%F %T') không kết nối được máy GPU, thử lại sau 60 giây"
    sleep 60
    continue
  fi
  new=()
  for run in $remote_done; do
    [ -f "$LOCAL_OUT/$run/DONE" ] || new+=("$run")
  done
  "${SSH[@]}" "cd '$REMOTE_OUT' && tar -cf - --exclude=checkpoint_last.pt --exclude='*.tmp' ${new[*]} logs \$(ls -d */history.jsonl */config.json 2>/dev/null)" \
    | tar -xf - -C "$LOCAL_OUT"
  have=$(find "$LOCAL_OUT" -mindepth 2 -maxdepth 2 -name DONE | wc -l)
  echo "$(date '+%F %T') kéo thêm ${#new[@]} run đã xong; ở máy này có $have/$EXPECTED run DONE"
  if [ "$have" -ge "$EXPECTED" ]; then
    echo "$(date '+%F %T') đủ $EXPECTED run, dừng"
    exit 0
  fi
  sleep "$INTERVAL"
done
