// Làm nổi HÀNG và CỘT của ô đang rê chuột (heatmap ở chương Dữ liệu, ma trận nhầm lẫn ở chương Kết quả) kèm tooltip nổi.
// Hai dải sáng + tooltip là phần tử tuyệt đối di chuyển bằng transform; không đụng style từng ô nên không layout/vẽ lại bảng.
// An toàn khi bảng được vẽ lại bằng innerHTML: overlay được tạo lười ở lần rê chuột kế tiếp.

const ATTACHED = "xhAttached";

export function attachCrosshair(wrap, { cell = "td.heat-cell, td.cm-cell" } = {}) {
  if (!wrap || wrap.dataset[ATTACHED]) return;
  wrap.dataset[ATTACHED] = "1";
  wrap.classList.add("has-crosshair");

  let rowBand = null;
  let colBand = null;
  let tip = null;
  let current = null;
  let litRow = null;
  let litCol = null;

  const ensure = () => {
    if (rowBand && rowBand.isConnected) return;
    rowBand = document.createElement("span");
    rowBand.className = "xh-band xh-row";
    colBand = document.createElement("span");
    colBand.className = "xh-band xh-col";
    tip = document.createElement("span");
    tip.className = "xh-tip";
    tip.setAttribute("aria-hidden", "true");
    wrap.append(rowBand, colBand, tip);
  };

  const clearHeads = () => {
    litRow?.classList.remove("xh-head");
    litCol?.classList.remove("xh-head");
    litRow = litCol = null;
  };

  const hide = () => {
    current = null;
    clearHeads();
    if (rowBand) { rowBand.style.opacity = "0"; colBand.style.opacity = "0"; tip.style.opacity = "0"; }
  };

  const show = td => {
    ensure();
    const table = td.closest("table");
    const tr = td.parentElement;
    if (!table || !tr) return;
    const tx = table.offsetLeft;
    const ty = table.offsetTop;
    const x = tx + td.offsetLeft;
    const y = ty + td.offsetTop;
    rowBand.style.transform = `translate3d(${tx}px, ${y}px, 0) scale(${table.offsetWidth}, ${td.offsetHeight})`;
    colBand.style.transform = `translate3d(${x}px, ${ty}px, 0) scale(${td.offsetWidth}, ${table.offsetHeight})`;
    rowBand.style.opacity = colBand.style.opacity = "1";

    clearHeads();
    litRow = tr.querySelector("th");
    const headRow = table.tHead?.rows?.[table.tHead.rows.length - 1];
    litCol = headRow?.cells?.[td.cellIndex] || null;
    litRow?.classList.add("xh-head");
    litCol?.classList.add("xh-head");

    // Ô khai báo chú thích bằng `title` (cảm ứng/trình đọc màn hình vẫn có); khi hiện tooltip riêng thì gỡ `title` để không bị hai tooltip.
    if (td.title) {
      td.dataset.tip = td.title;
      td.removeAttribute("title");
    }
    const text = td.dataset.tip;
    if (text) {
      tip.textContent = text;
      tip.style.opacity = "1";
      const left = wrap.scrollLeft;
      const width = wrap.clientWidth;
      const tipWidth = tip.offsetWidth;
      const tipHeight = tip.offsetHeight;
      let px = x + td.offsetWidth + 6;
      if (px + tipWidth > left + width - 4) px = Math.max(left + 4, x - tipWidth - 6);
      let py = y - tipHeight - 6;
      if (py < wrap.scrollTop + 4) py = y + td.offsetHeight + 6;
      tip.style.transform = `translate3d(${px}px, ${py}px, 0)`;
    } else {
      tip.style.opacity = "0";
    }
  };

  wrap.addEventListener("pointerover", event => {
    const td = event.target.closest && event.target.closest(cell);
    if (!td || !wrap.contains(td)) return;
    if (td === current) return;
    current = td;
    show(td);
  });
  // chuột: rời bảng là tắt. Cảm ứng: giữ chú thích của ô vừa chạm cho tới khi chạm ra ngoài bảng.
  wrap.addEventListener("pointerleave", event => { if (event.pointerType !== "touch") hide(); });
  document.addEventListener("pointerdown", event => { if (current && !wrap.contains(event.target)) hide(); }, { passive: true });
}
