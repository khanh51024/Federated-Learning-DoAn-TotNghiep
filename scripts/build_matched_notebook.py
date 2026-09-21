"""Generate the small, explicit Kaggle entry point for matched FedAvg v5."""
import json
from pathlib import Path


def cell(kind, text, cell_id=None):
    value = {"cell_type": kind, "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    if cell_id:
        value["id"] = cell_id
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [cell("markdown", """
# GĐ2: Dirichlet FedAvg, đối chiếu bộ train GĐ1

K=5, AdamW 0.001, batch 32, R=10, E=1, ImageNet weights, checkpoint theo val accuracy.
Mốc label100/label1 được chạy lại trên cùng split sạch với label01.
Không ghép số ~99% lịch sử hoặc đối chứng v4 vào bảng matched.
Upload package mới kèm `data/partitions_stage2_matched_v5` và dataset ảnh.
Chạy `preflight` trước, chọn `run` khi đã kiểm tra quota. Xem docs/MATCHED_STAGE2_V5.md.
""", cell_id="intro-markdown"), cell("code", """
from pathlib import Path
import json, shutil, subprocess, sys, os

print("Contents of /kaggle/input:", [p.name for p in Path('/kaggle/input').iterdir()])

# 1. Tìm PACKAGE_SOURCE chứa stage2_matched
candidates = list(Path('/kaggle/input').glob('**/stage2_matched/experiment.py'))
assert len(candidates) >= 1, f"Không tìm thấy package chứa stage2_matched/experiment.py trong /kaggle/input"
PACKAGE_SOURCE = candidates[0].parent.parent
print(f"PACKAGE_SOURCE: {PACKAGE_SOURCE}")

# 2. Tìm DATASET_ROOT chứa đủ 38 thư mục lớp của PlantVillage
def find_dataset_root():
    # Kiểm tra các đường dẫn quen thuộc trước
    candidate_datasets = [
        Path('/kaggle/input/plantvillage-raw-images/raw/color'),
        Path('/kaggle/input/plantvillage-raw-images/color'),
        Path('/kaggle/input/plantvillage-dataset/raw/color'),
        Path('/kaggle/input/plantvillage-dataset/color'),
        Path('/kaggle/input/plant-village/raw/color'),
        Path('/kaggle/input/plant-village/color'),
        *Path('/kaggle/input').glob('**/raw/color'),
        *Path('/kaggle/input').glob('**/color'),
    ]
    for d in candidate_datasets:
        if d.is_dir() and len([p for p in d.iterdir() if p.is_dir()]) == 38:
            return d
    # Tìm kiếm đệ quy bất kỳ thư mục nào có đúng 38 thư mục con của PlantVillage
    for d in Path('/kaggle/input').rglob('*'):
        if d.is_dir():
            try:
                subdirs = [p.name for p in d.iterdir() if p.is_dir()]
                if len(subdirs) == 38 and any('Apple' in s for s in subdirs) and any('Tomato' in s for s in subdirs):
                    return d
            except (PermissionError, OSError):
                continue
    return None

DATASET_ROOT = find_dataset_root()
assert DATASET_ROOT is not None, "Không tìm thấy thư mục PlantVillage chứa 38 lớp trong /kaggle/input"
print(f"DATASET_ROOT: {DATASET_ROOT}")

WORK = Path('/kaggle/working/matched_package')
OUTPUT = Path('/kaggle/working/stage2_matched_v5')
RESUME_SOURCE = None  # Path('/kaggle/input/previous-output/stage2_matched_v5')

# Các chế độ:
# 'preflight': Kiểm tra toàn bộ 54.305 ảnh, đối chiếu tính toàn vẹn và không rò rỉ dữ liệu
# 'smoke': Chạy thử nhanh 2 round trên subset nhỏ để kiểm tra GPU CUDA
# 'run': Chạy huấn luyện chính thức FedAvg (10 rounds/job)
# 'baseline': Chạy baseline Centralized (tập trung) hoặc Local-only (cục bộ)
# 'collect': Đọc và xuất bảng so sánh matched_comparison.json
ACTION = 'preflight'
CONDITIONS = ['label100', 'label1', 'label01']
SEEDS = [42]  # Sau đó 123 và 2026, cùng dữ liệu/cấu hình.
SESSION_MINUTES = 420  # Giới hạn phiên; không tự đọc quota tài khoản.
BASELINE_MODE = 'centralized'  # 'centralized', 'local-only', hoặc 'both'

assert (PACKAGE_SOURCE / 'stage2_matched/experiment.py').is_file()
assert DATASET_ROOT.is_dir()
assert ACTION in {'preflight', 'smoke', 'run', 'baseline', 'collect'}
""", cell_id="setup-paths"), cell("code", """
# Copy only code and frozen manifests. Reused WORK must match uploaded bytes.
folders = ['stage2_matched', 'stage1_compat', 'src', 'fl_training', 'data/partitions_stage2_matched_v5']
for folder in folders:
    source, target = PACKAGE_SOURCE / folder, WORK / folder
    assert source.is_dir(), source
    files = [p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    expected = {p.relative_to(source).as_posix() for p in files}
    if target.exists():
        actual = {p.relative_to(target).as_posix() for p in target.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        assert actual == expected, 'Stale WORK files: use a new WORK path'
    for p in files:
        dest = target / p.relative_to(source)
        if dest.exists():
            assert dest.read_bytes() == p.read_bytes(), f'Stale code/data: {dest}'
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
if RESUME_SOURCE is not None:
    if OUTPUT.exists():
        assert any(OUTPUT.iterdir()), 'Use a new OUTPUT path for restore'
    else:
        shutil.copytree(RESUME_SOURCE, OUTPUT)
os.chdir(WORK)
""", cell_id="copy-work"), cell("code", """
# Verify or install compatible PyTorch 2.6.x and torchvision 0.21.x
def check_versions():
    res = subprocess.run(
        [sys.executable, '-c', 'import torch, torchvision; print(torch.__version__.split("+")[0], torchvision.__version__.split("+")[0])'],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        return False, False, f"error: {res.stderr.strip()}"
    parts = res.stdout.strip().split()
    if len(parts) != 2:
        return False, False, f"unexpected: {res.stdout.strip()}"
    t_ver, v_ver = parts
    return t_ver.startswith('2.6.'), v_ver.startswith('0.21.'), f"torch={t_ver}, torchvision={v_ver}"

t_ok, v_ok, current_vers = check_versions()
print(f"Initial environment: {current_vers}")

if not (t_ok and v_ok):
    print("Installing PyTorch 2.6.x / torchvision 0.21.x with matching accelerator build...")
    has_gpu = False
    try:
        res = subprocess.run(['nvidia-smi'], capture_output=True)
        has_gpu = (res.returncode == 0)
    except Exception:
        has_gpu = False
    idx = 'https://download.pytorch.org/whl/cu124' if has_gpu else 'https://download.pytorch.org/whl/cpu'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                    'torch==2.6.0', 'torchvision==0.21.0', '--index-url', idx], check=True)
    t_ok, v_ok, current_vers = check_versions()
    print(f"Post-install environment: {current_vers}")

assert t_ok and v_ok, f"Incompatible versions after bootstrap: {current_vers}"

res = subprocess.run(
    [sys.executable, '-c', 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'],
    capture_output=True, text=True, check=True
)
print("Runtime info:", res.stdout.strip())
if ACTION in {'run', 'smoke', 'baseline'}:
    assert 'True' in res.stdout.split(), 'Bật GPU accelerator trước khi chạy train'

suite = json.loads((WORK / 'data/partitions_stage2_matched_v5/suite.json').read_text())
assert suite['smoke'] is False and suite['num_clients'] == 5
print({'counts': suite['counts'], 'conditions': CONDITIONS, 'seeds': SEEDS,
       'rounds': 10, 'local_epochs': 1, 'session_minutes': SESSION_MINUTES})
""", cell_id="bootstrap-runtime"), cell("code", """
if ACTION == 'preflight':
    command = [sys.executable, '-u', '-m', 'stage2_matched', 'preflight',
               '--dataset', str(DATASET_ROOT), '--output', str(OUTPUT)]
    subprocess.run(command, check=True)

elif ACTION == 'smoke':
    # Chạy smoke test 2 round trên subset nhỏ để kiểm tra GPU
    smoke_suite = WORK / 'data/matched_smoke'
    smoke_output = OUTPUT.parent / 'stage2_matched_smoke'
    if not smoke_suite.exists():
        print("Tạo smoke suite nhỏ (2 rounds, không pretrained, ~300 ảnh)...")
        # Tìm leaf-map.json
        leaf_candidates = list(Path('/kaggle/input').glob('**/leaf-map.json'))
        leaf_map = leaf_candidates[0] if leaf_candidates else (WORK / 'PlantVillage-Dataset/leaf-map.json')
        if leaf_map.exists():
            subprocess.run([sys.executable, '-u', '-m', 'stage2_matched', 'prepare',
                            '--smoke', '--suite', str(smoke_suite),
                            '--dataset', str(DATASET_ROOT), '--leaf-map', str(leaf_map)], check=True)
    if smoke_suite.exists():
        subprocess.run([sys.executable, '-u', '-m', 'stage2_matched', 'run',
                        '--suite', str(smoke_suite), '--dataset', str(DATASET_ROOT),
                        '--output', str(smoke_output),
                        '--conditions', 'label100', 'label01', '--seeds', '42',
                        '--device', 'cuda'], check=True)
    else:
        print("Chưa có leaf-map.json để tạo smoke suite; chuyển sang chạy preflight.")
        subprocess.run([sys.executable, '-u', '-m', 'stage2_matched', 'preflight',
                        '--dataset', str(DATASET_ROOT), '--output', str(OUTPUT)], check=True)

elif ACTION == 'run':
    command = [sys.executable, '-u', '-m', 'stage2_matched', 'run',
               '--dataset', str(DATASET_ROOT), '--output', str(OUTPUT),
               '--conditions', *CONDITIONS, '--seeds', *map(str, SEEDS),
               '--session-minutes', str(SESSION_MINUTES), '--device', 'cuda']
    subprocess.run(command, check=True)

elif ACTION == 'baseline':
    command = [sys.executable, '-u', '-m', 'stage2_matched', 'baseline',
               '--dataset', str(DATASET_ROOT), '--output', str(OUTPUT),
               '--mode', BASELINE_MODE,
               '--conditions', *CONDITIONS, '--seeds', *map(str, SEEDS),
               '--device', 'cuda']
    subprocess.run(command, check=True)

elif ACTION == 'collect':
    command = [sys.executable, '-u', '-m', 'stage2_matched', 'collect',
               '--output', str(OUTPUT)]
    subprocess.run(command, check=True)
""", cell_id="execute-action"), cell("markdown", """
## Tiếp tục và tổng hợp

1. Để chạy các trục non-IID còn lại, đặt `CONDITIONS` thành:
   `['quantity100', 'quantity01', 'label_quantity01', 'feature100', 'feature01']`.
2. Để chạy baseline đối chứng GĐ2:
   - Đặt `ACTION = 'baseline'`, `BASELINE_MODE = 'centralized'` để lấy trần hiệu năng (upper bound).
   - Đặt `ACTION = 'baseline'`, `BASELINE_MODE = 'local-only'` để lấy sàn hiệu năng (lower bound).
3. Sau khi hoàn thành các lượt train, đặt `ACTION = 'collect'` để tạo file `matched_comparison.json`.
4. Trước khi phiên kết thúc hãy lưu **toàn bộ thư mục OUTPUT** làm artifact.
""", cell_id="next-steps")]

root = Path(__file__).resolve().parents[1]
for item in cells:
    if item['cell_type'] == 'code':
        compile(''.join(item['source']), '<notebook-cell>', 'exec')
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}
(root / 'kaggle_stage2_matched_v5.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print('Notebook generated; Python cells compile successfully.')
