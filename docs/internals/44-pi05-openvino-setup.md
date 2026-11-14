# pi0.5 OpenVINO 비교 환경

Intel이 공개한 pi0.5 OpenVINO 공식 구현인 [robotics-ai-suite의 vla-pi0.5-openvino](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/vla-pi0.5-openvino)와 vAi를 같은 PC에서 비교하는 환경을 만든다. 공식 구현의 경로(LeRobot PyTorch → ONNX → OpenVINO IR)를 그대로 따르되, vAi와 같은 조건이 되도록 세 가지를 바꾼다.

- 학습용 forward 대신 실제 추론(prefill + denoise 10단계)을 내보낸다.
- 입력을 vAi 데모와 같은 카메라 3대, 같은 프롬프트 길이로 맞춘다.
- INT8 압축 없이 FP16 가중치로 저장하고, 32 GB 노트북에서 돌도록 가중치를 메모리에 한 벌만 둔다.

모델은 LeRobot의 pi0.5(`lerobot/pi05_base`, 3B)이고 OpenVINO GPU FP16으로 돌려 정확도와 시간을 잰다. 이 문서의 명령을 위에서부터 그대로 실행하면 된다.

- 1장: 요구 사항
- 2장: 설치
- 3장: 모델 만들기
- 4장: 검증과 측정
- 5장: 스크립트 전문

**한 줄 요약:** Python 3.12 venv(torch CPU + LeRobot 0.6.1) → `sample_actions`를 ONNX로 내보내기 → OpenVINO 2026.1로 FP16 IR 변환 → GPU에서 PyTorch FP32 기준 출력 확인 후 시간 측정.

---

## 1. 요구 사항

| 항목 | 기준 |
|---|---|
| OS | Windows 11, Git Bash (명령은 모두 Git Bash 기준) |
| RAM | 32 GB 이상. ONNX 내보내기 최대 사용량이 약 23 GB |
| 디스크 | 토큰 길이 하나당 약 19 GB (ONNX 13 GB + IR 6.1 GB). 두 길이를 모두 만들면 약 38 GB |
| GPU | Intel 내장 GPU, 최신 드라이버 |
| Python | 3.12 (`py -3.12`로 실행 가능해야 함). 3.13 이상은 torch·LeRobot 지원이 보장되지 않아 쓰지 않는다 |
| 체크포인트 | vAi가 쓰는 것과 같은 파일: `vAi/data/ckpts/pi05/lerobot/3B/model.safetensors` (HuggingFace `lerobot/pi05_base`, FP32, 14.5 GB) |
| 네트워크 | 설치 중 pip, HuggingFace(`lerobot/pi05_base`의 `config.json`만), OpenVINO 압축 파일 다운로드 |

## 2. 설치

설치 폴더는 바탕화면의 `pi05-openvino`로 한다. 아래에서 `$D`는 그 경로다.

```bash
D=~/Desktop/pi05-openvino
mkdir -p $D && cd $D
```

### 2.1 OpenVINO 2026.1

pip의 `openvino`가 아니라 Intel 배포 압축 파일을 쓴다. 이미 `C:\Program Files (x86)\Intel\openvino_2026.1.0`이 있고 `runtime/version.txt`가 `2026.1.0-21367-63e31528c62`로 시작하면 이 단계를 건너뛴다.

```bash
curl -L -o ov.zip "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/windows/openvino_toolkit_windows_2026.1.0.21367.63e31528c62_x86_64.zip"
unzip -q ov.zip -d /tmp/ov && rm ov.zip
# 관리자 권한이 필요하면 사용자에게 이 이동만 맡긴다.
mv /tmp/ov/openvino_toolkit_windows_* "/c/Program Files (x86)/Intel/openvino_2026.1.0"
cat "/c/Program Files (x86)/Intel/openvino_2026.1.0/runtime/version.txt"
```

### 2.2 Python venv

```bash
cd $D
py -3.12 -m venv venv
./venv/Scripts/python.exe -m pip install -q --upgrade pip
./venv/Scripts/python.exe -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
./venv/Scripts/python.exe -m pip install -q "lerobot[pi]==0.6.1" transformers==5.5.4 onnx==1.23.2 numpy==2.2.6 safetensors==0.8.0 psutil==7.2.2
./venv/Scripts/python.exe -m pip list 2>/dev/null | grep -iE "^torch |^lerobot|^transformers|^onnx |^numpy|^psutil"
```

- torch는 CPU 빌드면 된다. 내보내기와 기준 출력 계산만 PyTorch로 하고, GPU 실행은 OpenVINO가 한다.
- 버전은 위에 고정한 값을 그대로 쓴다. LeRobot의 `PI05Pytorch.sample_actions` 인자와 체크포인트 키 변환 함수(`_fix_pytorch_state_dict_keys`)가 버전마다 다르다.

### 2.3 환경 파일과 확인

5장의 `env.sh`, `peakmem.py`, `export_onnx.py`, `convert_ir.py`, `bench_ov.py`를 `$D`에 그대로 만든다. 그다음 OpenVINO가 venv에서 GPU를 보는지 확인한다.

```bash
cd $D && source env.sh
$PY -c "import openvino as ov; c = ov.Core(); print(ov.__version__, c.available_devices, c.get_property('GPU', 'FULL_DEVICE_NAME'))"
```

`2026.1.0-...`과 `['CPU', 'GPU', ...]`이 나와야 한다. `GPU`가 없으면 GPU 드라이버부터 갱신한다.

## 3. 모델 만들기

토큰 길이는 두 가지를 만든다.

| 이름 | 프롬프트 토큰 | 행렬곱 M (이미지 768 + 토큰) | 용도 |
|---|---|---|---|
| `pi05_onnx`, `pi05_ir_fp16` | 154 | 922 | vAi 데모와 같은 입력 |
| `pi05_onnx160`, `pi05_ir160_fp16` | 160 | 928 | 32의 배수. OpenVINO 행렬곱이 제 속도를 내는 길이. 추가 6토큰은 마스크로 가린다 |

154는 vAi 데모(`models/production/pi05/demo.cpp`)의 프롬프트 토큰 수다. 데모의 프롬프트나 상태 입력을 바꿨다면 데모가 출력하는 prompt 토큰 수로 바꾸고, 160 대신 `768 + 토큰`을 32의 배수로 올리는 길이를 쓴다.

```bash
cd $D && source env.sh
CKPT=~/Desktop/vAi/data/ckpts/pi05/lerobot/3B/model.safetensors
$PY -u peakmem.py $PY -u export_onnx.py --ckpt $CKPT --tokens 154 --out pi05_onnx/pi05.onnx    > export154.log 2>&1
$PY -u peakmem.py $PY -u export_onnx.py --ckpt $CKPT --tokens 160 --out pi05_onnx160/pi05.onnx > export160.log 2>&1
$PY -u peakmem.py $PY -u convert_ir.py --onnx pi05_onnx/pi05.onnx    --out pi05_ir_fp16/model.xml    > convert154.log 2>&1
$PY -u peakmem.py $PY -u convert_ir.py --onnx pi05_onnx160/pi05.onnx --out pi05_ir160_fp16/model.xml > convert160.log 2>&1
grep -E "copied|written|checker|peakmem" export154.log export160.log convert154.log convert160.log
```

통과 기준:

- `weights copied: 813 tensors, 0 parameters missing`
- `ONNX checker: valid`
- 모든 `[peakmem] exit 0`
- 시간: 내보내기 한 번 약 4분, 변환 한 번 약 30초
- 최대 메모리: 내보내기 21 ~ 23 GB, 변환 약 14 GB

다른 프로그램이 RAM을 많이 쓰고 있으면 내보내기가 페이지 파일로 넘어가 매우 느려진다. 시작 전에 닫는다.

## 4. 검증과 측정

### 4.1 정확도

`bench_ov.py`는 내보낼 때 저장한 PyTorch FP32 출력(`reference.pt`)과 GPU 출력을 먼저 비교한다.

```bash
cd $D && source env.sh
$PY -u bench_ov.py --model pi05_ir_fp16/model.xml    --ref pi05_onnx/reference.pt    --runs 3 2>&1 | grep -E "compiled|output vs|latency"
$PY -u bench_ov.py --model pi05_ir160_fp16/model.xml --ref pi05_onnx160/reference.pt --runs 3 2>&1 | grep -E "compiled|output vs|latency"
```

통과 기준: `cosine` 0.9999 이상, `max abs / max |ref|` 0.01 이하. 정상 모델은 cosine 0.99998, max abs / max |ref| 0.006 정도다. cosine이 0.97 근처면 `export_onnx.py`의 `cumsum` 보정이 빠진 것이다(5.3절 주석).

### 4.2 시간

추론 한 번 = prefill + denoise 10단계, 카메라 3대 224 × 224, 배치 1.

```bash
$PY -u bench_ov.py --model pi05_ir160_fp16/model.xml --ref pi05_onnx160/reference.pt --runs 20 --preheat 60 2>&1 | grep latency
```

- 결과는 20회 중앙값을 쓴다. 컴파일은 약 60초 걸린다.
- `--preheat 60`은 측정 전에 60초간 추론을 돌려 전력 상태를 지속 한도(PL1)로 내린다. 노트북은 처음 몇십 초 동안 높은 한도(PL2)로 돌아서 그대로 재면 빠르게 나온다.
- vAi와 비교할 때는 같은 전력 상태에서 OpenVINO와 vAi 데모를 번갈아 세 바퀴 이상 재고, 바퀴별 중앙값을 나란히 적는다.
- 기준값 (Core Ultra Meteor Lake, Xe-LPG 8코어, PL1): 160토큰 약 2.3초, 154토큰 약 20초. 154토큰이 느린 것은 OpenVINO 행렬곱이 M이 32의 배수가 아닐 때 느린 경로로 가기 때문이다. 비교에는 160토큰 결과를 쓰고 154토큰은 참고로 적는다.

### 4.3 옵션

| 옵션 | 뜻 |
|---|---|
| `--device CPU` | CPU에서 실행 (정확도 확인용) |
| `--precision f32` | GPU 추론 정밀도를 f32로 (기본 f16) |
| `--throttle LOW\|MEDIUM\|HIGH` | `GPU_QUEUE_THROTTLE` |
| `--cool N` | 측정 직전 N초 쉬기 |

## 5. 스크립트 전문

### 5.1 env.sh

```bash
# Source from Git Bash: the venv's Python plus the OpenVINO 2026.1 already installed (no pip openvino).
OV='C:\Program Files (x86)\Intel\openvino_2026.1.0'
export OPENVINO_LIB_PATHS="$OV\\runtime\\bin\\intel64\\Release;$OV\\runtime\\3rdparty\\tbb\\bin"
export PYTHONPATH="$OV\\python"
PY="$(dirname "${BASH_SOURCE[0]}")/venv/Scripts/python.exe"
```

### 5.2 peakmem.py

```python
"""Runs a command and reports its peak working set (Windows), sampled every 0.5 s and from the OS counter.

  python peakmem.py <command ...>
"""
import subprocess, sys, time
import psutil

p = subprocess.Popen(sys.argv[1:])
proc = psutil.Process(p.pid); peak = 0
while p.poll() is None:
    try:   # the venv's python.exe is a launcher: the interpreter is its child
        for q in [proc] + proc.children(recursive=True):
            mi = q.memory_info(); peak = max(peak, getattr(mi, 'peak_wset', mi.rss))
    except psutil.Error:
        pass
    time.sleep(0.5)
print(f'[peakmem] exit {p.returncode}, peak working set {peak / 2**30:.1f} GB', flush=True)
sys.exit(p.returncode)
```

### 5.3 export_onnx.py

공식 구현의 `convert_pytorch_onnx.py`를 대신하며, 문서 첫머리의 세 가지 변경 중 내보내기 쪽을 담는다.

```python
"""pi0.5 inference (prefill + 10 denoise steps) to ONNX, in fp32, with one copy of the weights in memory.

Differences from Intel's convert_pytorch_onnx.py (robotics-ai-suite/vla-pi0.5-openvino):
  - exports PI05Pytorch.sample_actions, the inference, instead of the training forward
  - three cameras and a fixed prompt length (vAi's demo: 3 x 224 x 224, 154 tokens)
  - builds the model once and copies the checkpoint into it tensor by tensor (no second full state dict)

  python export_onnx.py --ckpt PATH [--tokens 154] [--steps 10] [--out pi05_onnx/pi05.onnx]
"""
import argparse, os, time
import torch
from safetensors import safe_open
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

# ONNX CumSum takes no bool, and the TorchScript exporter passes LeRobot's bool masks straight to it (OpenVINO then
# counts in bool, so every position id collapses). Count in int64, as PyTorch does for a bool cumsum.
_cumsum = torch.cumsum
def _cumsum_int(x, *args, **kwargs):
    return _cumsum(x.to(torch.int64) if x.dtype == torch.bool else x, *args, **kwargs)
torch.cumsum = _cumsum_int

ap = argparse.ArgumentParser()
ap.add_argument('--tokens', type=int, default=154)
ap.add_argument('--steps', type=int, default=10)
ap.add_argument('--ckpt', required=True)   # lerobot/pi05_base model.safetensors, as vAi loads it
ap.add_argument('--out', default='pi05_onnx/pi05.onnx')
a = ap.parse_args()
t0 = time.time()

cfg = PreTrainedConfig.from_pretrained('lerobot/pi05_base')   # config.json only
cfg.compile_model = False
cfg.device = 'cpu'
policy = PI05Policy(cfg)          # fp32, randomly initialised: the one copy of the weights
policy.eval()
print(f'model built ({time.time() - t0:.0f} s)', flush=True)

# Checkpoint keys onto the model's, then each tensor copied in place.
target = dict(policy.named_parameters()); target.update(dict(policy.named_buffers()))
loaded = set()
with safe_open(a.ckpt, framework='pt') as f:
    keys = list(f.keys())
    class Key(str):          # stands in for a tensor: the fix-up only renames, and clones one tied weight
        def clone(self): return self
    fixed = policy._fix_pytorch_state_dict_keys({k: Key(k) for k in keys}, cfg)   # values are the source keys
    for new_key, src in fixed.items():
        name = new_key if new_key.startswith('model.') else 'model.' + new_key
        if name not in target:
            print('  unexpected', name); continue
        with torch.no_grad():
            target[name].copy_(f.get_tensor(src))
        loaded.add(name)
missing = [n for n, _ in policy.named_parameters() if n not in loaded]
print(f'weights copied: {len(loaded)} tensors, {len(missing)} parameters missing ({time.time() - t0:.0f} s)', flush=True)
for n in missing[:10]: print('  missing', n)


class Inference(torch.nn.Module):
    def __init__(self, model, steps):
        super().__init__()
        self.model, self.steps = model, steps

    def forward(self, image0, image1, image2, tokens, token_masks, noise):
        m = torch.ones(image0.shape[0], dtype=torch.bool)
        return self.model.sample_actions([image0, image1, image2], [m, m, m], tokens, token_masks,
                                         noise=noise, num_steps=self.steps)


net = Inference(policy.model, a.steps).eval()
B = 1
images = [torch.rand(B, 3, 224, 224) * 2 - 1 for _ in range(3)]
tokens = torch.randint(0, 257152, (B, a.tokens), dtype=torch.long)
masks = torch.ones(B, a.tokens, dtype=torch.bool)
noise = torch.randn(B, cfg.chunk_size, cfg.max_action_dim)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
with torch.no_grad():
    ref = net(*images, tokens, masks, noise)
    torch.save({'inputs': (images, tokens, masks, noise), 'output': ref}, os.path.join(os.path.dirname(a.out), 'reference.pt'))
    print(f'reference run done, output {tuple(ref.shape)} ({time.time() - t0:.0f} s)', flush=True)
    torch.onnx.export(net, (*images, tokens, masks, noise), a.out,
                      input_names=['image0', 'image1', 'image2', 'tokens', 'token_masks', 'noise'],
                      output_names=['actions'], opset_version=18, dynamo=False, do_constant_folding=False)
print(f'ONNX written to {a.out} ({time.time() - t0:.0f} s)', flush=True)
import onnx
onnx.checker.check_model(a.out, full_check=True)   # types and shapes; the weights stay on disk
print('ONNX checker: valid', flush=True)
```

### 5.4 convert_ir.py

```python
"""ONNX to OpenVINO IR with fp16 weights (the ONNX weights are memory-mapped, not loaded).

  python convert_ir.py [--onnx pi05_onnx/pi05.onnx] [--out pi05_ir_fp16/model.xml]
"""
import argparse, time
import openvino as ov

ap = argparse.ArgumentParser()
ap.add_argument('--onnx', default='pi05_onnx/pi05.onnx')
ap.add_argument('--out', default='pi05_ir_fp16/model.xml')
a = ap.parse_args()
t0 = time.time()
model = ov.convert_model(a.onnx)
print(f'converted ({time.time() - t0:.0f} s)', flush=True)
ov.save_model(model, a.out, compress_to_fp16=True)
print(f'IR written to {a.out} ({time.time() - t0:.0f} s)', flush=True)
```

### 5.5 bench_ov.py

```python
"""pi0.5 IR on an OpenVINO device: output against the PyTorch reference, then latency.

  python bench_ov.py [--model pi05_ir_fp16/model.xml] [--device GPU] [--runs 20]
"""
import argparse, statistics as st, time
import numpy as np, torch
import openvino as ov

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='pi05_ir_fp16/model.xml')
ap.add_argument('--device', default='GPU')
ap.add_argument('--runs', type=int, default=20)
ap.add_argument('--ref', default='pi05_onnx/reference.pt')
ap.add_argument('--precision', default='')   # f32 to check accuracy
ap.add_argument('--throttle', default='')    # LOW / MEDIUM / HIGH: GPU_QUEUE_THROTTLE
ap.add_argument('--preheat', type=float, default=0)   # seconds of inferences before timing (PL1)
ap.add_argument('--cool', type=float, default=0)      # seconds idle before timing (PL2)
a = ap.parse_args()

ref = torch.load(a.ref)
images, tokens, masks, noise = ref['inputs']
feed = {'image0': images[0].numpy(), 'image1': images[1].numpy(), 'image2': images[2].numpy(),
        'tokens': tokens.numpy(), 'token_masks': masks.numpy(), 'noise': noise.numpy()}

core = ov.Core()
t0 = time.time()
cfg = {'PERFORMANCE_HINT': 'LATENCY'}
if a.precision: cfg['INFERENCE_PRECISION_HINT'] = a.precision
if a.throttle: cfg['GPU_QUEUE_THROTTLE'] = a.throttle
compiled = core.compile_model(a.model, a.device, cfg)
print(f'compiled on {a.device} in {time.time() - t0:.1f} s, inference precision '
      f'{compiled.get_property("INFERENCE_PRECISION_HINT")}', flush=True)
req = compiled.create_infer_request()

out = req.infer(feed)[compiled.output(0)]
r = ref['output'].numpy()
err = np.abs(out - r); rel = err.max() / np.abs(r).max()
print(f'output vs PyTorch fp32: max abs {err.max():.4f}, max abs / max |ref| {rel:.4f}, '
      f'cosine {float((out * r).sum() / np.linalg.norm(out) / np.linalg.norm(r)):.5f}', flush=True)

for _ in range(3): req.infer(feed)          # warm-up
t0 = time.time()
while time.time() - t0 < a.preheat: req.infer(feed)
time.sleep(a.cool)
ms = []
for _ in range(a.runs):
    t = time.perf_counter(); req.infer(feed); ms.append((time.perf_counter() - t) * 1e3)
ms.sort()
print(f'{a.device} latency over {a.runs} runs: median {st.median(ms):.1f} ms, min {ms[0]:.1f}, max {ms[-1]:.1f}', flush=True)
```
