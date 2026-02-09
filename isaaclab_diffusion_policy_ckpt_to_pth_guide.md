# IsaacLab + Diffusion Policy：`.ckpt` 到 IsaacLab 可用权重格式的工程化方案

本文给出学术界/工业界常见的做法，解决你在 **IsaacLab 采集数据 + Diffusion Policy（官方代码）训练 + 回到 IsaacLab 测试** 流程中的权重格式与推理接口问题。

---

## 1. 先澄清：`.ckpt` 和 `.pth` 本质差异

在 PyTorch 生态中，`.ckpt` 与 `.pth` 只是**文件后缀习惯**，不是强制的二进制格式标准。

- `.ckpt`（常见于 PyTorch Lightning）通常是一个大字典，可能包含：
  - `state_dict`（模型参数）
  - optimizer 状态
  - lr scheduler 状态
  - epoch / global_step 等训练信息
- `.pth` 常见约定是：
  - 仅保存 `model.state_dict()`（推理部署更轻）
  - 或保存一个自定义字典（同样可以很复杂）

所以问题通常不是“`.ckpt` 不能在 IsaacLab 用”，而是：

1. IsaacLab 侧加载器是否只读取了纯 `state_dict`。
2. 模型结构（类定义、key 命名、观测/动作维度）是否一致。
3. 归一化、时序窗口、动作后处理等训练/推理契约是否一致。

---

## 2. 主流可落地方案（推荐顺序）

### 方案 A（首选）：训练后导出“推理专用 `.pth`”

这是工业界最常见的部署方式：

1. 用 Diffusion Policy 正常训练得到 `.ckpt`。
2. 写一个导出脚本：
   - 从 `.ckpt` 里取出 `state_dict`。
   - 去掉前缀（比如 Lightning 常见的 `model.`）。
   - 只保留推理需要字段，另存为 `policy_infer.pth`。
3. IsaacLab 侧只加载该 `.pth` 进行推理。

优点：
- 模型包小、依赖少、部署稳定。
- 推理路径与训练解耦，便于版本管理。

---

### 方案 B：在 IsaacLab 直接兼容 `.ckpt` 读取

在 IsaacLab 的 policy wrapper 里做“多格式加载”：

- 若检测到 checkpoint 字典中有 `state_dict`，则取该字段。
- 若是纯参数字典，则直接 `load_state_dict`。

优点：
- 改动小，短期联通快。

缺点：
- 推理端绑定训练框架细节（Lightning 等），长期维护成本较高。

---

### 方案 C（更工程化）：导出 TorchScript / ONNX

对高稳定部署、跨语言/跨平台或加速部署需求，常见做法是从 `.ckpt` 恢复后导出：

- TorchScript（PyTorch 友好）
- ONNX（跨框架生态）

但对 diffusion policy 这类包含采样循环与复杂控制逻辑的策略，导出难度和兼容成本往往高于直接 `.pth`。

---

## 3. 你这个项目最可能踩坑的“非后缀问题”

即使成功转成 `.pth`，以下不一致仍会导致性能异常（这是论文复现和工业落地中最常见失败点）：

1. **观测字典结构不一致**
   - 键名、顺序、shape、坐标系（世界系/机体系）、单位（m/rad）是否完全一致。
2. **归一化统计量不一致**
   - 训练时使用的数据均值方差、范围缩放参数必须带到推理。
3. **时序窗口不一致**
   - `n_obs_steps`、`horizon`、动作重规划频率必须一致。
4. **动作后处理不一致**
   - clip、scale、IK/控制器接口（位置/速度/力矩）必须匹配。
5. **domain gap**
   - IsaacLab 训练环境与测试环境随机化参数、物理步长、控制频率不一致。

> 实践建议：将“策略权重 + 归一化统计量 + 配置 yaml + 代码 commit hash”打包成一个可追溯 artifact。

---

## 4. 标准导出脚本模板（`.ckpt` -> `.pth`）

你可以在 Diffusion Policy 工程里放一个独立脚本（如 `tools/export_policy_pth.py`），核心逻辑如下：

```python
import torch


def extract_state_dict(ckpt_obj):
    # Lightning checkpoint 常见结构
    if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj:
        sd = ckpt_obj['state_dict']
    else:
        sd = ckpt_obj

    # 兼容前缀（按你的工程实际修改）
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith('model.'):
            nk = nk[len('model.'):]
        new_sd[nk] = v
    return new_sd


if __name__ == '__main__':
    in_ckpt = 'checkpoints/last.ckpt'
    out_pth = 'checkpoints/policy_infer.pth'

    ckpt = torch.load(in_ckpt, map_location='cpu')
    state_dict = extract_state_dict(ckpt)

    export_obj = {
        'state_dict': state_dict,
        # 建议同时打包这些元数据，确保可复现部署
        'meta': {
            'obs_keys': ['...'],
            'action_dim': 0,
            'n_obs_steps': 0,
            'horizon': 0,
            'normalization_ref': 'stats.npz'
        }
    }
    torch.save(export_obj, out_pth)
    print(f'Exported: {out_pth}')
```

---

## 5. IsaacLab 侧推荐加载器模式（健壮版本）

在 IsaacLab 推理入口实现：

1. 先 `torch.load(weight_path)`。
2. 若存在 `state_dict` 字段，取 `ckpt['state_dict']`，否则把对象当作参数字典。
3. `model.load_state_dict(sd, strict=True)`（首次联调建议 strict=True 快速暴露 key 不一致）。
4. 确保 `model.eval()`，并在推理中正确处理设备和精度。

当 key 不一致时：
- 打印 `missing_keys` / `unexpected_keys`。
- 对照训练代码的模块命名修复前缀，而不是盲目 `strict=False`。

---

## 6. 工业级交付建议（强烈推荐）

为了避免“能跑但不对”，建议建立以下最小闭环：

1. **离线回放验证**
   - 在训练数据上跑策略，检查动作分布、成功率与训练日志是否一致。
2. **仿真 A/B 对照**
   - 同一环境里对比：
     - 训练框架原生推理
     - IsaacLab 集成推理
   - 指标：成功率、回报、轨迹误差、动作平滑度。
3. **配置锁定**
   - 固定随机种子、物理参数、控制频率。
4. **artifact 版本化**
   - 权重、归一化参数、配置和代码 hash 一并存档。

---

## 7. 结论（针对你的问题）

你的核心解决路径应是：

- **不要纠结后缀**，重点做“训练 checkpoint 到推理权重”的标准导出。
- 采用 **方案 A：导出推理专用 `.pth`**，并在 IsaacLab 写兼容加载器。
- 同时严格对齐观测、归一化、时序与控制接口，这比文件后缀更关键。

如果你愿意，我下一步可以直接按你当前 `diffusion_policy` 目录结构，给出一版“可直接运行”的导出脚本与 IsaacLab 侧加载代码骨架（含 key 对齐检查日志）。
