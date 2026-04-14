# RFUAV ChannelSim Viewer — 产品需求文档（PRD）

> 版本：v1.3 | 日期：2026-04-09 | 状态：草稿

---

## 一、项目概述

**项目名称：** RFUAV ChannelSim Viewer

**项目类型：** 桌面端可视化离线工具（Python + PySide6/Qt，基于本地 Python 环境运行）

**核心功能：** 加载本地 IQ 二进制文件，通过交互式 UI 配置并叠加 8 种信道退化效果，实时展示原始信号与各信道处理后信号的时频谱图及差异对比，并支持导出处理后的 IQ 数据。

**目标用户：** 无线通信算法工程师、RF 物理层研究人员、通信波形设计与验证人员。

---

## 二、功能范围

### 2.1 信道模拟模块

| 编号 | 信道类型 | 后端入口函数 | 可配置参数 | 默认值 |
|------|----------|------------|-----------|--------|
| 1 | **AWGN** | `apply_awgn()` | SNR (dB) | 15 dB |
| 2 | **CFO** | `apply_cfo()` | 频偏量 CFO (Hz) | 15 kHz |
| 3 | **Fading** | `apply_rayleigh_or_rician_fading()` | 衰落类型 (Rayleigh/Rician)、K因子、块大小 | Rayleigh, K=5, block=256 |
| 4 | **Multipath** | `apply_multipath()` | 最大时延 (样本数)、路径数、指数衰减系数 | delay=20, n_paths=4, decay=0.35 |
| 5 | **Phase Noise** | `apply_phase_noise()` | 每样本相位标准差 (弧度) | 0.002 |
| 6 | **IQ Imbalance** | `apply_iq_imbalance()` | 幅度不平衡 (dB)、相位不平衡 (度) | 1.0 dB, 3.0° |
| 7 | **Crop+TimeShift** | `apply_random_crop_and_time_shift()` | 最小保留比例、时间偏移上限 | 0.7, 300 样本 |
| 8 | **Doppler** | `apply_doppler()` | 多普勒频移 (Hz)、多普勒扩展 (Hz)、Jakes 模型开关 | 0 Hz, 100 Hz, 关闭 |

- 每个信道效果可**独立开关**（启用 / 禁用），快速对比叠加效果。
- 所有参数通过 UI 控件（滑块 + 输入框）实时调整。
- 信道按固定顺序串行叠加：`preprocess → fading → doppler → multipath → cfo → phase_noise → iq_imbalance → awgn → crop_time_shift`。
- 全局随机种子可配置（保证结果可复现）。

### 2.2 信号预处理链

在信道模拟之前，部分场景需要对输入信号进行预处理。预处理作为独立的操作单元，位于信道链最前端：

| 预处理操作 | 描述 | 可配置参数 | 默认值 |
|-----------|------|-----------|--------|
| **数字滤波** | 带通/低通/高通 FIR/IIR 滤波 | 滤波器类型（Chebyshev/Butterworth）、截止频率、过渡带宽 | 关闭 |
| **重采样** | 改变采样率（up/down sampling） | 目标采样率、滤波器类型 | 关闭（保持原采样率） |
| **增益控制** | 手动幅度缩放 | 增益值 (dB) | 0 dB |
| **DC 偏置去除** | 去除 IQ 直流分量 | 启用/禁用 | 启用 |

> **处理顺序**：预处理 → 信道模拟链 → 导出。预处理结果可在中间 stage 选择器中查看。

### 2.3 信号输入输出

- **输入：** 本地 `.iq` 二进制文件（float32 交错格式：`I0, Q0, I1, Q1, ...`）。
- 支持采样率 `fs` 和中心频率 `fc` 手动设定（UI 输入框）。
- 输入信号信息（样本数、时长、功率）实时显示在界面上。
- **导出处理后 IQ：** 支持将最终叠加结果或任意中间 stage 的 IQ 数据导出为 `.iq` 二进制文件。

### 2.4 可视化展示

每种展示均可同时显示**原始信号**与**信道处理后信号**（左右/上下分栏对比），展示类型包括：

| 展示类型 | 描述 | 对应后端逻辑 |
|----------|------|------------|
| **时域包络图** | `|x(t)|` 幅度随时间变化 | 直接绘制 |
| **功率谱密度 (PSD)** | FFT 幅度谱 (dB)，加汉宁窗 | `np.fft.fft + 窗函数` |
| **STFT 时频图** | 短时傅里叶变换谱图 | 复用 `visualization.plot_time_frequency` |
| **原始 vs 处理信号对比图** | 左右子图叠加，直观展示差异 | 由 pipeline 中各 stage 产生 |
| **残差 / 差值谱图** | 处理前后信号差的频谱 | 计算 `|FFT(原始) - FFT(处理后)|` |

STFT 使用默认参数：`nperseg=256, noverlap=192`，可在 UI 中调节。

### 2.4.1 谱图交互增强

在基本谱图展示的基础上，增加丰富的交互功能，提升信号分析的精确度和效率：

| 交互功能 | 描述 | 实现方式 |
|---------|------|---------|
| **光标读数** | 鼠标悬停显示频率/时间/幅度坐标值 | `mpl_connect('motion_notify_event')` 捕获鼠标位置，动态更新坐标显示 |
| **峰值标记** | 自动标注频谱峰值频率与幅度 | `scipy.signal.find_peaks` 峰值检测，结果标注在谱图上 |
| **频率轴缩放** | 可放大到特定频段，支持拖拽平移 | `ax.set_xlim` + 鼠标滚轮缩放，`panalytics` 平移 |
| **动态播放** | 对长信号支持播放动画，时间窗口滑动查看 STFT | QTimer 定时器驱动 STFT 窗口滑动，播放速度可调 |
| **区域选择** | 框选频谱特定区域进行局部放大 | `mpl_connect('button_press_event')` 框选逻辑 |
| **十字光标** | 显示垂直/水平参考线，辅助读数 | Matplotlib `axvline` + `axhline` |

> **性能注意**：交互事件（鼠标移动、缩放）同样需要 200ms 防抖处理，避免频繁重绘。

### 2.4.2 信道质量评估指标面板

谱图提供直观的视觉展示，指标面板提供量化的信号质量评价。在信号概览区下方增加独立的**统计指标卡片**，实时计算并显示：

| 指标 | 计算方式 | 物理含义 | 典型阈值 |
|------|---------|---------|---------|
| **EVM (%)** | $\sqrt{P_{error}/P_{reference}} \times 100\%$ | 调制信号质量综合评价 | ≤5% (良好), ≤10% (可接受) |
| **峰值因子 (Crest Factor)** | $10\log_{10}(P_{peak}/P_{avg})$ | 信号动态范围 | 6-10 dB (OFDM), 12-15 dB (QAM) |
| **ACLR (dBc)** | $10\log_{10}(P_{adj}/P_{main})$ | 邻道泄漏比 | ≥45 dBc (4G/5G 标准) |
| **信号功率 (dBFS)** | $10\log_{10}(P_{signal}/P_{fullscale})$ | 信号电平（满量程为参考） | -3 dBFS (峰值安全阈值) |
| **ACPR (dBc)** | 与 ACLR 等价 | 相邻信道功率比 | 通信标准常用指标 |
| **邻道功率 (dBm)** | 积分相邻频段功率 | 带外辐射绝对值 | — |

> **实现注意**：EVM、ACLR 等指标需同时传入原始参考信号与处理后信号，在 pipeline 的 `finished` 信号回调中计算，`processing_worker.py` 负责计算逻辑，`signal_info_panel.py` 负责结果显示。

### 2.5 配置管理

- **保存配置：** 将当前所有参数（fs, fc, 8个信道的开关与参数值、随机种子）导出为 `.yaml` 文件。
- **加载配置：** 从 `.yaml` 文件恢复完整参数。
- **重置：** 一键恢复所有默认值。
- **配置对比：** 支持同时加载 2 个 YAML 配置，谱图区域分为三栏对比显示。

### 2.6 导出功能

- **导出 IQ 数据：** 支持导出最终叠加结果（`processed.iq`），以及各中间 stage 的 IQ（原始/fading/multipath/cfo 等），文件格式为 float32 二进制交错。
- **导出谱图：** 将当前展示区的所有谱图导出为 PNG 图片（支持导出当前视图或全部四种视图）。

### 2.7 频偏估计与补偿工具

针对 CFO 信道效果，增加一对互补工具，用于验证信道注入效果和教学演示：

| 功能 | 描述 | 实现方式 |
|------|------|---------|
| **频偏估计** | 对含 CFO 的信号进行频偏估计，显示估计值 vs. 实际注入值 | FFT 峰值法、互相关法两种模式可选 |
| **频偏补偿** | 对含 CFO 信号应用补偿算法，验证补偿效果 | 基于估计结果的相位旋转补偿 |

> **展示方式**：在指标面板中显示「估计值 / 注入值」对比，并显示补偿后的残余误差。

### 2.8 批量处理模式

对于需要处理大量文件或测试多组参数的场景，增加独立的批处理入口：

```yaml
# batch_config.yaml 示例
input_dir: "./iq_files/"
output_dir: "./results/"
config_file: "./configs/scenario.yaml"
parameter_sweep:
  snr_db: [10, 15, 20, 25, 30]  # 参数扫描
  cfo_hz: [5000, 10000, 15000]
export_summary: true  # 输出汇总 CSV
```

- 前台显示进度条（复用当前 worker 机制），支持暂停/取消
- 后台生成每组参数的结果文件（IQ + PNG）
- 汇总 CSV 包含：文件名、参数组合、处理耗时、关键指标
- 支持断点续传（记录已完成的处理任务）

### 2.9 自定义信道脚本（高级用户 API）

当前所有信道效果都硬编码在后端中。对于高级用户，暴露一个 Python API 入口：

```python
# 示例：自定义信道脚本
from ChannelSimulation.viewer.pipeline_runner import CustomPipeline

pipeline = CustomPipeline()
pipeline.add_stage("awgn", snr_db=20)
pipeline.add_stage("custom", func=my_custom_function)
result = pipeline.run(complex_array)
```

- 支持用户传入自定义 NumPy 函数作为新的信道模块
- 通过 UI 菜单「高级 → 加载自定义脚本」注入 `.py` 文件
- 需要沙箱隔离（执行超时限制 30s，禁止文件系统访问）

### 2.10 暗色模式主题

除浅色主题外，增加暗色模式切换：

- `ui_style.py` 中定义两套 QSS 变量集（`THEME_LIGHT` / `THEME_DARK`）
- 工具栏或设置菜单中提供「浅色 / 深色」切换按钮，快捷键 `Ctrl+T`
- 暗色模式下谱图配色自动切换为高对比度色图（降低整体亮度）
- 主题偏好自动保存到用户配置

### 2.11 预设配置模板库

在配置管理区增加一个预设模板下拉框，包含常见的信道场景：

| 预设名称 | 信道配置 | 典型应用 |
|---------|---------|---------|
| **卫星下行 (LEO)** | AWGN + 高多普勒 + Rayleigh 衰落 + 多径 | LEO 卫星通信 |
| **无人机集群** | Rayleigh 衰落 + 中等多普勒 + Phase Noise | UAV 蜂群网络 |
| **地面蜂窝** | Rician 衰落 + 轻度多径 + AWGN | 4G/5G 地面信道 |
| **雷达回波** | 多径 + 加性噪声 + 目标 Doppler | 雷达系统仿真 |
| **纯净参考** | 所有信道关闭 | 基线对比 |

- 预设模板以 YAML 形式存储在 `viewer/presets/` 目录下
- 支持用户自定义添加/编辑/删除预设
- 预设列表可在「配置 → 管理预设」菜单中维护

### 2.12 帮助与参数说明面板

将参数帮助内容文档化，提升工具可学习性：

- 内置**可折叠的帮助面板**（位于主窗口右侧或独立标签页）
- 每个参数附带：物理意义说明、典型取值范围、工程经验值、引用标准
- 提供「关于」弹窗：项目版本、依赖库版本、许可证信息、联系方式
- 内置快速入门教程（首次启动时显示引导）

### 2.13 完整日志记录

建立完整的日志系统，便于问题排查和用户反馈收集：

```python
# 日志记录内容
- 用户操作日志（加载文件、修改参数、导出操作）
- 错误日志（文件读取失败、计算异常）
- 性能日志（处理耗时、内存占用、帧率）
```

- 日志级别可切换（DEBUG / INFO / WARNING / ERROR）
- 日志文件存放在 `viewer/logs/` 目录，按日期滚动
- 支持 UI 内置日志查看器（可滚动查看/搜索/导出）

### 2.14 性能监控面板

在状态栏或独立面板中实时显示系统资源占用：

| 指标 | 显示方式 | 刷新频率 |
|------|---------|---------|
| **CPU 使用率** | 百分比 + 迷你进度条 | 1 Hz |
| **内存占用** | 当前 / 最大（MB） | 1 Hz |
| **GPU 使用率** | （如有 CUDA） | 1 Hz |
| **处理帧率** | FPS（STFT 刷新频率） | 实时 |

- 大文件处理场景下内存监控尤为重要，可预防 OOM
- 可在「视图」菜单中切换性能监控面板的显示/隐藏

### 2.15 中间信号查看器

在流水线处理过程中，可查看任意中间 stage 的信号状态，便于调试和分析：

| 功能 | 描述 | 实现方式 |
|------|------|---------|
| **Stage 下拉选择器** | 位于谱图区工具栏，`[原始 ▼]`，可选任意中间 stage | QComboBox，下拉列表显示所有可用的 stage |
| **流水线可视化** | 显示当前信号经过的链路：`IQ → DC去除 → 滤波 → Fading → ... → 输出` | 流程图形式的 Mini 面板，标记当前选中 stage |
| **任意两点比较** | 选择任意两个 stage 对比其信号差异 | 谱图区扩展为三栏模式 |

> **可选 stage 列表**：`原始` → `预处理` → `Fading` → `Doppler` → `Multipath` → `CFO` → `PhaseNoise` → `IQImbalance` → `AWGN` → `CropShift` → `最终输出`

### 2.16 撤销/重做功能

支持参数调整的撤销/重做，便于参数探索与对比：

```python
# 快捷键
- Ctrl+Z：撤销最近一次参数修改
- Ctrl+Y / Ctrl+Shift+Z：重做
- 记录参数修改历史（最多 50 步）
```

- **触发时机**：参数控件（滑块、输入框、下拉框）失焦后记录
- **批量操作处理**：粘贴配置或加载 YAML 时作为单次操作记录
- **实现方式**：使用 `QUndoCommand` 栈管理，或自行实现 command pattern

### 2.17 多文件格式支持

除默认的 `.iq` float32 交错格式外，增加对其他常见 IQ 数据格式的支持：

| 格式 | 文件后缀 | 数据类型 | 说明 | 优先级 |
|------|---------|---------|------|--------|
| **Raw Binary** | `.cfile` / `.raw` / `.bin` | float32 / int16 / int8 | 无 header，直接二进制 | P1 |
| **MATLAB** | `.mat` | float32 | MATLAB v7.3+ HDF5 格式 | P2 |
| **NumPy** | `.npy` / `.npz` | complex64 / complex128 | NumPy 数组格式 | P2 |
| **Signed Int16** | `.iq16` | int16 交错 | 量化数据格式 | P2 |

> **自动检测**：文件对话框根据扩展名自动选择解析器；无法判断时弹出格式选择对话框。

### 2.18 窗口状态持久化

除 YAML 配置外，还保存用户界面的窗口状态：

```yaml
# 窗口状态持久化（存储在 ~/.channelsim_viewer/state.json）
window:
  geometry: [x, y, width, height]  # 窗口位置和大小
  maximized: false                  # 是否最大化
  splitter_sizes: [280, 920]        # QSplitter 分隔比例
  theme: "light"                    # 当前主题
  log_panel_visible: false          # 日志面板可见性

recent:
  input_dir: "/path/to/iq_files"   # 上次打开的目录
  files: [file1, file2, ...]       # 最近打开的文件（最多 10 个）
  configs: [config1, ...]          # 最近使用的配置（最多 10 个）
```

- 状态文件存储在用户配置目录（`~/.channelsim_viewer/`）
- 程序启动时自动恢复上次退出时的窗口状态
- 可在「设置」中清除所有持久化状态

### 2.19 参数边界约束

为所有参数配置硬性约束，防止无效输入：

| 约束类型 | 示例 | 说明 |
|---------|------|------|
| **取值范围** | SNR: -50~50 dB, CFO: 0~fs/4 Hz, K因子: 0~100 | 超出范围时自动截断或拒绝输入 |
| **数据类型** | 整数参数不接受小数 | 自动取整或提示错误 |
| **单位自动转换** | 支持 "100 MHz" / "0.1 GHz" 两种输入 | 自动转换为内部 SI 单位 |
| **参数联动** | 启用 Doppler 时，自动建议合理的 Doppler Spread | 提供智能默认值提示 |

> **错误提示**：无效输入时控件边框变红，tooltip 显示有效范围。

### 2.20 离线与安全声明

本工具作为完全离线的桌面应用，明确声明：

- **完全离线运行**：无需互联网连接，所有依赖库通过 pip 本地安装
- **无遥测上报**：不包含任何数据分析或使用情况上报功能
- **内网部署**：支持在严格隔离的内网环境中部署
- **许可证声明**：采用 [MIT / Apache 2.0] 开源许可证，所有依赖库（PySide6/Numpy/SciPy/Matplotlib）兼容商业使用

---

## 三、非功能需求

### 3.1 性能要求（核心优先级）

> 高性能是本系统的首要非功能需求，贯穿整个设计。

| 指标 | 目标 | 实现手段 |
|------|------|----------|
| 参数调整 → 谱图刷新延迟 | ≤ 1 秒（100 万样本以内） | 后台线程 + NumPy 向量化计算 |
| 大文件处理（>1 GB） | 不卡顿，内存占用可控 | 复用 `run_channel_pipeline_streaming` 分块处理 + 进度反馈 |
| 谱图重绘帧率 | 交互无撕裂 / 延迟感 | Matplotlib `blit=True` 增量刷新 + `QtAgg` 硬件加速 |
| 内存峰值 | ≤ 输入文件大小的 3 倍 | 流式读写、中间信号按需生成、避免不必要拷贝 |
| 启动时间 | ≤ 3 秒 | 延迟导入非核心模块（PyYAML、scipy 等） |

---

## 四、界面设计规范

### 4.1 整体视觉风格

遵循 **Qt Fusion Design Language**（PySide6 原生风格），界面简洁、专业、高效，符合工程软件审美：

- **主题：** 默认浅色（Light Fusion），程序主色调为深蓝（#1a73e8）与科技灰（#f8f9fa）搭配
- **字体：** 系统默认无衬线字体（Windows: Segoe UI / macOS: SF Pro / Linux: Ubuntu），等宽字体用于数值显示
- **圆角：** 控件圆角半径 4px，弹窗/面板 8px，营造柔和现代感
- **阴影：** 面板卡片使用 0 3px 6px rgba(0,0,0,0.08) 轻阴影，层次分明但不喧宾夺主
- **间距系统：** 基准间距 8px，控件间距 12px，内边距 16px，区块间距 24px

### 4.2 色彩规范

| 用途 | 色值 | 说明 |
|------|------|------|
| 主色调（Primary） | #1a73e8 | 按钮、选中态、进度条 |
| 次要色（Secondary） | #5f6368 | 次要文字、图标 |
| 强调色（Accent） | #ea4335 | 警告、删除、关键异常 |
| 成功色（Success） | #34a853 | 加载完成、成功状态 |
| 背景色（Background） | #ffffff | 主窗口背景 |
| 卡片背景 | #f8f9fa | 配置面板、信道卡片背景 |
| 边框色（Border） | #dadce0 | 分割线、控件边框 |
| 深色文字 | #202124 | 主要文字 |
| 浅色文字 | #5f6368 | 辅助说明文字 |

**谱图配色方案（Turbo 色图）：** 科技感强，色阶平滑过渡，dark mode 下自动切换为高对比度模式。

### 4.3 控件规范

| 控件类型 | 视觉规范 | 交互行为 |
|----------|----------|----------|
| **滑块 (QSlider)** | 滑块宽度 18px，高度 4px，轨道主色填充 | 拖动时实时显示当前值浮窗 |
| **数值输入框 (QLineEdit)** | 右侧单位标签(Label)，背景 #f1f3f4 | 失焦或回车触发参数更新 |
| **开关复选框 (QCheckBox)** | 自定义样式：28×16px 圆角胶囊，选中态为 #1a73e8 | 勾选/取消触发重算 |
| **下拉框 (QComboBox)** | 宽度自适应，展开最大高度 240px | 选择后立即触发重算 |
| **按钮 (QPushButton)** | 高度 36px，文字 #1a73e8，主按钮填充背景色 | Primary 按钮用于"应用预览"、"导出" |
| **标签 (QLabel)** | 等宽字体显示数值，#202124 色 | — |

### 4.4 界面布局（详细设计）

采用经典**三分区**布局，最大化利用屏幕空间：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌─ 菜单栏 ──────────────────────────────────────────────────────────┐ │
│  │  文件 F                      配置 C          帮助 H                │ │
│  │  ├─ 加载 IQ Ctrl+O           ├─ 保存配置      ├─ 关于              │ │
│  │  ├─ 导出 IQ Ctrl+S           ├─ 加载配置      └─ 使用说明          │ │
│  │  ├─ 导出谱图 Ctrl+E          └─ 重置默认                             │ │
│  │  └─ 退出程序                 [已修改 ●] 指示标签（配置变更提醒）      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ 工具栏 ──────────────────────────────────────────────────────────┐ │
│  │  [📂 加载] [💾 保存] [▶ 预览]  │  采样率: [________] MHz           │ │
│  │  [↻ 重置]  [🔍 放大] [🔎 缩小] │  中心频率: [________] GHz         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ 左侧栏 280px ─┐  ┌─ 主区域（自适应）───────────────────────────┐  │
│  │                │  │  ┌─ 视图切换 TabBar ─────────────────────┐  │  │
│  │ ┌─ 信号概览 ─┐ │  │  │  [时域]  [PSD]  [STFT]  [残差谱]     │  │  │
│  │ │  文件: ...  │ │  │  └───────────────────────────────────────┘  │  │
│  │ │  样本: ...  │ │  │                                             │  │
│  │ │  时长: ...  │ │  │  ┌──────────────┬──────────────┐           │  │
│  │ │  采样率:.. │ │  │  │              │              │           │  │
│  │ │  功率: ... │ │  │  │  原始信号    │  处理后信号   │           │  │
│  │ └────────────┘ │  │  │              │              │           │  │
│  │                │  │  │   Spectrum   │  Spectrum    │           │  │
│  │ ┌─ 随机种子 ─┐ │  │  │              │              │           │  │
│  │ │ [    42  ] │ │  │  └──────────────┴──────────────┘           │  │
│  │ └────────────┘ │  │                                             │  │
│  │                │  │  ┌─ 颜色条 ─────────────────────────────┐  │  │
│  │ ┌─ 信道配置 ─┐ │  │  │ -80 dB  ████████████░░░░░░  0 dB    │  │  │
│  │ │ ─────────  │ │  │  └───────────────────────────────────────┘  │  │
│  │ │ ▣ AWGN  ● │ │  │                                             │  │
│  │ │    SNR  ──○── │  │  ┌─ 状态栏 ─────────────────────────────┐  │  │
│  │ │          15 dB│ │  │  ●就绪  │  处理: 0.32s  │  样本: 1M   │  │  │
│  │ │ ─────────  │ │  │  └───────────────────────────────────────┘  │  │
│  │ │ ▣ CFO   ● │ │  │                                             │  │
│  │ │   CFO   ──○── │  │  ┌─ 导出控制条 ──────────────────────────┐  │  │
│  │ │         15 kHz│ │  │  导出IQ: [原始 ▼] [导出]   导出图: [导出]│  │  │
│  │ │ ─────────  │ │  │  └───────────────────────────────────────┘  │  │
│  │ │ ▣ Fading  ○ │ │  └─────────────────────────────────────────────┘  │
│  │ │   类型 [Rayl] │ │                                                │
│  │ │   K因子 ──○── │ │                                                │
│  │ │          5.0  │ │                                                │
│  │ │ ─────────  │ │  │                                                │
│  │ │ ▣ Multipath○ │ │                                                │
│  │ │ ▣ PhsNise ○ │ │                                                │
│  │ │ ▣ IQImbal ○ │ │                                                │
│  │ │ ▣ CropTSh ○ │ │                                                │
│  │ └────────────┘ │                                                │
│  └────────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**说明：**
- **左侧栏（280px 固定）：** 信号概览 + 随机种子 + 8 个信道配置卡片 + 4 个预处理卡片，垂直滚动。
- **右侧主区域（自适应）：** 谱图展示区自适应填满剩余空间，工具栏/状态栏固定。
- **视图切换：** TabBar 风格的 Segmented Button，位于谱图区顶部。
- **状态栏：** 显示处理耗时、当前样本数、就绪/处理中/完成状态。
- **信道卡片：** 每个信道折叠为一个可展开卡片，点击卡片标题展开/折叠参数组。

### 4.5 交互设计原则

| 原则 | 实现方式 |
|------|----------|
| **即时反馈** | 参数变化时，控件旁显示当前值（浮层 tooltip），处理中状态栏显示沙漏图标 |
| **防抖合并** | 滑块拖动、输入框修改统一经过 200ms 防抖，避免一次拖动触发十几次重算 |
| **进度感知** | 大文件处理时底部进度条（QProgressBar），实时显示当前 stage 和百分比 |
| **错误恢复** | 处理异常时状态栏变红+震动，错误信息可展开查看，不影响程序运行 |
| **键盘快捷键** | Ctrl+O 加载 / Ctrl+S 导出IQ / Ctrl+E 导出图 / Ctrl+R 重置 / F5 重新预览 / Ctrl+T 切换主题 / Ctrl+Z 撤销 / Ctrl+Y 重做 |
| **上下文提示** | 每个参数旁提供 "?" 按钮，悬停显示参数物理意义与取值范围说明 |

---

## 五、技术架构

### 5.1 技术栈总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              应用层（Application Layer）                              │
│   主窗口 · 信道面板 · 谱图画布 · 菜单/工具栏 · 状态栏 · 帮助面板 · 日志查看器              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                              业务逻辑层（Business Logic Layer）                       │
│   config_manager · export_utils · metrics_calculator · cfo_estimator              │
│   presets_manager · batch_processor · custom_script_engine                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                              信号处理层（Signal Processing Layer）                    │
│   pipeline_runner · spectrum_data · processing_worker                               │
│   NumPy 向量化计算 · FFT/PSD/STFT · 信道质量指标计算                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                              后端集成层（Backend Integration Layer）                 │
│   ChannelSimulation/* (awgn / cfo / fading / multipath / doppler / ...)           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                              跨平台运行时（Cross-Platform Runtime）                  │
│   Python 3.10+ · PySide6 (Qt 6.x) · NumPy · SciPy · Matplotlib                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 技术栈详细清单

| 层级 | 技术选型 | 版本要求 | 用途说明 |
|------|---------|---------|---------|
| **编程语言** | Python | ≥ 3.10 | 主开发语言，支持类型注解（`from __future__ import annotations`） |
| **GUI 框架** | PySide6 | ≥ 6.5 | Qt for Python，LGPL 许可，商业友好 |
| **数学计算** | NumPy | ≥ 1.24 | 向量化运算、FFT、内存映射 |
| **科学计算** | SciPy | ≥ 1.10 | 滤波器设计（`scipy.signal`）、峰值检测（`scipy.signal.find_peaks`） |
| **数据可视化** | Matplotlib | ≥ 3.7 | 谱图渲染，`matplotlib.backends.backend_qtagg` |
| **配置序列化** | PyYAML | ≥ 6.0 | YAML 配置文件读写 |
| **类型检查** | mypy | ≥ 1.0 | 静态类型检查（可选，提高代码质量） |
| **代码格式化** | Black | ≥ 23.0 | 统一代码风格 |
| **打包分发** | PyInstaller | ≥ 5.0 | 生成可执行文件 |
| **单元测试** | pytest | ≥ 7.0 | 自动化测试框架 |

### 5.3 UI 框架详解

#### 5.3.1 核心组件矩阵

| 组件类别 | PySide6 类 | 使用场景 |
|---------|-----------|---------|
| **主窗口** | `QMainWindow` | 程序主容器，包含菜单栏、工具栏、状态栏 |
| **对话框** | `QDialog`, `QMessageBox` | 文件对话框、关于弹窗、确认对话框 |
| **布局容器** | `QVBoxLayout`, `QHBoxLayout`, `QGridLayout`, `QSplitter` | 控件排列、响应式布局 |
| **折叠卡片** | `QWidget` + `QPushButton` | 信道参数折叠面板（自实现） |
| **输入控件** | `QSlider`, `QLineEdit`, `QComboBox`, `QCheckBox` | 参数配置 |
| **显示控件** | `QLabel`, `QProgressBar`, `QStatusBar` | 信息展示 |
| **绘图画布** | `FigureCanvasQTAgg` | Matplotlib 嵌入 Qt |
| **菜单系统** | `QMenuBar`, `QMenu`, `QAction` | 菜单栏 |
| **工具栏** | `QToolBar` | 工具栏按钮 |
| **定时器** | `QTimer` | 防抖定时、播放动画、性能监控刷新 |

#### 5.3.2 信号与槽（Signal/Slot）

```python
# 关键信号定义
from PySide6.QtCore import Signal, QThread
from typing import Any, Dict, Optional
import numpy as np

class ProcessingWorker(QThread):
    """后台处理线程，处理 IQ 信号并更新谱图"""
    finished = Signal(np.ndarray)                    # 处理完成，携带结果数据
    progress = Signal(float, str)                   # 进度更新（百分比，当前 stage 名称）
    error = Signal(str)                             # 错误信息
    metrics_ready = Signal(dict)                     # 指标计算完成

    def __init__(self, iq_data: np.ndarray, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.iq_data = iq_data
        self.config = config

class MainWindow(QMainWindow):
    """主窗口，管理所有 UI 组件和信号连接"""
    # 菜单/工具栏信号
    load_iq_triggered = Signal(str)                 # 加载 IQ 文件
    export_iq_triggered = Signal(str, int)           # 导出 IQ 文件，参数：路径、stage_id
    export_fig_triggered = Signal(str, int)          # 导出图片，参数：路径、view_type

    # 参数变更信号（带防抖）
    stage_config_changed = Signal(dict)               # 信道配置变更
    preprocess_config_changed = Signal(dict)          # 预处理配置变更
```

#### 5.3.3 布局策略

```
主窗口布局（QMainWindow）
├── 菜单栏（QMenuBar）—— 固定高度
├── 工具栏（QToolBar）—— 固定高度 48px
├── 中央区域（QSplitter 水平分割）
│   ├── 左侧栏（280px 固定宽度）
│   │   ├── 信号概览卡片（QGroupBox）
│   │   ├── 随机种子卡片（QGroupBox）
│   │   ├── 预处理卡片（4个折叠卡片 QWidget）
│   │   │   ├── DC 去除卡片
│   │   │   ├── 数字滤波卡片
│   │   │   ├── 重采样卡片
│   │   │   └── 增益控制卡片
│   │   └── 信道卡片（8个折叠卡片 QWidget）
│   │       ├── AWGN 卡片
│   │       ├── CFO 卡片
│   │       ├── Fading 卡片
│   │       ├── Multipath 卡片
│   │       ├── Phase Noise 卡片
│   │       ├── IQ Imbalance 卡片
│   │       ├── Doppler 卡片
│   │       └── Crop+TimeShift 卡片
│   └── 主区域（自适应宽度）
│       ├── 视图切换 TabBar（QTabWidget / SegmentedButton）
│       ├── 谱图画布区（QSplitter 垂直分割）
│       │   ├── 原始信号画布（FigureCanvasQTAgg）
│       │   └── 处理后信号画布（FigureCanvasQTAgg）
│       ├── 颜色条（QLabel + QLinearGradient）
│       └── 导出控制条（QWidget）
└── 状态栏（QStatusBar）—— 固定高度 24px
    ├── 状态指示（处理中/就绪）
    ├── 处理耗时
    ├── 样本数
    └── 性能监控（CPU/内存）
```

#### 5.3.4 QSS 样式系统

| QSS 特性 | 示例 | 说明 |
|---------|------|------|
| **ID 选择器** | `QPushButton#btn_load` | 根据 objectName 精确匹配 |
| **类选择器** | `QPushButton` | 匹配所有该类型的控件 |
| **子控件** | `QSlider::groove:horizontal` | 滑块轨道样式 |
| **状态伪类** | `QPushButton:hover` | 悬停效果 |
| **选中伪类** | `QPushButton:checked` | 选中状态 |
| **禁用伪类** | `QPushButton:disabled` | 禁用状态 |
| **渐变背景** | `qlineargradient` | 按钮渐变效果 |
| **圆角** | `border-radius: 4px` | 控件圆角 |
| **阴影** | `box-shadow: 0 2px 4px rgba(0,0,0,0.1)` | 卡片阴影 |

> **暗色主题变量示例**：
> ```css
> /* THEME_LIGHT */
> :root {
>     --bg-primary: #ffffff;
>     --bg-secondary: #f8f9fa;
>     --text-primary: #202124;
>     --text-secondary: #5f6368;
>     --border-color: #dadce0;
>     --accent-color: #1a73e8;
> }
>
> /* THEME_DARK */
> [theme="dark"] {
>     --bg-primary: #1e1e1e;
>     --bg-secondary: #2d2d2d;
>     --text-primary: #e0e0e0;
>     --text-secondary: #9aa0a6;
>     --border-color: #404040;
>     --accent-color: #4a90e2;
> }
> ```

> **注意：** 避免使用 `QtDesigner` 生成 `.ui` 文件后反编译的方式，全部 UI 代码使用纯 Python 编写，保持代码可读性和版本可控性。

### 5.4 图表渲染技术

#### 5.4.1 Matplotlib QtAgg 集成

| 组件 | PySide6 类 | Matplotlib 类 | 说明 |
|------|------------|--------------|------|
| **画布容器** | `QWidget` | — | 承载 Matplotlib 画布的 Qt 容器 |
| **绘图画布** | — | `FigureCanvasQTAgg` | PySide6 与 Matplotlib 的绑定桥梁 |
| **图形对象** | — | `Figure` | 包含一个或多个坐标轴的顶层容器 |
| **坐标轴** | — | `Axes` | 实际绑制数据的区域 |
| **图像项** | — | `AxesImage` | STFT 时频图的像素数据 |
| **线条项** | — | `Line2D` | 时域/PSD 的曲线数据 |

#### 5.4.2 blit 加速渲染流程

```python
class SpectrumCanvas(FigureCanvasQTAgg):
    def __init__(self, figure):
        super().__init__(figure)
        self.axes = figure.add_subplot(111)
        self.background = None

    def initial_draw(self):
        self.draw()
        self.background = self.copy_from_bbox(self.axes.bbox)

    def update_plot(self, data: np.ndarray):
        self.axes.cla()
        self.axes.imshow(data, cmap='turbo', aspect='auto', origin='lower')
        self.restore_region(self.background)
        self.axes.draw_artist(self.axes.images[0])
        self.blit(self.axes.bbox)
```

#### 5.4.3 谱图配色方案

| 色图名称 | 适用场景 | 说明 |
|---------|---------|------|
| **turbo** | 默认 | 谷歌开源，色阶平滑，高动态范围 |
| **viridis** | 通用/色盲友好 | Matplotlib 默认色图 |
| **plasma** | 备选 | 蓝-紫-黄渐变 |
| **magma** | 暗色主题 | 黑-红-白渐变 |

#### 5.4.4 谱图渲染参数

| 参数 | 值 | 说明 |
|------|---|------|
| **STFT nperseg** | 256 | 窗长，影响频率分辨率 |
| **STFT noverlap** | 192 | 重叠点数（75%重叠） |
| **STFT nfft** | 256 | FFT 长度 |
| **STFT return_onesided** | False | 双边谱 |
| **PSD 窗函数** | Hanning | 减少频谱泄漏 |
| **DPI（屏幕）** | 150 | 兼顾清晰与性能 |
| **DPI（导出）** | 300 | 印刷质量 |
| **最大绘制点数** | 2,000,000 | 防止 Matplotlib OOM |

#### 5.4.5 交互事件处理

| 事件类型 | Matplotlib 事件 | 说明 |
|---------|----------------|------|
| **鼠标移动** | `motion_notify_event` | 更新光标坐标显示 |
| **鼠标按下** | `button_press_event` | 框选放大 |
| **滚轮缩放** | `scroll_event` | 频率轴缩放 |
| **按键** | `key_press_event` | R 重置，Space 播放 |

### 5.5 信号处理技术

#### 5.5.1 数据流水线架构

```
IQ 文件 → PipelineRunner → 谱图计算 → 指标计算
                ↓
         每个 stage 输出中间信号
```

#### 5.5.2 NumPy 向量化运算

| 运算类型 | NumPy API | 说明 |
|---------|-----------|------|
| **IQ 读取** | `np.fromfile()` | 二进制文件读取 |
| **幅度计算** | `np.abs(complex_array)` | 向量幅度 |
| **FFT** | `np.fft.fft()` | 快速傅里叶变换 |
| **STFT** | `scipy.signal.stft()` | 短时傅里叶变换 |
| **窗函数** | `np.hanning()` | 窗函数 |
| **滤波** | `scipy.signal.lfilter()` | IIR 滤波 |

#### 5.5.3 FFT 与频谱计算

```python
def compute_psd(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    windowed = signal * np.hanning(n)
    spectrum = np.fft.fft(windowed)
    psd = np.abs(spectrum[:n//2]) ** 2 / (np.sum(np.hanning(n)**2) * fs)
    psd_db = 10 * np.log10(psd + 1e-12)
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    return freqs, psd_db

def compute_stft(signal: np.ndarray, fs: float,
                 nperseg: int = 256, noverlap: int = 192):
    from scipy.signal import stft
    freqs, times, Zxx = stft(signal, fs=fs, nperseg=nperseg,
                               noverlap=noverlap, return_onesided=False)
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
    stft_db = 20 * np.log10(np.abs(Zxx_shifted) + 1e-12)
    return np.fft.fftshift(freqs), times, stft_db
```

#### 5.5.4 内存映射与流式处理

| 场景 | 方法 | 内存占用 |
|------|------|---------|
| **小文件（<100MB）** | `np.fromfile()` | 文件大小 × 2 |
| **中文件（100MB-1GB）** | `np.memmap()` | 按需加载，峰值 < 500MB |
| **大文件（>1GB）** | 分块处理 | 块大小 × 3 |

```python
def load_iq_memmap(file_path: str, dtype='float32'):
    return np.memmap(file_path, dtype=dtype, mode='r', offset=0)

def run_pipeline_streaming(file_path: str, config: dict, block_size: int = 10_000_000):
    mm = np.memmap(file_path, dtype='float32', mode='r')
    n_samples = len(mm) // 2
    result = np.zeros(n_samples, dtype=np.complex128)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        block = mm[start*2:end*2].view(np.complex128)
        result[start:end] = apply_pipeline(block, config)[:end-start]
        yield (end / n_samples, f"处理中: {end/n_samples*100:.1f}%")
```

#### 5.5.5 随机数与可复现性

| 函数 | 说明 |
|------|------|
| `np.random.default_rng(seed)` | 创建确定性随机数生成器 |
| `rng.normal(size=n)` | 正态分布随机数 |

### 5.6 多线程与异步模型

| 组件 | 说明 |
|------|------|
| **主 UI 线程** | Qt EventLoop，响应用户操作，渲染 Matplotlib |
| **ProcessingWorker** | QThread，运行 pipeline，FFT/PSD/STFT 计算 |

#### 5.6.1 QThread 生命周期

```python
class ProcessingWorker(QThread):
    finished = Signal(np.ndarray)
    progress = Signal(float, str)
    error = Signal(str)

    def run(self):
        try:
            result = self._process()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._should_stop = True
        self.wait(5000)
```

- 永远不在主线程做信号处理，避免 UI 冻结
- Signal/Slot 跨线程自动处理，无需手动 marshal
- QThread 比 Python threading 更适合信号处理

### 5.7 文件格式处理技术

| 格式 | 后缀 | 数据类型 | 解析器 |
|------|------|---------|-------- |
| **IQ Float32** | `.iq` | float32 | `parse_iq_float32()` |
| **Raw Binary** | `.cfile`, `.raw` | float32/int16 | `parse_raw_binary()` |
| **Signed Int16** | `.iq16`, `.s16` | int16 | `parse_int16()` |
| **MATLAB** | `.mat` | float32 | `scipy.io.loadmat()` |
| **NumPy** | `.npy` | complex64/complex128 | `np.load()` |

```python
def detect_format(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    format_map = {
        '.iq': 'iq_float32', '.cf32': 'iq_float32',
        '.cfile': 'raw_binary', '.raw': 'raw_binary',
        '.iq16': 'int16', '.s16': 'int16',
        '.mat': 'matlab', '.npy': 'numpy', '.npz': 'numpy_zip'
    }
    return format_map.get(suffix, 'unknown')
```

### 5.8 配置持久化格式

```yaml
# viewer 配置文件格式（与 channel_pipeline.sample.yaml 完全兼容）
version: "1.0"              # 格式版本，便于未来格式演进
fs: 100000000               # 采样率 Hz
fc: 5800000000.0            # 中心频率 Hz（可选）
seed: 42

preprocess:                  # 信号预处理配置（可选，默认全部关闭）
  dc_removal:
    enabled: true           # 默认启用
  digital_filter:
    enabled: false
    type: lowpass          # lowpass / highpass / bandpass
    filter_class: butterworth  # butterworth / chebyshev
    cutoff_hz: 50000000     # 截止频率
    order: 8                # 滤波器阶数
  resampling:
    enabled: false
    target_fs: 200000000    # 目标采样率
    filter_type: firls      # 滤波器类型
  gain_control:
    enabled: false
    gain_db: 0             # 增益值 dB

stages:                     # 启用的 stage 顺序（与后端 STAGE_ORDER 一致）
  - fading
  - doppler
  - multipath
  - cfo
  - phase_noise
  - iq_imbalance
  - awgn
  - crop_time_shift

stage_configs:
  fading:
    enabled: true
    method: rayleigh
    params:
      k_factor: 5.0
      block_size: 256

  doppler:
    enabled: false
    method: jakes
    params:
      doppler_shift_hz: 1000.0    # 多普勒频移
      doppler_spread_hz: 100.0   # 多普勒扩展

  multipath:
    enabled: true
    method: tdl
    params:
      max_delay_samp: 20
      n_paths: 4
      exp_decay: 0.35

  cfo:
    enabled: true
    method: constant
    params:
      cfo_hz: 15000.0

  phase_noise:
    enabled: false
    method: wiener
    params:
      phase_std_per_sample: 0.002

  iq_imbalance:
    enabled: false
    method: widely_linear
    params:
      gain_imbalance_db: 1.0
      phase_imbalance_deg: 3.0

  awgn:
    enabled: true
    method: gaussian
    params:
      snr_db: 15.0

  crop_time_shift:
    enabled: false
    method: basic
    params:
      min_keep_ratio: 0.7
      max_shift: 300
```

> **兼容性保证：** `channel_pipeline.sample.yaml` 缺少 `enabled` 字段，`config_manager` 在加载时默认 `enabled=true`（向后兼容）。

---

## 六、项目结构

```
ChannelSimulation/
    viewer/
        __init__.py
        app.py                    # 程序入口，延迟导入，启动画面
        main_window.py            # 主窗口（菜单栏/工具栏/布局/signal连接）
        ui_style.py               # QSS 样式表定义（主题色/控件样式/布局常量，浅色/暗色双主题）
        signal_info_panel.py       # 信号信息展示面板 + 指标卡片
        channel_config_panel.py   # 8个信道折叠卡片 + 4个预处理卡片
        spectrum_viewer.py         # 谱图画布（Matplotlib QtAgg Canvas + blit）
        spectrum_data.py           # 谱图数据计算（FFT/PSD/STFT/残差，与绘图解耦）
        pipeline_runner.py         # 封装后端，提供单一 pipeline 入口
        processing_worker.py       # QThread 后台处理线程
        config_manager.py          # YAML 配置保存/加载/校验，配置对比
        presets_manager.py         # 预设模板库管理
        metrics_calculator.py      # EVM/ACLR/峰值因子等指标计算
        export_utils.py            # 导出 IQ（二进制）+ 导出 PNG
        shortcuts.py               # 键盘快捷键定义
        utils.py                   # 通用工具函数（文件大小格式化/进度百分比等）
        cfo_estimator.py           # 频偏估计与补偿算法
        batch_processor.py         # 批量处理模式 + 参数扫描
        custom_script_engine.py    # 自定义信道脚本引擎（沙箱执行）
        help_panel.py              # 帮助面板 + 参数说明
        log_manager.py             # 日志记录与查看器
        performance_monitor.py    # CPU/内存/GPU 监控面板
        presets/                    # 预设模板 YAML 文件目录
        logs/                      # 日志文件目录
        requirements.txt           # 依赖清单
```

---

## 七、关键技术实现要点与注意事项

### 7.1 UI 开发注意事项

1. **PySide6 vs PyQt6 API 差异：** 本项目使用 PySide6，注意信号定义使用 `Signal` 而非 `pyqtSignal`。
2. **Qt 对象父子关系：** 所有 Qt 控件在创建时必须指定 `parent`，避免内存泄漏；布局管理器自动承担删除职责。
3. **QSS 样式隔离：** 滑块、下拉框等复杂控件的自定义样式需精确指定 `::subcontrol` 等子控件，否则样式可能不生效。
4. **高 DPI 屏幕适配：** 主窗口创建时设置 `QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)` 和 `Qt.AA_UseHighDpiPixmaps`，保证 Retina/4K 屏幕清晰。
5. **Dark Mode 适配：** 如需支持暗色主题，通过 `QPalette` 设置或切换 QSS，不修改 Matplotlib rcParams（matplotlib 与 Qt 主题独立）。
6. **窗口最小尺寸：** 设置 `setMinimumSize(1200, 800)`，避免谱图在小窗口下无法正常显示。
7. **延迟导入加速启动：** `import yaml`、`import scipy`、`import matplotlib.pyplot` 均在首次使用时导入，不在 `app.py` 顶部全量导入。

### 7.2 信号处理注意事项

1. **数据类型一致性：** IQ 数据在 pipeline 中全程使用 `np.complex128`，避免 float32 精度损失。
2. **FFT 长度选择：** PSD 的 FFT 长度使用信号本身长度（不补零），STFT 使用 `nperseg=256` 固定窗长。
3. **频谱泄漏控制：** PSD 计算时使用 Hanning 窗（`np.hanning`），窗口化后补零至信号长度再 FFT。
4. **STFT 与后端一致性：** `spectrum_data.py` 中的 STFT 参数（`nperseg`、`noverlap`、`return_onesided=False`）必须与 `visualization.plot_time_frequency` 完全一致，否则 PSD 峰值得分贝值不一致。
5. **残差谱归一化：** 残差 `|FFT(原始) - FFT(处理后)|` 需转换为 dB，`+ EPS (1e-12)` 防止 log(0)。
6. **power vs amplitude：** `normalize_power` 使用 `|x|^2` 归一化，PSD 使用 `20*log10(|X|)`，两者物理含义一致。
7. **大文件 decimation：** `MAX_PLOT_POINTS = 2_000_000`、`MAX_STFT_COLS = 3_000` 两个阈值严格控制渲染数据量，防止 Matplotlib OOM。

### 7.3 多线程注意事项

1. **禁止主线程等待子线程：** 永远不要在主线程中使用 `worker.wait()` + 轮询查询进度，会导致 UI 假死。
2. **QThread 生命周期：** `ProcessingWorker` 在 `finished` 信号发出后，主窗口负责 `deleteLater()` 销毁对象，避免内存泄漏。
3. **信号断开时机：** 在启动新 worker 前，必须先 `disconnect()` 旧 worker 的所有信号，防止旧 worker 回调新窗口。
4. **异常不能跨线程传播：** `run_pipeline` 中的异常必须在 worker 内捕获，通过 `error.emit()` 传递到主线程处理，绝不能让异常逃逸到 Qt 事件循环。
5. **NumPy 与 QThread：** NumPy 的 GIL 释放特性使 `np.fft` 等操作可以在 QThread 中高效执行，无需额外处理。
6. **避免在子线程操作 Qt 控件：** 只能在主线程更新 QLabel/QLineEdit 等控件，子线程通过 Signal 触发主线程的 Slot 更新。

### 7.4 文件 I/O 注意事项

1. **文件路径编码：** Windows 中文路径支持，`file_path` 使用 `Path` 对象，`open(..., encoding='utf-8')`。
2. **IQ 文件校验：** 读取后检查 `len(iq) < 2` 则报错；文件大小为奇数字节时丢弃最后一个字节（与 `io_utils` 行为一致）。
3. **导出覆盖确认：** 导出 IQ/图片时若文件已存在，弹出 `QMessageBox.warning` 确认是否覆盖。
4. **大文件写入：** `save_interleaved_iq` 使用 `np.tofile()` 直接写二进制，比 `with open()` + `write()` 快 3~5 倍。
5. **YAML 安全性：** 使用 `yaml.safe_load()` 而非 `yaml.load()`，防止任意代码执行漏洞。

### 7.5 性能优化 checklist

| 检查项 | 目标 | 验证方法 |
|--------|------|----------|
| 启动时间 | ≤ 3s | `time python -m ChannelSimulation.viewer.app` |
| 谱图刷新延迟（1M样本） | ≤ 1s | 掐表测量参数变化到图刷新完成 |
| 内存占用（100M样本） | ≤ 500MB | `taskmgr` 或 `tracemalloc` |
| 内存占用（1GB文件） | ≤ 300MB（memmap模式） | 同上 |
| 滑块拖动无卡顿 | 帧率 ≥ 30fps | 肉眼观察，拖动时图随动无撕裂 |
| 大文件进度反馈 | 每秒更新一次 | 掐表验证 |

### 7.6 防抖与批量更新策略

参数变化时的防抖和批量更新机制，确保性能最优：

| 场景 | 策略 | 实现 |
|------|------|------|
| **滑块拖动** | Debounce 200ms | 拖动结束后才触发重算，避免频繁更新 |
| **输入框修改** | Debounce 200ms | 失焦或回车后触发 |
| **下拉框选择** | 即时触发 | 无需防抖，单次操作 |
| **批量粘贴/加载配置** | 取消 pending 任务 | 立即取消旧的 pending 任务，仅执行新任务 |
| **配置覆盖** | 作为单次 undo 操作 | 加载配置视为一次完整的参数修改 |

> **注意**：使用 `QTimer.singleShot` 实现 debounce，而非 `QTimer` 循环。

### 7.7 内存泄漏防护

以下措施防止内存泄漏，确保长时间运行稳定：

| 检查项 | 方法 | 验证 |
|--------|------|------|
| **Qt 对象父子关系** | 所有控件创建时指定 `parent=` 参数 | 使用 `tracemalloc` 检测 |
| **Matplotlib 图形清理** | 每次谱图更新前调用 `plt.close('all')` | 内存 profiler 检测 |
| **NumPy 数组生命周期** | 避免在闭包中捕获大数组引用 | 代码审查 |
| **QThread 资源释放** | `worker.deleteLater()` + `wait(5000)` | 长时间运行稳定性测试 |
| **信号连接清理** | 断开旧连接：`worker.disconnect()` | 每次启动新 worker 前执行 |

---

## 八、风险与约束

1. **内存约束：** 对超大文件（>1 GB）使用 `np.memmap` 内存映射读取，避免全量加载；超出内存映射范围时走 `run_channel_pipeline_streaming` 并显示进度条。
2. **UI 响应性：** 所有信号处理在子线程（`QThread` + Signal/Slot）中执行，主 UI 绝不阻塞；参数变化通过防抖（debounce 200ms）合并重复触发。
3. **向后兼容：** YAML 格式完全兼容现有 `channel_pipeline.sample.yaml`，用户可将 UI 中配置的参数导出后在 CLI 端复用。
4. **Python 环境：** 用户需具备 Python 3.10+ 环境，通过 `pip install -r requirements_viewer.txt` 安装依赖后运行。
5. **离线部署：** 完全离线运行，不依赖网络，可在内网隔离环境中部署。

---

## 九、开发方案

### 9.1 开发阶段划分

```
阶段一：框架与基础设施（预计 1 天）
  ├── 新建 viewer/ 目录与 __init__.py
  ├── 新建 requirements.txt（依赖清单）
  ├── 新建 ui_style.py（QSS 样式表，主题色/控件样式，浅色/暗色双主题）
  ├── 新建 app.py（程序入口，延迟导入，Splash 启动画面）
  ├── 新建 main_window.py（主窗口骨架 + 菜单栏 + 工具栏）
  └── 安装并验证 PySide6 + matplotlib QtAgg 可用

阶段二：核心数据层（预计 1~2 天）
  ├── 新建 signal_info_panel.py（信号信息读取与显示 + 指标卡片）
  ├── 新建 channel_config_panel.py（8个信道折叠卡片控件 + 预处理卡片）
  ├── 新建 pipeline_runner.py（封装现有后端，提供单一入口）
  │     → 按 stage_configs 调用各 apply_* 函数
  │     → 支持按 stage 逐级输出中间信号
  │     → 对大文件使用 memmap 读取，不整块加载内存
  ├── 新建 processing_worker.py（QThread 后台处理 + 进度 Signal）
  └── 新建 metrics_calculator.py（EVM/ACLR/峰值因子等指标计算）

阶段三：可视化层（预计 2~3 天）
  ├── 新建 spectrum_data.py（谱图数据计算，与绘图逻辑解耦）
  │     → compute_time_domain / compute_psd / compute_stft / compute_residual
  ├── 新建 spectrum_viewer.py（Matplotlib 嵌入 Qt）
  │     → 时域包络、PSD、STFT、残差谱 四种视图
  │     → 原始信号 vs 处理信号 左右对比
  │     → blit=True 增量刷新，TabBar 切换视图
  │     → 光标读数、峰值标记、缩放平移、动态播放
  └── 处理长信号的 decimation 逻辑

阶段四：配置管理与导出（预计 1~2 天）
  ├── 新建 config_manager.py（YAML 保存/加载/校验，配置对比）
  ├── 新建 presets_manager.py（预设模板库管理）
  ├── 新建 export_utils.py（导出 IQ + 导出 PNG，支持导出中间 stage）
  ├── 新建 shortcuts.py（键盘快捷键定义）
  ├── 新建 utils.py（通用工具函数）
  └── 端到端集成测试（加载→配置→预览→导出）

阶段五：高级功能（预计 2~3 天）
  ├── 新建 cfo_estimator.py（频偏估计与补偿算法）
  ├── 新建 batch_processor.py（批量处理模式 + 参数扫描）
  ├── 新建 custom_script_engine.py（自定义信道脚本引擎，沙箱执行）
  ├── 新建 help_panel.py（帮助面板 + 参数说明）
  ├── 新建 log_manager.py（日志记录与查看器）
  └── 新建 performance_monitor.py（CPU/内存/GPU 监控面板）
```

### 9.2 关键实现策略摘要

| 策略编号 | 策略名称 | 核心目标 |
|----------|----------|----------|
| 策略 1 | 零修改复用后端 | 不碰 ChannelSimulation/ 下任何文件 |
| 策略 2 | blit 增量刷新 | 谱图重绘帧率提升 3~5 倍 |
| 策略 3 | QThread + Signal/Slot | 主 UI 永不阻塞，异常安全跨线程通信 |
| 策略 4 | np.memmap 内存映射 | GB 级文件处理不 OOM |
| 策略 5 | 200ms 防抖 | 滑块拖动时避免重复计算 |
| 策略 6 | 延迟导入 | 启动时间 ≤ 3 秒 |
| 策略 7 | YAML 格式兼容 | UI 配置可导出供 CLI 复用 |
| 策略 8 | 谱图数据与绘图解耦 | `spectrum_data.py` 纯计算，`spectrum_viewer.py` 纯绘图，便于单元测试 |
| 策略 9 | 双主题 QSS 变量集 | 浅色/暗色主题共用一套控件样式，仅切换颜色变量 |
| 策略 10 | 沙箱执行环境 | 自定义脚本使用 `exec()` 隔离，限制时间和内存访问 |
| 策略 11 | 指标与谱图解耦 | `metrics_calculator.py` 独立计算，信号与参考信号同时传入 |
| 策略 12 | 预设模板目录化 | `presets/` 目录下 YAML 文件自动扫描注册为预设选项 |

### 9.3 依赖清单

```
# requirements.txt
PySide6>=6.5
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
PyYAML>=6.0
```

### 9.4 里程碑与交付物

| 里程碑 | 完成标准 | 预计工时 |
|--------|----------|----------|
| **M1** 项目脚手架 | 窗口可启动，QSS 主题生效，浅色/暗色双主题切换，菜单/工具栏可用，布局正确响应缩放 | 0.5 天 |
| **M2** IQ 加载 + 信号信息 | 能加载 .iq 文件，显示样本数/时长/功率；>1GB 文件自动走内存映射 | 0.5 天 |
| **M3** 信道参数面板 + 预处理 | 8 个信道折叠卡片 + 4 个预处理卡片，QSS 样式美观，防抖触发后台重算 | 1.5 天 |
| **M4** 谱图展示（高性能） | 四种视图 TabBar 切换，`blit` 加速；原始vs处理左右对比；状态栏显示耗时；光标读数/峰值标记/缩放平移 | 2 天 |
| **M5** 指标面板 | EVM/ACLR/峰值因子/dBFS 等量化指标实时显示 | 0.5 天 |
| **M6** 配置管理 + 预设模板 | YAML 保存/加载/重置，配置变更指示器，预设模板下拉框，5 种预设场景 | 1 天 |
| **M7** 导出功能 | IQ 导出（含中间 stage 选择）+ PNG 导出（含 DPI 选项）+ 多配置对比视图 | 1 天 |
| **M8** 集成测试 | 端到端流程（加载→配置→预览→导出）+ 性能 checklist 验证 | 0.5 天 |

**基础功能总预计工时：约 7 天**

#### 高级功能里程碑

| 里程碑 | 完成标准 | 预计工时 | 优先级 |
|--------|----------|----------|--------|
| **M9** 频偏估计与补偿 | CFO 估计（FFT 峰值法/互相关法）+ 补偿验证，指标面板展示对比 | 1 天 | P2 |
| **M10** 批量处理 | 批处理入口 + 参数扫描 + 进度条 + CSV 汇总 + 断点续传 | 1.5 天 | P3 |
| **M11** 自定义脚本 | Python API 入口 + 沙箱执行 + UI 菜单注入 | 1 天 | P3 |
| **M12** 帮助与日志 | 可折叠帮助面板 + 参数说明 + 日志记录器 + 日志查看器 UI | 1 天 | P2 |
| **M13** 性能监控 | CPU/内存/GPU 实时监控 + OOM 预警 | 0.5 天 | P3 |

**高级功能总预计工时：约 5 天**

### 9.5 运行方式

```bash
# 安装依赖
pip install -r ChannelSimulation/viewer/requirements.txt

# 启动程序
python -m ChannelSimulation.viewer.app
```
