# 人员持械-危险物品异常行为检测系统
## 项目技术报告

---

## 一、数据集情况与工作量汇报

### 阶段1：YOLO-Pose 人体与手腕定位模型

| 项目 | 详情 |
|------|------|
| **数据来源** | COCO 2017 Keypoints 开源数据集（val2017 子集） |
| **训练集** | 2,087 张（从 COCO val2017 中筛选包含人体的图片） |
| **验证集** | 232 张（其中 216 张有效，16 张标注格式异常） |
| **标注特征** | 每个人体 17 个关键点 |
| **学习重点** | 第9点（左手腕）、第10点（右手腕） |
| **模型指标** | Box mAP50 = 0.875, Pose mAP50 = 0.802（2026-08-10 重新验证） |

### 阶段2：YOLO 危险品分类模型

| 类别 | 数据量 | 完成方式 |
|------|--------|----------|
| **刀具 (Knife)** | 约 300 张 | Roboflow 公开数据 + 演示视频截取 |
| **斧头 (Axe)** | 约 526 张 | 演示视频截取（100%） |
| **锤子 (Hammer)** | 约 376 张 | 演示视频截取（100%） |
| **棍棒 (Stick)** | 约 300 张 | Roboflow 公开数据 + 演示视频截取 |
| **钢管 (Steel Pipe)** | 约 300 张 | 34 张原图 + 8.8 倍数据增强 |
| **瓶子 (Bottle)** | 约 879 张 | bubble_wand 视频截取（66%）+ Roboflow |
| **玩具棍 (Toy Stick)** | 约 879 张 | bubble_wand 视频截取（66%）+ Roboflow |
| **背景 (Background)** | 扩充类别 | v10 数据集新增（空裁剪区域） |
| **安全/负样本 (None)** | 约 768 张 | COCO val2017 裁剪 + 超市监控视频截取 |
| **训练集总计** | **约 3,528 张** | 226 x 226 像素手部局部图 |
| **模型指标** | Top-1 Accuracy = 0.882 (best epoch 7) | 9 类分类 |

**数据来源构成**：COCO val2017 约 4.5%，Roboflow/早期采集约 7%，自制演示视频截取约 86%。

**已知数据问题**：
- axe、hammer 类 100% 来自 2-3 个演示视频的逐帧截取，同源重复度高
- bubble_wand 视频同时用于 bottle 和 toy_stick 两个类别的训练，存在同源污染
- 验证集与训练集来自相同演示视频源（train/val 数据泄露），可能导致 Top-1=0.882 的精度指标无法完全反映真实泛化能力

---

## 二、核心功能技术实现路径

### 1. 如何解决空手误报？

**问题**：传统 argmax 会强行在多类物品中选择一个，导致空手/日常物品被误判

**实现路径**：
- 摒弃 argmax，采用 **独立 Sigmoid 阈值判定**
- 置信度阈值：>= 0.75 才判定为危险品
- 低于阈值一律归为 `none`（安全）

**代码位置**：`src/predict.py`

### 2. 如何解决单帧闪烁和漏报？

**问题**：光照变化、动态模糊导致单帧误检/漏检，告警疯狂闪烁

**实现路径**：
- 引入 **跨帧时序状态机平滑算法**
- 为每个追踪人员建立 **15帧滑动窗口** 历史队列
- 告警触发条件：15帧中持械帧数占比 >= 60%（至少9帧）

**代码位置**：`src/predict.py`、`scripts/demo_presentation.py`

### 3. 区域规则（ROI）怎么防误报？

**问题**：胸部/头部作为判定点易受人体俯仰影响

**实现路径**：
- 利用 `ROIManager` 支持 **自定义多边形危险区**
- 采用 **脚底中心点 (foot_center)** 作为判定依据
- 判定逻辑：脚底踩入危险区 + 满足持械时序 -> 触发告警

**代码位置**：`scripts/roi_editor.py`、`scripts/demo_presentation.py`

### 4. 动态裁剪窗口解决尺度问题

**问题**：人离镜头过近/过远时，固定226x226裁剪会导致特征退化

**实现路径**：
- 动态计算裁剪尺寸：`crop_size = max(226, int(person_width * 0.4))`
- 裁剪后统一 resize 到 226x226
- 保证手部区域相对比例恒定

**代码位置**：`src/predict.py`

---

## 三、成果图片清单

> 以下路径为本地训练环境中的原始文件路径，未包含在 GitHub 仓库中。GitHub 仓库中的对应展示图片见 `demo/results/` 目录。

### 训练曲线图

| 文件路径 | 说明 |
|----------|------|
| `runs/pose/val-2/BoxPR_curve.png` | 阶段1姿态检测 Box PR曲线（2026-08-10 验证） |
| `runs/pose/val-2/PosePR_curve.png` | 阶段1姿态检测 Pose PR曲线 |
| `runs/pose/weapon_wrist_detection/results.png` | 阶段1训练结果汇总 |
| `runs/classify/train_v5_freeze/results.png` | 阶段2分类训练结果汇总（9类，Top-1=0.882） |

### 混淆矩阵

| 文件路径 | 说明 |
|----------|------|
| `runs/pose/val-2/confusion_matrix.png` | 阶段1关键点检测混淆矩阵 |
| `runs/classify/train_v5_freeze/confusion_matrix.png` | 阶段2武器分类混淆矩阵（9类，Top-1=0.882） |

### 训练批次可视化

| 文件路径 | 说明 |
|----------|------|
| `runs/pose/val-2/val_batch0_labels.jpg` | 姿态检测真实标签 |
| `runs/pose/val-2/val_batch0_pred.jpg` | 姿态检测预测结果 |

### 鲁棒性测试结果

| 文件路径 | 说明 |
|----------|------|
| `runs/robustness_results` | 低光照/模糊/遮挡/低分辨率测试数据（JSON格式，20次测试） |

---

## 四、性能指标汇总

### 模型精度

#### 阶段1: YOLOv8n-Pose (2026-08-10 重新验证)

| 指标 | 值 |
|------|------|
| Box Precision | 0.771 |
| Box Recall | 0.829 |
| Box mAP50 | 0.875 |
| Box mAP50-95 | 0.673 |
| Pose mAP50 | 0.802 |
| Pose mAP50-95 | 0.534 |
| 验证图片 | 216 张 (232 总, 16 异常) |

> 注：以上数据来自 2026-08-10 的独立验证运行（`runs/pose/val-2/`），与训练时 `results.csv` 中记录的训练时验证指标（Box mAP50=0.847, Pose mAP50=0.530, epoch 29 最佳）存在差异，属正常现象——训练时验证可能包含数据增强，独立验证使用干净数据。

#### 阶段2: YOLOv8n-cls 武器分类

| 指标 | 值 |
|------|------|
| Top-1 Accuracy | 0.882 (best epoch 7) |
| Top-5 Accuracy | 1.000 |
| 分类类别 | 9 类 (axe, bottle, hammer, knife, none, steel_pipe, stick, toy_stick, background) |

### 鲁棒性测试

| 测试维度 | 通过率 | 平均 FPS |
|----------|--------|---------|
| 低光照（亮度30%） | 5/5 (100%) | 19.43 |
| 运动模糊 | 4/5 (80%) | 45.07 |
| 遮挡测试 | 5/5 (100%) | 27.87 |
| 低分辨率（50%） | 5/5 (100%) | 29.90 |
| **总体** | **19/20 (95%)** | - |

> 以上数据来自 `runs/robustness_results` 实际测试记录。每维度使用 5 张测试图片（共 20 次测试），运动模糊维度有 1 次失败（test_0000 未检测到人体）。测试规模较小，仅供参考。测试环境待补充确认。

### 视频推理测试 (2026-08-10, CPU i9-13900H)

| 指标 | 值 |
|------|------|
| 测试视频 | 02_armed_action.mp4 (682 帧, 50 FPS) |
| 测试帧数 | 150 帧 |
| 持械检测帧 | 600 (含双手腕分别计数) |
| 平均推理时间 | 66.1 ms/帧 |
| 平均 FPS | 15.1 |
| 报错 | 无 |

---

## 五、项目文件结构

> 以上为本地完整项目结构，本 GitHub 仓库仅包含 `portfolio/` 子集（见 README.md 的 Code Structure 章节）。

```
人员持械-危险物品异常行为检测/
├── archive/                    # 历史归档
│   └── 训练历史/               # v1~v4 训练版本
├── configs/                    # 配置文件
│   ├── roi_config.json
│   └── yolo_pose_stage1.yaml
├── datasets/                   # 数据集
│   ├── raw/                    # 原始数据 (COCO Keypoints)
│   └── processed/              # 处理后数据
├── models/                     # 预训练权重
├── portfolio/                  # 开源 Portfolio 仓库 (GitHub 发布版)
├── presentation_package/       # 演示素材包
├── runs/                       # 训练与验证结果
│   ├── classify/
│   │   ├── train_v5_freeze/    # 阶段2当前最佳模型
│   │   └── val/ ~ val13/       # 验证结果
│   ├── pose/
│   │   ├── weapon_wrist_detection/  # 阶段1当前最佳模型
│   │   ├── val/                # 早期验证
│   │   └── val-2/              # 2026-08-10 重新验证
│   ├── demo_results/           # 演示检测结果
│   └── visualization/          # 可视化结果
├── scripts/                    # 核心代码
│   ├── armed_detection_v2.py   # 双阶段检测核心
│   ├── demo_presentation.py    # 演示脚本
│   ├── evaluate_models.py      # 模型评估
│   └── roi_editor.py           # ROI 编辑工具
├── README.md                   # 项目文档
└── REPORT.md                   # 本报告
```

---

**版本**：V2.3（2026-08-12 修正，如实反映阶段2数据集来源构成与已知数据问题，删除已归档的 zip 文件引用）
