# SpatialTIP 整理版与 GitHub 初版的改动说明

整理日期：2026-11-04。

## 1. 对照基准与分类

本文以 [GitHub 初版固定提交 `d2d076f0f2aa5195df926bbf916b83976a22d529`](https://github.com/Yoyiming/SpatialTIP/tree/d2d076f0f2aa5195df926bbf916b83976a22d529) 为基准，对照当前 `SpatialTIP_GitHub` 目录。这里的“初版”均指该公开提交，不指后来本地研究目录中的某个中间版本。

改动分为：**新增文件或入口、已有文件修改、会影响运行结果的行为变化、合并或未纳入、保留的核心设计、兼容性与验证范围**。新增文件不代表首次提出相应算法；例如，初版已包含遮蔽评估、空间平滑、mclust 和标签精修。

按文件路径和内容比较，初版共 46 个文件；整理版包含本文后共 28 个文件：新增 15 个、同名修改 12 个、内容完全相同 1 个，初版另有 33 个文件未纳入。未纳入只表示不在本次发布目录中，原始文件仍保留。

公开初版原本就使用 `spatialtip.py`、`spatialtip_model.py`、`spatialtip_init.py`。本次从本地 `stGPT_*` 研究代码导出这些公开模块名，不能将其描述为相对公开初版的首次命名统一。实际来源和哈希见 [source_provenance.json](source_provenance.json)。

## 2. 新增文件或入口

| 新增内容 | 作用 | 与初版的关系 |
| --- | --- | --- |
| [runtime.py](runtime.py) | 集中读取模型配置、校验词表、初始化模型并严格加载 checkpoint | 初版相关逻辑分散在各脚本中；本次新增统一加载层 |
| [preprocess.py](preprocess.py) | 提供 `fit` / `transform`，分别建立新映射和复用冻结映射 | 初版 README 已描述预处理，但所列的两个 SpatialTIP/HEST 预处理脚本未实际包含在公开快照中；这里补齐实现入口，不将 HVG、归一化或 log1p 称为新算法 |
| [clustering.py](clustering.py) | 将表达合并、空间平滑、PCA、mclust 和标签精修整理为命令行流程 | 初版已有聚类 notebook 和工具函数；属于入口替代与通用化，部分行为差异见第 4 节 |
| [mask_expression.py](mask_expression.py) | 独立生成遮蔽表达、布尔 heldout mask 及元数据 | 初版 `finetune_masked.py` 已有生成 mask 的功能；本次拆出独立入口，同时改变了采样机制及 mask 语义 |
| [output_attentions.py](output_attentions.py) | 导出全体输入 spots 的平均 gene self-attention logits，记录轴和数量 | 初版模型已预留 attention 输出接口，但未在根目录提供对应的 SpatialTIP 导出脚本；初版依赖树中的 `scBERT/attn_sum_save.py` 属于另一套模型 |
| [attention_network.py](attention_network.py) | 从导出矩阵按 query gene 排序 key genes，生成关联边 CSV | 新增轻量结果导出入口，不等同于新增或验证因果调控推断方法 |
| [configs/model.example.json](configs/model.example.json) | 将模型尺寸、词表容量、组织类别数及 attention 后端显式配置 | 初版主要在 Python 初始化代码内固定这些值 |
| `examples/train_samples.csv`、`examples/target_samples.csv` | 给出数据清单格式，显式列出样本、组织、表达和图像路径 | 新增格式示例；其中样本名和路径为占位示例，不是历史训练／测试名单 |
| [requirements.txt](requirements.txt) | 列出直接 Python 依赖和版本 | 配合精简后的 Conda 环境说明使用 |
| `.gitignore`、`.gitattributes` | 排除数据、权重、缓存和个人配置；统一文本换行 | 新增发布目录管理文件 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 说明 BERT、ADE 等代码来源及外部资源范围 | 根 Apache-2.0 LICENSE 保留，补充第三方来源说明 |
| [source_provenance.json](source_provenance.json)、本文 | 记录来源、文件哈希、验证范围及分类改动 | 新增可追溯说明，不包含实验运行器或测试数据 |

## 3. 已有文件修改

| 文件 | 主要修改 | 需要注意的边界 |
| --- | --- | --- |
| [spatialtip.py](spatialtip.py) | FlashAttention 改为可选，增加 SDPA／显式数学后端；显式 attention 在 float32 中计算；无 labels 推理也去掉 CLS 列；外部 heldout 标签从重建监督中排除 | 解码器基本结构和重建损失的全矩阵均值方式保留；后端和监督有效位置的变化不应统称为纯排版调整 |
| [spatialtip_init.py](spatialtip_init.py) | 架构参数外置；UNI 初始化路径显式传入；加载完整 SpatialTIP checkpoint 时无需再次提供独立 UNI 权重 | 历史权重仍要求匹配的模型尺寸、基因 ID 和组织 ID |
| [spatialtip_model.py](spatialtip_model.py) | 参数化初始化；支持未初始化 distributed 的单进程 rank；内部解码器属性保留本地权重的 `stGPT` 名称 | 初版权重前缀为 `spatialtip.*`，整理版通过 `runtime.py` 转换；直接调用旧式加载代码并不自动兼容 |
| [dataset.py](dataset.py) | Dataset 类改为 `SpatialTIPDataset`；明确解码、索引并校验 barcode；支持 dense／sparse；worker 分别打开 HDF5；增加 heldout、尺寸、非负值和组织 ID 等检查 | 联合读取图像与表达、gene→ID、组织输入及 ImageNet 图像归一化均为已有功能；旧类名及构造参数不能原样照搬 |
| [utils.py](utils.py) | 保留 tokenizer、MLM 和分布式基础函数；去掉未使用的 scFoundation loader 导入；空间聚类工具移入新聚类入口 | 旧脚本直接从 `utils` 导入聚类函数时需要修改；不是删除了聚类功能 |
| [train.py](train.py) | 数据清单驱动；与微调共用引擎；支持单进程和 torchrun；记录配置、训练曲线、best／last checkpoint；独立表达周期学习率公式 | 训练 batch 组成、尾批和验证随机性确有改变，详见第 4 节 |
| [finetune.py](finetune.py) | 改为共享引擎的目标切片入口；使用显式文件路径；通过可选 `--heldout` 支持遮蔽训练 | 合并了普通／遮蔽微调入口，但没有完整照搬初版 masked 入口的全部默认设置 |
| [impute.py](impute.py) | 用显式表达、patch、配置、词表、checkpoint 和输出文件替代默认目录拼接；统一三种 `--input-mask`；输出定义和轴检查明确 | 推理默认 batch size 为1、workers为0；初版普通／masked推理分别为64／32、workers均为8，吞吐量会不同。默认遮蔽方式也改变；初版推理原本就保留尾批，不能将全量推理本身列为首次新增 |
| [evaluate.py](evaluate.py) | 显式输入真值、预测和 heldout；支持轴对齐、元数据校验及有效向量计数；不同处理阶段分行记录 | 初版已有 RMSE、MAE、spot／gene PCC 和 cosine；这些指标不是本次新增。默认评估阶段发生变化 |
| [hest_download.py](hest_download.py) | 改为显式 CLI、样本选择和 `snapshot_download`；复用使用者认证；全量下载需 `--all`；导入时不登录或下载 | 公开初版的 token 是占位字符串，不能描述成“初版泄露了真实 token”；本地研究副本的凭据问题与此基准区分 |
| [environment.yml](environment.yml) | 从带 Linux 构建号和绝对 prefix 的完整环境导出，改为 Python 3.11 环境入口；依赖移到 requirements；PyTorch／torchvision 更新为 2.4.1／0.19.1 | 公开初版是 Python 3.9、PyTorch 2.4.0／torchvision 0.19.0，不能与本地另一个旧环境混淆。R/mclust 仍是聚类依赖 |
| [README.md](README.md) | 重写安装、资源要求、命令及输出说明；给出独立的遮蔽微调→预测→评估流程 | 说明示例所需资源和验证范围，未新增权重公开地址或声称已复现论文结果 |

## 4. 会影响运行结果或兼容性的行为变化

以下内容应优先核对，不能仅凭文件名相同就认为结果与初版等价。

| 项目 | GitHub 初版 | 当前整理版与影响 |
| --- | --- | --- |
| **推理输入遮蔽** | 普通 `impute.py` 遮蔽非零；`impute_masked.py` 已经遮蔽零值 | 统一入口默认 `zeros`，保留 `observed` 和 `none`。相对普通 impute，默认条件确实改变；不是给所有旧入口修正同一个错误 |
| **外部评估 mask 的取值** | `0=heldout`，`1=keep` | 布尔 `True=heldout`。旧 mask 不能仅转成 bool 后直接使用，否则含义反转；需按 `legacy_mask == 0` 显式转换，并保证与真值的 spot／gene 顺序一致 |
| **外部 mask 的生成** | 每个 spot 从非零位置无放回选择 `floor(p × 非零数)` 个 | 每个非零条目独立 Bernoulli 抽样，使用不同 RNG。相同比例和 seed 不会重建同一张旧 mask；需要对照旧结果时应复用原 mask |
| **外部 heldout 的训练监督** | masked 微调先将 heldout 置零，后续普通 MLM 未传入单独的外部排除标记 | heldout 输入设为 `-1`、标签设为 `-2`，不进入重建监督。这改变了有效监督条目，不能当作无行为差异的重构 |
| **预训练分组与 batch** | 先合并样本，再对合并数据做全局 90/10 划分和 batching | 逐样本 90/10、分别建 loader。比例保留，但样本划分、batch 组成及对比学习的负样本分布可能改变。普通目标微调仍默认 80/20 |
| **训练／验证尾批** | train、普通 FT 和 masked FT 均 `drop_last=True` | 统一保留尾批。参与训练和验证的 spots、最后一个 batch 的对比损失及验证均值可能改变；DDP 的 DistributedSampler 仍可能补齐重复 spots，保留尾批不表示每个 spot 恰好计入一次，验证均值和计数包含必要的补齐重复 |
| **验证随机性** | 已有 `model.eval()` 和 `torch.no_grad()`，但 MLM 随机状态随运行推进 | 验证使用固定 RNG 并恢复训练 RNG。新增的是可比较的验证遮蔽条件，不是首次加入验证或关闭梯度 |
| **masked FT 默认预算与周期** | `finetune_masked.py` 默认 30 epochs、首周期 30、`cycle_mult=2`；普通 FT 默认 60 epochs、固定 30 周期 | 统一 `finetune.py` 默认 60 epochs、固定 30 周期。仅设 `--epochs 30` 可限制预算，但当前 CLI 不提供 `cycle_mult=2`，不能完整还原旧 masked 入口后续周期 |
| **默认评估阶段** | `evaluate.py` 默认合并保留观测值，再进行 radius150 顺序平滑 | 默认先评估传入预测矩阵；提供 `--input` 增加 completed，显式指定 smoothing 再增加 smoothed。必须比较同一输出阶段，不能直接把默认 CSV 与旧 CSV 对比 |
| **聚类前表达合并** | notebook 实际执行 `np.where(zero_mask, 0, imputed_matrix)`：原零置零、非零取预测，与其文字描述不一致；初版 evaluate 已有正确保留观测值的合并 | 新聚类 CLI 默认保留原非零、用预测填原零。因此相对 notebook 的实际输入矩阵改变；这不是首次引入表达合并策略 |
| **PCA、随机种子与邻域平票** | notebook 的 PCA 未明确 solver；R helper 有独立默认 seed；精修依赖原距离排序 | 新 CLI 明确 randomized PCA，并传递 seed；投票平票仍按有序近邻列表首次出现，完全等距按 spot index 排序。默认配置相近不意味着逐位结果完全相同 |
| **checkpoint 与训练恢复** | 最佳模型为裸 state_dict；微调 `strict=False`，两种推理本已默认严格加载 | 统一严格加载，兼容部分旧前缀；新格式含配置及词表 SHA256，并保存 best／last。没有保存优化器／scheduler 状态，加载 checkpoint 是权重初始化，不是完整断点续训 |

普通预训练与普通微调的核心默认预算仍为 80／60 epochs，非零 MLM 概率仍为 0.15／0.6，零值 MLM 概率仍为 0.05。学习率周期40／warmup5与周期30／warmup2、最低学习率 0.00005、峰值衰减 0.98 保留；独立公式已与相应固定周期参考实现核对。这个结论不扩展到上表中 `cycle_mult=2` 的初版 masked 入口。

### 聚类与 refine30 的具体定位

初版**已有** refinement 算法和 notebook 中 refine30 的调用意图，但 notebook 传入 `n_neighbors=30`，其 `utils.refine_label` 实际形参为 `radius`，存在关键字不匹配。不能说初版没有 refine，也不能仅凭这份 notebook 断言该调用已成功运行。

整理版 `--refine-neighbors 30` 默认执行精修，设为 `0` 关闭。空间表达平滑保留初版已有的 sequential 行更新、包含中心 spot 的方式，没有改成同步均值。`--already-integrated` 显式跳过再次合并和平滑，适用于已含相应处理的矩阵。

R 调用由 Python 进程内的 rpy2 改为 `Rscript` 子进程；mclust EEE 保留，R/mclust 没有被移除。初版也存在不加载 GT 的分支；新版将显式 K、可选参考标签和 ARI／NMI 输出通用化，不根据 GT 自动选择 K。

## 5. 合并、替代或未纳入的文件

| 初版内容 | 文件数 | 本次处理及替代方式 |
| --- | ---: | --- |
| `finetune_masked.py` | 1 | 功能拆到 `mask_expression.py` 与 `finetune.py --heldout`；行为差异见第 4 节 |
| `impute_masked.py` | 1 | 推理并入 `impute.py`，指标计算归入 `evaluate.py` |
| `spatial_domain_clustering.ipynb` | 1 | 聚类流程由 `clustering.py` 提供；未保留 notebook 的交互绘图输出 |
| `scBERT/` | 13 | 未纳入其独立模型、依赖树及权重相关代码；不再为一个 scheduler 导入整个目录。其独立 GPLv3 来源在第三方说明中交代 |
| `DLPFC/*/*_truth.txt` | 12 | 本次代码版不附带这些标注；聚类需要评估时由用户提供参考标签列 |
| `hest_data/*.json` | 5 | 初版的三份 vocabulary 和两份 tissue map 未复制到本次代码版；当前匹配资源亦未从离线盘导出 |

**初版确实提供了词表和组织映射。** 未纳入的三份词表为 `vocabulary_3k.json`、`vocabulary_healthy_3k.json`、`vocabulary_cancer_3k.json`；组织映射为 `tissue_dict_healthy.json`、`tissue_dict_cancer.json`。不能将本次未复制资源写成“初版没有资源”。

词表不能仅凭大小替换：初版前两份词表均有 20,938 个条目、gene key 集合相同，但其中 20,936 个 key 的 ID 不同。旧权重必须使用各自确切的原映射；新 `fit` 词表的稳定排序也不能替代它。新 checkpoint 的词表哈希检查只能校验带该元数据的文件，不能补证旧裸权重的基因 ID 语义。

本地研究目录中的补充实验、baseline 扩展、消融、敏感性、绘图、临时测试、论文和结果文件，本次均未引入发布目录。这些内容原本不在上述公开初版中，不计为“从初版删除的文件”。初版没有随源码提供完整 UNI／SpatialTIP 权重，本次也未新增权重分发。

## 6. 保留的核心设计与接口迁移

保留 UNI ViT-L 与六层基因解码器的默认结构、前18个视觉 blocks冻结／末6个可训练、CLS 表示、零／非零分别进行 MLM、重建 MSE 按完整 batch×gene 矩阵取均值、对比损失系数 0.5，以及原来的跨 rank gather 约定。参数化允许指定不同尺寸，但这并不使任意配置都能加载已有权重。

根目录 [LICENSE](LICENSE) 与初版逐字节一致。源码来源说明增加；未使用的 loader 和独立 scBERT 模型分发被排除。

迁移已有命令时，需对照 [README](README.md) 更新：

- 常用 CLI 参数由 `--batch_size`／`--learning_rate` 等改为 `--batch-size`／`--learning-rate`；原来的 sample＋默认目录拼接改为显式文件路径和 manifest。
- 旧 masked 两个入口、旧 `SpatialTIPData` 类及从 `utils` 导入的聚类函数，不保证原调用方式兼容。
- 旧 `spatialtip.*`／`module.*` 权重前缀通过 `runtime.load_model` 规范到当前 `stGPT.*`；仅修改文件名不能完成权重和词表兼容。
- 复用历史结果时，保持原始 mask、基因／spot 轴、词表、输入遮蔽方式和处理阶段；不能只对齐比例、seed 或最终指标名称。

## 7. 已完成验证与仍需资源验证的部分

| 验证范围 | 已确认内容 |
| --- | --- |
| 静态与接口 | 16个 Python 模块的语法、CLI、导入及输出保护；未将测试脚本或测试数据放入发布目录 |
| 数据预处理与加载 | 隔离 Python 3.11 环境中的真实 Seurat-v3、dense／sparse、H5AD／HDF5、条码乱序、多 worker 与 heldout 检查 |
| 模型数值与 checkpoint | 小规模真实基因解码器的新旧参数键／形状和显式 attention 计算对照、反传；合成 checkpoint 的严格加载、前缀转换及词表哈希拒绝 |
| 训练管理 | 小型替代模型的单进程与两进程 CPU/Gloo 流程；普通预训练／微调的固定周期 LR 序列对照 |
| 推理、指标与关联导出 | 合成数据的全量输出、三种输入遮蔽、已知 RMSE／MAE、独立 cosine 复算、attention 平均和关联排序 |
| 聚类 | R 4.4.2／mclust 6.1.1 的有 GT／无 GT 完整流程，含 PCA、空间平滑、refine30 与结果写入 |

这些检查不等同于完整科研模型的重新训练。完整原始 checkpoint、其配套词表／组织映射、实际 GPU／多GPU运行以及论文结果复现尚未验证；HEST 仅验证下载接口，未实际重新下载。数据、UNI 和 checkpoint 所在的外接盘在整理时不可访问，未据旧公开 JSON 猜配当前权重。

本说明描述当前代码版已发生的变化，不将性能提升、历史实验重现或真实权重兼容性写成已完成结论。
