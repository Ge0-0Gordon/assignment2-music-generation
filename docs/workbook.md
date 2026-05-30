# CSE253 Assignment 2 Workbook 说明文档

本文档根据 `notebooks/workbook.ipynb` 的当前内容和已运行输出整理。目标是解释每一部分为什么这样写、代码起什么作用、每个函数负责什么，以及 notebook 中每个主要输出的意义。

当前 notebook 的主线是：

- Dataset: Nottingham
- Task 1: symbolic unconditioned MIDI generation
- Task 2: symbolic conditioned MIDI continuation
- Baseline: Markov / n-gram
- Main model: density-conditioned GPT2-style causal Transformer from scratch
- Output root: `outputs/final/`

重要约束：

- 不使用 MAESTRO checkpoint。
- 不使用 pretrained GPT-2 weights。
- 不使用 GPT-2 text tokenizer。
- 不使用 pretrained music checkpoint。
- notebook 中 submission creation 和 HTML export 默认关闭。

---

## 先回答：为什么第 9 部分里很多 score 是 NaN？

第 9 部分的 `candidate_ranking.csv` 是一个“混合候选表”，里面同时放了：

- Transformer unconditioned candidates
- Transformer conditioned candidates
- Markov unconditioned baseline
- Markov conditioned baseline

这些候选的指标列并不完全相同，所以 Pandas 会把所有可能出现的列合并成一个大表。某些列对某类候选不适用，就会显示为 `NaN`。

最容易混淆的是三类分数字段：

| 字段 | 用途 | 为什么会有 NaN |
| --- | --- | --- |
| `score` | unconditioned 候选或普通 MIDI 的直接评分 | conditioned Transformer 主要按 continuation-only 评分，所以 full row 的 `score` 可能为空 |
| `continuation_score` | conditioned Transformer 的 continuation-only 评分 | unconditioned 候选没有真实 prefix / continuation split，所以这一列必然是 NaN |
| `rank_score` | 最终排序统一使用的分数 | 这是最应该看的列；unconditioned 用 `score`，conditioned 用 `continuation_score` |

因此，第 9 部分里“很多 score 是 NaN”通常不是错误，而是因为当前表格把不同任务的候选放在一起展示：

- unconditioned rows: `score` 和 `rank_score` 有值，`continuation_*` 列为 NaN。
- conditioned Transformer rows: `continuation_score` 和 `rank_score` 有值，某些 full-level `score` 列可能为 NaN。
- Markov rows: 没有 Transformer 的 prefix split metadata，所以很多 `continuation_*` 列为 NaN。

实际选择候选时，应该看：

```text
rank_score
rank_reject_reason
usable
```

对于 conditioned Transformer，还要看：

```text
continuation_note_count
continuation_duration_seconds
continuation_notes_per_second
continuation_reject_reason
continuation_usable
continuation_score
```

当前第 11 部分 baseline comparison 已经用 continuation-only 指标来判断 Transformer conditioned 的 `usable`，避免了 NaN reject reason 被误判为 unusable。

---

## 1. Overview and Task Definitions

### 这一节在做什么

这一节是 notebook 的实验摘要。它说明 Assignment 2 做两个 symbolic MIDI generation 任务：

1. Task 1: symbolic unconditioned generation  
   模型从开始 seed 直接生成新的 MIDI，不依赖用户给定旋律。

2. Task 2: symbolic conditioned continuation  
   模型拿到真实 MIDI prefix，然后生成 continuation。评估时重点看 generated continuation，而不是 prefix + continuation 的整体听感。

### 为什么这样写

之前的 MAESTRO 实验发现一个关键问题：如果保存的是 full-with-prefix MIDI，听感可能被真实 prefix 抬高，导致误以为模型生成很好。因此最终 notebook 一开始就明确：

- conditioned task 必须评估 continuation-only。
- 最终路线切到 Nottingham。
- Transformer 是 scratch initialized。
- Markov 是 baseline，不是最终主模型。

### 输出说明

这一节主要是 Markdown 输出，没有代码执行结果。它的意义是给报告读者一个清晰实验边界：最终结果来自 Nottingham + density-conditioned Transformer，而不是 MAESTRO 或预训练模型。

---

## 2. Reproducible Setup

### 这一节在做什么

这一节负责导入库、定位项目根目录、设置输出路径、设置随机种子、显示运行环境，并定义 run switches。

### 为什么这样写

实验 notebook 必须能复现。路径、seed、输出目录和环境信息都集中在开头，可以避免后面代码散落硬编码路径，也方便检查是否误运行了 submission/export。

### 主要变量解释

| 变量 | 作用 |
| --- | --- |
| `PROJECT_ROOT` | 自动定位项目根目录 |
| `DATA_DIR` | Nottingham MIDI 数据目录 |
| `FINAL_ROOT` | 最终输出根目录 `outputs/final` |
| `CHECKPOINT_DIR` | Transformer checkpoint 保存位置 |
| `METRICS_DIR` | training history 和 summary 保存位置 |
| `CANDIDATE_DIR` | final MIDI candidates 保存位置 |
| `RAW_CANDIDATE_DIR` | raw candidate pool 保存位置 |
| `EVALUATION_DIR` | candidate ranking / selected candidates 保存位置 |
| `FIGURE_DIR` | 所有图像保存位置 |
| `RUN_DATA_ANALYSIS` | 是否运行数据分析 |
| `RUN_MARKOV_BASELINE` | 是否生成 Markov baseline |
| `RUN_TRANSFORMER_TRAINING` | 是否训练 Transformer |
| `RUN_GENERATION` | 是否生成 candidates |
| `RUN_EVALUATION` | 是否评估 candidates |
| `CREATE_SUBMISSION_FILES` | 是否创建 submission，默认 `False` |
| `EXPORT_HTML` | 是否导出 HTML，默认 `False` |
| `VERBOSE_TRAINING` | 是否打印详细训练日志，默认 `False` |

### 函数解释

#### `find_project_root(start: Path) -> Path`

从当前路径向上查找项目根目录。判断依据是是否存在：

```text
data/nottingham-dataset-master/MIDI
```

这样 notebook 不要求用户必须从某个固定目录打开，只要在 repo 内部运行，就能找到根路径。

#### `relpath(path: Path) -> str`

把绝对路径转换成相对于 `PROJECT_ROOT` 的路径。输出 CSV 和表格时使用相对路径更简洁，也更适合报告展示。

### 输出说明

| 输出 | 说明 |
| --- | --- |
| `Project root: ...` | 当前项目根目录，检查路径定位是否正确 |
| `Python: ...` | Python 版本，用于复现 |
| `Platform: ...` | 操作系统信息 |
| `Torch: ...` | PyTorch 版本 |
| `CUDA available: ...` | 是否可用 GPU |
| `GPU: ...` | GPU 型号 |
| `Nottingham data: ... exists=True` | 确认 Nottingham 数据存在 |
| `Submission disabled: True` | 确认不会创建 submission |
| `HTML export disabled: True` | 确认不会导出 HTML |

---

## 3. Dataset: Nottingham

### 这一节在做什么

这一节读取 Nottingham MIDI 文件并统计数据集性质，包括：

- 文件数
- note count
- duration
- notes per second
- pitch range
- pitch class distribution

### 为什么这样写

生成模型训练前必须知道数据集的真实分布。之前模型出现过“时间很长但有效音符很少”的问题，所以这里特别统计 note density，也就是 `notes_per_second`。

Nottingham 比 MAESTRO 更适合最终路线的原因：

- 曲目更短。
- symbolic pattern 更规整。
- 不像 MAESTRO 那样有复杂 expressive piano timing。
- 更适合稳定做 Task 1 unconditioned generation。

### 函数解释

#### `discover_midi_files(root: Path) -> list[Path]`

递归扫描 `root` 下所有 `.mid` 和 `.midi` 文件。它是数据入口函数。

#### `read_note_events(path: Path) -> tuple[list[dict[str, int]], int]`

用 `mido` 解析 MIDI 文件，抽取 note events。它处理：

- `note_on velocity > 0` 作为 note start。
- `note_off` 或 `note_on velocity == 0` 作为 note end。
- 同一 pitch/channel 上可能同时有多个 active note，所以用 `active` 队列记录。

返回：

- `notes`: 每个元素包含 `start`, `end`, `pitch`, `velocity`, `channel`
- `ticks_per_beat`: MIDI tick 到时间的转换基准

这个函数是所有 MIDI 统计和评估的基础。

#### `midi_statistics(path: Path) -> dict[str, object]`

调用 `read_note_events` 后计算单个 MIDI 的统计指标：

- `note_count`
- `duration_seconds`
- `notes_per_second`
- `pitch_min`
- `pitch_max`
- `pitch_range`
- `unique_pitch_count`
- `max_gap_seconds`

如果 MIDI 无法解析或没有 notes，会返回 `valid=False`。

#### `save_and_show_hist(series, filename, title, xlabel, bins=40)`

画直方图，同时：

1. 在 notebook 中 `plt.show()` 显示。
2. 保存到 `outputs/final/figures/`。

这样 notebook 输出和文件 artifact 保持一致。

### 表格输出说明

| 输出 | 当前意义 |
| --- | --- |
| `MIDI file count: 3089` | 当前 Nottingham 目录共发现 3089 个 MIDI 文件 |
| `sample_file` table | 展示前几个 MIDI 路径，证明数据扫描正常 |
| `describe()` statistics | 展示 note count、duration、density、pitch range 的均值/分位数/最大最小值 |
| `basic_dataset_summary` | 数据集核心摘要 |

当前 `dataset_summary.csv` 中关键数值：

| 指标 | 当前值 |
| --- | ---: |
| `num_files` | 3089 |
| `total_notes` | 712532 |
| `mean_duration` | 67.41s |
| `median_duration` | 64.00s |
| `mean_note_count` | 230.67 |
| `median_note_count` | 180.00 |
| `mean_notes_per_second` | 3.50 |
| `median_notes_per_second` | 3.15 |

### 图像输出说明

| 图像 | 文件 | 说明 |
| --- | --- | --- |
| Duration histogram | `outputs/final/figures/duration_histogram.png` | 曲目时长分布，用来判断生成目标时长是否合理 |
| Note count histogram | `outputs/final/figures/note_count_histogram.png` | 每首曲子的音符数量分布，用来判断候选是否过稀疏 |
| Note density histogram | `outputs/final/figures/note_density_histogram.png` | `notes_per_second` 分布，是 density-conditioned training 的依据 |
| Pitch class histogram | `outputs/final/figures/pitch_class_histogram.png` | 12 个 pitch class 的出现次数，用于检查音高覆盖 |

---

## 4. MIDI Preprocessing and Tokenization

### 这一节在做什么

这一节把 MIDI 转换成 REMI-style token ids，并构造语言模型训练窗口。

### 为什么这样写

Transformer 不能直接读取 MIDI 文件，需要离散 token 序列。REMI tokenization 会把 symbolic music 拆成事件：

- `Bar`: 小节推进
- `Position`: 小节内位置
- `Pitch`: 音高
- `Velocity`: 力度
- `Duration`: 时值

之前的失败模式是模型生成很多时间结构 token，但没有稳定生成足够完整的 note events。因此 notebook 加入 density label 和 density control token。

### 类和函数解释

#### `NotebookREMITokenizer`

这是对 MidiTok `REMI` tokenizer 的轻量封装，使 notebook 中 tokenization / detokenization 逻辑更清晰。

方法：

| 方法 | 作用 |
| --- | --- |
| `__init__(num_velocities=8, vocab_size=512)` | 创建 REMI tokenizer，并设置力度分桶和目标 vocab size |
| `fit(files)` | 用 Nottingham MIDI 文件训练 tokenizer vocabulary |
| `vocab_size` | 返回 tokenizer 词表大小 |
| `encode_file(path)` | 把一个 MIDI 文件编码成 token id list |
| `decode_ids(ids, output_path)` | 把 token ids 解码回 MIDI |
| `token_label(token_id)` | 把 token id 转换成人类可读 token label |

#### `token_type_counts(tokenizer, ids)`

统计一个 token 序列里不同事件类型的数量：

- Pitch
- Bar
- Position
- Velocity
- Duration
- Other

同时计算：

- `pitch_per_100_tokens`
- `pitch_ratio`

这个函数用于估计窗口中实际 note event 的密度。

#### `density_label(tokenizer, ids)`

根据 pitch token 数量和比例，把窗口分为：

- `low`
- `med`
- `high`

逻辑直观含义：

- pitch 太少或比例太低 -> `low`
- pitch 比例高 -> `high`
- 其他 -> `med`

#### `make_lm_windows(sequences, block_size, stride)`

把长 token sequence 切成 language-model windows。每个 window 长度最多是 `block_size + 1`，因为训练时：

- 输入是前 `block_size` 个 token。
- label 是后移一位的 `block_size` 个 token。

`stride=256` 控制窗口重叠程度。

#### `split_files(files, valid_fraction=0.1)`

把文件列表切成 train / validation split。当前按末尾约 10% 做 validation。

### 输出说明

| 输出 | 当前意义 |
| --- | --- |
| `token_length_histogram.png` | 每首 MIDI 被编码后的 token 长度分布 |
| `density_bucket_bar.png` | low / med / high density windows 数量 |
| `example_file` table | 展示一个样例 MIDI 的 token 长度、window 长度、density label |
| `First token ids` | 展示 token ids，确认 tokenizer 正常工作 |
| `First token labels` | 展示 ids 对应的 REMI token label |
| `dataset_summary` table | 保存和展示完整数据集/token/window summary |

当前 `dataset_summary.csv` 中 token/window 相关数值：

| 指标 | 当前值 |
| --- | ---: |
| `mean_token_length` | 222.21 |
| `density_low` | 1143 |
| `density_med` | 637 |
| `density_high` | 1425 |
| `train_file_count` | 2780 |
| `valid_file_count` | 309 |
| `train_window_count` | 2889 |
| `valid_window_count` | 316 |
| `token_min` | 13 |
| `token_max` | 3465 |

---

## 5. Markov / n-gram Baseline

### 这一节在做什么

这一节训练一个简单 n-gram Markov baseline，并生成：

- `outputs/final/candidates/markov_unconditioned.mid`
- `outputs/final/candidates/markov_conditioned.mid`

### 为什么这样写

Markov baseline 是一个可解释的简单模型。它不理解长程结构，但可以作为 sanity check：

- 如果 Transformer 连 Markov 都不如，说明训练或 sampling 可能有问题。
- 如果 Transformer 更稳定、更自然，就说明 neural model 学到了更强结构。

### 类和函数解释

#### `MarkovNGram`

一个 token-level n-gram 模型。

方法：

| 方法 | 作用 |
| --- | --- |
| `__init__(order=3, seed=253)` | 设置 n-gram 阶数和随机种子 |
| `fit(sequences)` | 从训练 windows 中统计 context -> next token 的转移表 |
| `_draw(counts)` | 根据计数分布随机抽样下一个 token |
| `sample(max_tokens, prefix=None)` | 无条件或有 prefix 条件下生成 token 序列 |

#### `decode_to_midi(ids, path)`

把 token ids 解码成 MIDI，并检查解码后是否包含 notes。用于 Markov 和部分 Transformer 输出保存。

### 输出说明

| 输出 | 说明 |
| --- | --- |
| `Saved Markov baseline MIDI files.` | Markov baseline MIDI 已经生成并保存 |
| `markov_unconditioned.mid` | Markov Task 1 baseline |
| `markov_conditioned.mid` | Markov Task 2 baseline |

---

## 6. Density-conditioned GPT2-style Transformer

### 这一节在做什么

这一节构建最终主模型：一个从零初始化的 GPT2-style causal Transformer。

### 为什么这样写

Transformer 做 next-token prediction，适合 symbolic music sequence modeling。为了修复“长但稀疏”的生成问题，notebook 做了两件事：

1. 每个 window 前面加 density control token。
2. 训练窗口按目标比例重采样：low 10%、med 60%、high 30%。

这样模型在生成时可以用 `med` 或 `high` 控制 token，倾向生成更合理密度。

### 函数解释

#### `controlled_window(window, label)`

把 density control token 放到 window 开头。例如：

```text
[CONTROL_TOKEN_IDS["med"], token1, token2, ...]
```

#### `density_balanced_windows(windows, balance=True)`

先用 `density_label` 给每个 window 分类，然后按照目标比例重新采样。返回：

- 加了 control token 的 windows。
- density summary。

这样做是为了避免训练集低密度/高密度不平衡导致模型偏向稀疏输出。

#### `make_batches(windows, block_size, batch_size)`

把 token windows 转成 PyTorch batch。每个样本：

- `inputs = window[:-1]`
- `labels = window[1:]`

这是标准 causal language modeling 训练方式。

#### `token_weight_vector()`

构造每个 token 的 loss weight。Pitch token 权重大一些，Velocity/Duration 次之。原因是 note event 的完整性比单纯时间推进更重要。

#### `weighted_ce_loss(logits, labels, weights)`

计算加权 cross entropy。普通 CE 会平均对待所有 token；这里给 note event token 更大权重，减少模型只学会 Bar/Position 时间结构而忽略 note 的风险。

### 模型配置说明

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `BLOCK_SIZE` | 512 | context length |
| `BATCH_SIZE` | 16 | batch size |
| `N_EMBD` | 256 | embedding dimension |
| `N_LAYER` | 4 | Transformer layers |
| `N_HEAD` | 4 | attention heads |
| `DROPOUT` | 0.1 | dropout |
| `LR` | 0.0003 | learning rate |
| `WEIGHT_DECAY` | 0.01 | weight decay |
| `GRAD_CLIP` | 1.0 | gradient clipping |
| `MAX_STEPS_FULL` | 12000 | final training steps |

### 输出说明

| 输出 | 当前意义 |
| --- | --- |
| `Control token ids: {'low': 512, 'med': 513, 'high': 514}` | 三个 density token 被加在原始 REMI vocab 后面 |
| `Train density summary` | 训练窗口原始分布和采样后分布 |
| `Validation density summary` | 验证窗口分布，不做重采样 |
| `Transformer initialized from scratch.` | 明确没有加载预训练权重 |
| `Parameter count: 3422464` | 模型参数量 |

当前 summary 中：

- `base_vocab_size = 512`
- `vocab_size = 515`
- `steps_completed = 12000`
- `best_step = 5500`
- `best_valid_loss = 0.7218`
- `best_valid_perplexity = 2.0582`

---

## 7. Training Curves and Checkpoint Selection

### 这一节在做什么

这一节训练 Transformer、保存 best checkpoint、保存 training history，并展示训练曲线。

### 为什么这样写

最终生成不一定使用最后一步 checkpoint，而是使用 validation loss 最好的 checkpoint。这样可以避免训练后期过拟合。

当前结果中 best checkpoint 出现在 step 5500，而训练总步数到 12000。这说明后期 train loss 继续下降，但 validation loss 不一定继续改善。

### 函数解释

#### `evaluate_model_loss(model, windows)`

在 validation windows 上计算平均 weighted CE loss。它会：

1. 切到 eval mode。
2. 禁用 gradient。
3. 遍历 validation batches。
4. 返回平均 loss。
5. 再切回 train mode。

### 输出说明

| 输出 | 说明 |
| --- | --- |
| training history head | 展示训练初期 loss，检查是否正常下降 |
| training history tail | 展示训练末期 loss，检查是否稳定 |
| best validation row | 展示 validation loss 最低的 checkpoint step |
| `loss_curve.png` | train loss 和 validation loss 曲线 |
| summary table | 展示模型参数、训练步数、best step、best loss、runtime |

当前训练曲线的意义：

- step 1 validation perplexity 很高，说明模型刚初始化。
- 前几百步 loss 快速下降，说明模型学到 token structure。
- best validation 在 step 5500 左右，说明 checkpoint selection 是必要的。
- 后期 train loss 继续下降而 valid loss 不同步下降，可能是轻微过拟合。

---

## 8. Candidate Generation

### 这一节在做什么

这一节用 best Transformer checkpoint 生成 candidate pool。

生成包括：

- Task 1 unconditioned candidates
- Task 2 conditioned candidates with multiple validation prefixes

### 为什么这样写

单次 sampling 很随机，不一定好听。candidate pool + ranking 更稳。Task 2 使用多个 validation prefixes，是为了避免某一个 prefix 太难或太简单导致结果偏差。

### 函数解释

#### `strip_control_tokens(ids)`

把 density control token 从 token sequence 中移除。因为 REMI tokenizer 不认识这些额外 control ids，decode 前必须去掉。

#### `sample_transformer(prefix, max_new_tokens, temperature, top_k)`

Transformer autoregressive sampling 函数。

流程：

1. 用 prefix 初始化 context。
2. 每一步取最后一个位置 logits。
3. 除以 temperature 控制随机性。
4. 用 top-k 限制候选 token。
5. 从概率分布中采样一个 token。
6. 加入序列，继续生成。

#### `write_note_events(path, notes, ticks_per_beat)`

把 note event list 写成 MIDI 文件。用于保存 continuation-only MIDI。

#### `crop_after(notes, split_tick)`

从 full-with-prefix notes 中裁掉 prefix，只保留 `split_tick` 之后的 notes，并把时间轴归零。这个函数是 conditioned continuation-only 诊断的核心。

#### `decode_candidate(ids, path)`

去掉 control token 后 decode 成 MIDI。用于保存 raw Transformer candidates。

### 输出说明

| 输出 | 说明 |
| --- | --- |
| `Generated candidate count: ...` | 实际成功 decode 的候选数量 |
| raw Transformer MIDI files | 保存在 `outputs/final/candidates/raw/`，用于后续 ranking |

生成参数的意义：

- `temperature`: 越高越随机。
- `top_k`: 限制每步只从概率最高的 k 个 token 中采样。
- `candidate_count`: 候选池越大，越容易筛到可听结果。
- `CONDITIONED_PREFIX_COUNT`: 使用多少个 validation prefixes。
- `CONDITIONED_CANDIDATES_PER_PREFIX`: 每个 prefix 生成多少 continuation。

---

## 9. Candidate Evaluation and Ranking

### 这一节在做什么

这一节评估所有候选 MIDI，应用 hard reject rules，计算 ranking score，保存：

- `outputs/final/evaluation/candidate_ranking.csv`
- `outputs/final/evaluation/selected_candidates.csv`
- `outputs/final/evaluation/candidate_diagnostics.csv`
- `outputs/final/evaluation/model_metrics.csv`

### 为什么这样写

MIDI 是否“可听”不能只看是否 decode 成功。必须检查：

- 是否太短
- 是否太长
- 是否音符太少
- 是否密度太低/太高
- 是否有巨大空白
- 是否 polyphony 异常
- 是否 pitch range 太窄
- 是否重复率太高

Task 2 必须用 continuation-only 评估，否则真实 prefix 会抬高 full-with-prefix 听感。

### 函数解释

#### `max_polyphony(notes)`

扫描 note start/end events，计算任意时刻最多同时响起多少 notes。用于发现异常堆叠音符。

#### `repetition_rate(notes)`

把 notes 按时间排序后统计 pitch bigram 的重复程度。重复率太高可能说明模型陷入循环。

#### `metric_from_notes(path, notes, ticks_per_beat)`

从 notes 直接计算候选指标：

- `valid`
- `note_count`
- `duration_seconds`
- `first_note_start`
- `active_duration`
- `notes_per_second`
- `notes_per_second_active`
- `max_gap`
- `pitch_range`
- `unique_pitch_count`
- `max_polyphony`
- `repetition`

#### `analyze_midi(path)`

读取 MIDI 文件并调用 `metric_from_notes`。如果读取失败，则返回 invalid row。

#### `reject_reason(row, min_notes=100)`

根据 hard reject rules 生成 reject reason 字符串。当前规则包括：

- invalid MIDI
- `note_count < min_notes`
- `duration_seconds < 25`
- `duration_seconds > 90`
- `notes_per_second < 1.5`
- `notes_per_second > 8`
- `max_gap > 4`
- `max_polyphony > 32`
- `pitch_range < 8`
- `repetition > 0.9`

#### `blank_reason(value)`

判断 reject reason 是否为空。它把 `NaN`、空字符串和 `"nan"` 都当成空 reason。这个函数解决了 CSV 读回时 blank cell 变成 NaN 的问题。

#### `normalized_reason(value)`

把 reject reason 标准化。如果是空 reason，返回空字符串；否则返回去掉前后空格的字符串。

#### `score_candidate(row)`

给候选打分。拒绝候选会给极低分；可用候选根据以下因素综合评分：

- duration 是否接近 50 秒
- notes_per_second 是否接近 3
- max_gap 是否过大
- note_count 是否足够

这个分数是自动筛选分数，不等于最终音乐审美。

### 表格输出说明

第一个表是 top ranked candidates。它从 `ranking_df.sort_values("rank_score", ascending=False).head(12)` 得到。

重要列：

| 列 | 说明 |
| --- | --- |
| `path` | raw candidate MIDI 路径 |
| `task` | unconditioned 或 conditioned |
| `model` | transformer 或 markov |
| `note_count` | full MIDI 音符数 |
| `duration_seconds` | full MIDI 时长 |
| `notes_per_second` | full MIDI 密度 |
| `reject_reason` | full MIDI reject reason |
| `score` | full MIDI score，主要用于 unconditioned |
| `rank_score` | 最终排序分数，最应该看这一列 |
| `rank_reject_reason` | 排名使用的 reject reason |
| `continuation_*` | conditioned continuation-only 指标 |

为什么很多 `score` / `continuation_score` 是 NaN：

- unconditioned 没有 continuation，所以 `continuation_score = NaN`。
- conditioned Transformer 用 continuation-only 评分，所以 full-level `score` 可能不是主要字段。
- 最终统一看 `rank_score`。

第二个表是 selected candidates。它只包含最终选中的 Transformer candidates：

| row | 说明 |
| --- | --- |
| unconditioned selected | 被复制为 `outputs/final/candidates/symbolic_unconditioned.mid` |
| conditioned selected | 被复制为 `outputs/final/candidates/symbolic_conditioned.mid`，同时保存 prefix/full/continuation split |

当前 selected 结果：

| task | selected path | 关键指标 |
| --- | --- | --- |
| Task 1 | `outputs/final/candidates/symbolic_unconditioned.mid` | 159 notes, 51.5s, 3.09 notes/s |
| Task 2 | `outputs/final/candidates/symbolic_conditioned.mid` | continuation-only: 276 notes, 73.0s, 3.78 notes/s |

### 图像输出说明

| 图像 | 文件 | 意义 |
| --- | --- | --- |
| Candidate reject reason bar | `candidate_reject_reason_bar.png` | 显示候选被 reject 的主要原因，例如时长过长、密度过低、重复过高 |
| Usable vs rejected bar | `candidate_usable_vs_rejected.png` | 显示候选池里 usable 和 rejected 的比例 |
| Candidate metric comparison | `candidate_metric_comparison.png` | 对 top candidates 的 note_count、duration、density、gap、polyphony 做对比 |

第 9 部分的核心意义是：自动筛掉明显不可用 MIDI，把最终试听范围缩小到更合理的 candidates。

---

## 10. Results Section Placeholder

### 这一节在做什么

这一节动态读取 evaluation outputs，而不是写死旧结果。

读取文件：

- `selected_candidates.csv`
- `model_metrics.csv`
- `candidate_diagnostics.csv`

### 为什么这样写

notebook 可能会重新运行。结果 section 必须显示当前 run 的结果，而不是历史实验数字。

### 输出说明

| 输出 | 说明 |
| --- | --- |
| selected candidates table | 当前 run 选出的 final candidates |
| model metrics table | best checkpoint 和 validation metrics |
| diagnostics groupby table | 每个 model/task 的候选数量 |

---

## 11. Baseline Comparison

### 这一节在做什么

这一节比较：

- Markov unconditioned
- Markov conditioned
- Transformer unconditioned
- Transformer conditioned

### 为什么这样写

最终报告需要说明 Transformer 是否优于简单统计 baseline。这里比较的不是所有候选，而是每个 model/task 里排名最高的候选。

### 函数解释

#### `metric_value(row, preferred, fallback)`

如果 `preferred` 列是 NaN，就回退到 `fallback` 列。它主要用于 conditioned Transformer：

- 优先用 `continuation_note_count`
- 如果不存在，再用 `note_count`

### 输出说明

Baseline comparison table 的列：

| 列 | 说明 |
| --- | --- |
| `model` | markov 或 transformer |
| `task` | unconditioned 或 conditioned |
| `note_count` | 音符数；Transformer conditioned 优先使用 continuation-only |
| `duration_seconds` | 时长；Transformer conditioned 优先使用 continuation-only |
| `notes_per_second` | 密度；Transformer conditioned 优先使用 continuation-only |
| `max_gap` | 最大音符间隔 |
| `max_polyphony` | 最大同时发声音符数 |
| `usable` | 是否通过 hard reject |
| `comment` | baseline 或 main model |

当前输出说明：

- Markov 两行 `usable=False`，因为 duration / gap 等指标不满足 hard reject。
- Transformer unconditioned `usable=True`。
- Transformer conditioned `usable=True`，并且使用 continuation-only 指标判断。

---

## 12. Failure Analysis

### 这一节在做什么

总结为什么最终从 MAESTRO 切回 Nottingham，以及为什么加入 density conditioning。

### 关键结论

- MAESTRO 是更难的 expressive piano dataset。
- full-with-prefix 听感可能误导，因为真实 prefix 本身已经好听。
- continuation-only diagnostics 显示之前生成有稀疏问题。
- Nottingham 更适合最终 symbolic generation route。
- density-conditioned training 直接针对 long-duration / low-note-density failure。

### 输出说明

这一节是 Markdown 分析，没有代码输出。它的作用是解释实验路线选择，而不是展示新结果。

---

## 13. Final Selected Outputs

### 这一节在做什么

这一节列出最终 assignment 对应的两个 MIDI 输出，并检查相关文件是否存在。

### 为什么这样写

最终 submission 只需要：

- `symbolic_unconditioned.mid`
- `symbolic_conditioned.mid`

但 notebook 还保留 Markov baseline 和 conditioned split files，用于对照和诊断。

### 输出说明

Final mapping table：

| task | final submission filename | source path | model | generation mode | notes |
| --- | --- | --- | --- | --- | --- |
| Task 1 | `symbolic_unconditioned.mid` | `outputs/final/candidates/symbolic_unconditioned.mid` | density-conditioned Transformer | BOS / fully generated | no real primer used |
| Task 2 | `symbolic_conditioned.mid` | `outputs/final/candidates/symbolic_conditioned.mid` | density-conditioned Transformer | conditioned continuation | evaluated using continuation-only |

File existence output：

| 文件 | 作用 |
| --- | --- |
| `symbolic_unconditioned.mid` | Task 1 final selected Transformer output |
| `symbolic_conditioned.mid` | Task 2 final selected Transformer output |
| `markov_unconditioned.mid` | Task 1 baseline |
| `markov_conditioned.mid` | Task 2 baseline |
| `conditioned_prefix_only.mid` | Task 2 real prefix only |
| `conditioned_full_with_prefix.mid` | prefix + generated continuation |
| `conditioned_continuation_only.mid` | generated continuation only |

---

## Files to Listen To

### 这一节在做什么

这一节把试听路径集中列出来。

### 输出说明

| role | path | 意义 |
| --- | --- | --- |
| final selected Task 1 | `outputs/final/candidates/symbolic_unconditioned.mid` | 最终 Task 1 unconditioned result |
| final selected Task 2 | `outputs/final/candidates/symbolic_conditioned.mid` | 最终 Task 2 conditioned full output |
| baseline Task 1 | `outputs/final/candidates/markov_unconditioned.mid` | Markov unconditioned baseline |
| baseline Task 2 | `outputs/final/candidates/markov_conditioned.mid` | Markov conditioned baseline |
| conditioned diagnostic | `outputs/final/candidates/conditioned_prefix_only.mid` | 真实 prefix，只用于诊断 |
| conditioned diagnostic | `outputs/final/candidates/conditioned_full_with_prefix.mid` | prefix + generated continuation |
| conditioned diagnostic | `outputs/final/candidates/conditioned_continuation_only.mid` | generated continuation only，最能说明 Task 2 生成质量 |

---

## 14. Submission Checklist

### 这一节在做什么

列出最终提交需要的文件，但不执行复制、不创建 submission、不导出 HTML。

### 为什么这样写

这样 notebook 可以安全 review 和复现实验，不会误创建 submission artifacts。

### 输出说明

| 输出 | 说明 |
| --- | --- |
| `Submission creation disabled: True` | notebook 当前不会创建 submission |
| `HTML export disabled: True` | notebook 当前不会导出 HTML |

最终手动提交时需要：

- `workbook.html`
- `video_url.txt`
- `symbolic_unconditioned.mid`
- `symbolic_conditioned.mid`

---

## 主要文件输出总览

### Evaluation tables

| 文件 | 内容 |
| --- | --- |
| `outputs/final/evaluation/dataset_summary.csv` | 数据集、token、window、density summary |
| `outputs/final/evaluation/candidate_ranking.csv` | 所有候选的完整评估与排序 |
| `outputs/final/evaluation/selected_candidates.csv` | 最终选中的 Transformer candidates |
| `outputs/final/evaluation/candidate_diagnostics.csv` | candidate ranking 的诊断副本 |
| `outputs/final/evaluation/model_metrics.csv` | best checkpoint 和 validation metrics |

### Figure outputs

| 文件 | 说明 |
| --- | --- |
| `outputs/final/figures/duration_histogram.png` | 数据集时长分布 |
| `outputs/final/figures/note_count_histogram.png` | 数据集 note count 分布 |
| `outputs/final/figures/note_density_histogram.png` | 数据集 notes per second 分布 |
| `outputs/final/figures/token_length_histogram.png` | token length 分布 |
| `outputs/final/figures/pitch_class_histogram.png` | pitch class 分布 |
| `outputs/final/figures/density_bucket_bar.png` | low/med/high density bucket 分布 |
| `outputs/final/figures/loss_curve.png` | training/validation loss curve |
| `outputs/final/figures/candidate_reject_reason_bar.png` | reject reason 分布 |
| `outputs/final/figures/candidate_usable_vs_rejected.png` | usable/rejected 数量 |
| `outputs/final/figures/candidate_metric_comparison.png` | top candidates 指标对比 |

### Candidate MIDI outputs

| 文件 | 说明 |
| --- | --- |
| `outputs/final/candidates/symbolic_unconditioned.mid` | Task 1 final Transformer candidate |
| `outputs/final/candidates/symbolic_conditioned.mid` | Task 2 final Transformer candidate |
| `outputs/final/candidates/markov_unconditioned.mid` | Markov Task 1 baseline |
| `outputs/final/candidates/markov_conditioned.mid` | Markov Task 2 baseline |
| `outputs/final/candidates/conditioned_prefix_only.mid` | conditioned prefix only |
| `outputs/final/candidates/conditioned_full_with_prefix.mid` | conditioned full with prefix |
| `outputs/final/candidates/conditioned_continuation_only.mid` | conditioned generated continuation only |

---

## 最后建议：第 9 部分表格如何阅读

第 9 部分的大表很宽，容易误读。建议阅读顺序：

1. 先看 `rank_reject_reason` 是否为空。
2. 再看 `rank_score`。
3. 对 unconditioned，看 `note_count`, `duration_seconds`, `notes_per_second`, `max_gap`。
4. 对 conditioned Transformer，看 `continuation_note_count`, `continuation_duration_seconds`, `continuation_notes_per_second`, `continuation_max_gap`, `continuation_score`。
5. 不要把非适用列里的 NaN 当成失败。例如 unconditioned 的 `continuation_score` 是 NaN 是正常的。

如果想让 notebook 展示更清晰，可以在第 9 部分只显示一个精选列集合，例如：

```python
display_cols = [
    "path", "task", "model",
    "note_count", "duration_seconds", "notes_per_second",
    "continuation_note_count", "continuation_duration_seconds", "continuation_notes_per_second",
    "rank_score", "rank_reject_reason",
]
display(ranking_df.sort_values("rank_score", ascending=False)[display_cols].head(12))
```

这不会改变评估逻辑，只会让表格更容易读。
