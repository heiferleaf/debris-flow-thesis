"""Export a public, metadata-only experiment archive without changing source Git.

Only final run allowlists below are exported. Small head weights require a saved
successful replay proof. Large model weights are indexed, never silently claimed
to have been re-executed. The export is a separate Git publication tree.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT = {'.py', '.md', '.json', '.jsonl', '.csv', '.txt', '.log', '.sh', '.js', '.html', '.css', '.example', '.snapshot', '.yaml', '.yml', '.toml'}
MAX_TEXT = 2 * 1024 * 1024
SKIP_DIRS = {'__pycache__', '.git', '.wb-thumbs', '.cache', 'node_modules', 'frames', 'video_frames', 'video_windows', 'pixels', 'images', 'thumbnails', 'qc', 'anchors'}

# title, status, data / model / split, conclusion, selected final run directories
SPECS = {
 '003': ('单帧迁移试点', '历史试点；原始指标文件缺失', '独立图片；pHash簇分割；冻结ConvNeXt-T及线性头。', '只保留代码与原卡片记录，不把缺失原始指标的叙述标为本轮已验证成绩。', ['convnext/EXP-003']),
 '004': ('外部负图片及本地冒烟', '准备与冒烟；无独立正式成绩', '独立图片及外部负图片；ConvNeXt冻结特征。', '本卡未形成独立正式训练结果；不上传冒烟成绩充当成果。', []),
 '005': ('单帧正式基线', '已运行；原始指标文件缺失', '正9347/负925为审计池；簇级70/15/15且每簇限额后实际1729张。冻结ConvNeXt-T+768→2线性头，类加权CE、AdamW、val F1选轮。', '原卡记录test F1=0.9306；原始metrics/report未留存，本次不冒称已从原始预测验证该数字。', ['convnext/EXP-005-final']),
 '006': ('二维CNN与时序头', '最终修订运行已完成', '冻结ConvNeXt逐帧特征；LSTM/TCN/Attention、TSN、单帧投票及融合。按场次/旧场景名分组五折，另有留一组分析；16帧输入。', '最终地址映射与尾窗修订版作为归档；换场景存在反向折。旧机位命名划分不能自动等同当前实际物理场景隔离。', ['temporal/EXP-006-20260924-addrmap-session-pred', 'temporal/EXP-006-20260924-addrmap-scene-pred', 'temporal/EXP-006-20260924-addrmap-scene-logo']),
 '008': ('论文三项MIL损失重实现', '最终150条重跑及头权重回放通过', '150条视频=113P/37N；每视频48片段×16帧。冻结K400 R(2+1)D-18→512维→四层MLP。弱标签MIL三项损失；另有片段BCE对照。场次五折及旧划分敏感性。', 'MIL full逐折均值AUC=0.9412；MIL strict=0.8699。权重与原预测复算一致。验证折选轮，属于开发成绩；并非原论文同架构、同数据的数值复现。', ['exp008/mil-v2-150-full', 'exp008/mil-v2-150-strict', 'exp008/mil-v2-150-full-iid', 'exp008/mil-v2-150-strict-iid', 'exp008/mil-v2-150-strict-scene', 'exp008/strong-v2-150-strict', 'exp008/strong-v2-150-loose', 'exp008/strong-v2-150-strict-iid', 'exp008/strong-v2-150-strict-scene']),
 '009': ('强标签全参数微调', '最终无增强运行已完成', '同150条旧标签；原生连续16帧；R(2+1)D-18及深MLP全参数微调，片段BCE、场次五折。', '最终开发AUC=0.9909；验证折选最好轮次。历史执行脚本原件缺失，当前源码不能称为历史字节级复现。只归档最终noaug运行。', ['exp008/EXP-009-strong-v2-150-strict-ft-noaug']),
 '010': ('图像复核级联', '未训练草案', '拟补论文的图像复核级联；未形成新模型。', '无结果，不上传虚构成绩。', []),
 '011': ('弱标签负包超阈惩罚', '有效阴性实验；头权重回放通过', '150条同冻结特征、strict掩码，seed7/8/9；三项MIL与增加负包惩罚对比。', '减少误报同时显著增加漏报，未满足预定目标；阴性结果属于有效实验。', []),
 '012': ('强标签整视频上下文', '有效中性实验；头权重回放通过', '150条冻结特征；深MLP、gated/selfattn上下文与惩罚的析因对照；三种子五折。', '没有稳定跨种子优势；上下文聚合不等于已证明学习了物理时序。', []),
 '013': ('历史全量训练设想', '未执行草案', '旧固定2轮最终训练设想。', '没有该卡训练结果；不作为最终模型。', []),
 '014': ('增大帧间隔', '最终对照已完成', '128条共同支持池；16帧0.5秒间隔、跨度7.5秒，与短输入配方比较；强标签全参数微调及五折。', '两设定开发AUC均记0.9923；倒序敏感性增加不等于识别能力提高。采样落点和可用片段也改变，不能仅归因时间跨度。', ['exp008/EXP-014-strong-v2-128-strict-ft-noaug', 'exp008/EXP-014-strong-v3-128-strict-ft-noaug']),
 '015': ('原间隔重采样草案', '未执行草案', '旧150条重采样方案。', '无训练结果。', []),
 '016': ('RGB与帧差', '双种子开发实验完成', '150条开发池；单角色119训练/31开发评价；3秒16帧，六通道[R,0]与[R,D]，R(2+1)D及深头，Adam，最大20轮按验证BCE选轮。', '帧差严格减少误报未跨种子复现；D置零诊断限选中权重的136片段。', ['exp016/paired-s7-20261030-v2', 'exp016/paired-s8-20261031-v1', 'exp016/seed7-seed8-comparison-v2', 'exp016/difference-dependency-s8-20261031-v2']),
 '017': ('RGB场次分组五折参照', '成功续跑后的五折完成', '150条、113P/37N；3秒16帧；R(2+1)D深头。训练侧inner选轮后fresh outer重训，外层留出各评价一次。', 'max/0.5下113TP/37FP/0FN/0TN；这不是可交付效果。保存成功父组件和续跑组件，不以失败父流水线充当完整成绩。', ['exp017/cv-s7-20261102-v1/fold-01', 'exp017/cv-s7-20261102-v2', 'exp017/report-cv-s7-20261102-v2-v1', 'exp017/audit-resume-cv-s7-20261102-v2-v2']),
 '018': ('正负配对排序约束', '三组匹配开发实验完成', '普通RGB、跨场次配对、同场次配对；相同骨干和深头，附加softplus(zN-zP)；两个已知开发角色。', '未达到改善目标；不晋级为有效改进。', ['exp018/paired-s7-20261101-v1']),
 '019': ('真实顺序与固定置换', '匹配开发实验完成', '保持16帧集合，真实顺序与固定置换分别重训；其余训练配方匹配。', '出现有限顺序依赖信号，不能直接证明理解物理运动；未据此选定3秒或7.5秒。', ['exp019/order-s7-20261101-v1', 'exp019/negative-alarm-analysis-s7-v1']),
 '020': ('简化分类头小试', '配对开发实验完成', '两个角色、seed7/8；深MLP与512→1线性头；RGB骨干及训练配方保持。', '线性头片段门槛4/4通过，支持进入后续五折；不是独立泛化成绩。', ['exp020/head-s7s8-20261103-v2', 'exp020/recovered-comparison-head-s7s8-20261103-v2-v1', 'exp020/report-head-s7s8-20261103-v2-v1', 'exp020/selected-logit-review-head-s7s8-20261103-v2-v1']),
 '021': ('局部时空能量', '有效阴性开发实验完成', '同RGB线性参照；无跨时项/完整局部能量增加112维线性后融合；相同角色和预算。', '片段门槛通过但视频错误未改善，原晋级条件失败；旧整条负标签与现有修订标签不混用。', ['exp021/energy-s7-20261103-v2', 'exp021/report-energy-s7-20261103-v2-v1', 'exp021/window-errors-energy-s7-20261103-v2-v1']),
 '022': ('线性头完整开发五折', '完整五折及结果复核完成', '150条113P/37N；与017同场次折、3秒16帧；512→1线性头，训练侧选轮后重训；视频max/0.5评价。', '109TP/21FP/4FN/16TN。比深头少16误报、多4漏报，未满足不增加漏报的预定条件。', ['exp022/linear-cv-s7-20261104-v1', 'exp022/report-linear-cv-s7-20261104-v1-v1']),
 '023': ('时间跨度受控对照', '未执行草案', '计划共同输入支持下比较时间跨度。', '没有本卡训练结果。', []),
 '024': ('连续两窗确认', '已有预测上的有限开发比较完成', '读取两个已知角色、42条不同视频的同一Linear分数；不重训，比较max与连续两窗规则。', '修复2条不同误报、未新增FN，宏FPR 0.75→0.5417；相对输出等待最大增加92.96秒，非真实检测延迟。', ['exp024/confirmation-s7-dev-v1']),
 '025': ('完整开发连续确认方案', '元数据覆盖检查；未完成全量评价', '拟复用022同一模型分数做完整五折判决比较，不另训模型。', '147/150为旧两窗支持覆盖核查，不是模型正确率；没有新的完整验证成绩。', []),
 '026': ('三状态与跨界软目标', '历史设计；未训练', '纯N/T/P窗与时间占比软目标的候选；语义不等同。', '未训练，不声称三分类有效。', []),
 '027': ('时间组成三锚点', '未采用设计；未训练', '三锚点T表示N/P时间组成，改变了视觉阶段T的意义。', '无结果，不作为已解决标签问题的成果。', []),
 '028': ('逐片段参照草案', '已暂停设计；未训练', '旧标签与旧机位角色安排已经暂停；同景多视角应同侧。', '不把撤回的U/A或旧角色安排作为现行训练合同。', []),
 '029': ('局部三类头与跨片段上下文', '当前设计；未训练', '150条新版N/P/T；共同冻结RGB特征，局部线性三类头对照双向GRU三类头；3秒网格中点监督，实际物理场景阻断；折数和成员未最终冻结。', '仅设计与元数据盘点，不存在新训练成绩或最终模型。', []),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def files(root):
    if not root.exists():
        return
    if root.is_file():
        yield root
        return
    for parent, dirs, names in os.walk(str(root), followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not Path(parent, d).is_symlink())
        for name in sorted(names):
            p = Path(parent, name)
            if not p.is_symlink():
                yield p


def sanitize(text):
    # Credential values only; source files remain untouched. Export hashes record changes.
    patterns = [
        r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{25,})\b',
        r'(?i)(?<=ftp://)[^\s/@:]+:[^\s/@]+(?=@)',
        r'(?i)(?<=Bearer )[A-Za-z0-9_.-]{25,}',
        r'(?i)(?<=[?&]token=)[A-Za-z0-9_.-]{16,}',
    ]
    changes = 0
    for pattern in patterns:
        text, n = re.subn(pattern, '[REDACTED]', text)
        changes += n
    # Literal secret assignments; leave placeholder/example/regex expressions alone.
    pattern = r'''(?im)((?:\b(?:[a-z0-9]+_)*(?:password|passwd|api_key|access_token|secret_key)\b|密码)["']?\s*[:=：]\s*["']?)([^\s"'<>`,;}{\]\[)]+)'''
    def redact(m):
        nonlocal changes
        value = m.group(2)
        if len(value) < 8 or re.search(r'(?i)example|placeholder|fixture|nonsecret|your|redact|none|false|true|os\.|getenv|input|password|token|api.key|^fake|^test|^\$|^%|^\\', value):
            return m.group(0)
        changes += 1
        return m.group(1) + '[REDACTED]'
    text = re.sub(pattern, redact, text)
    return text, changes


class Export:
    def __init__(self, root, destination):
        self.root, self.destination = root, destination
        self.manifest, self.omitted, self.metadata_refs = {}, {}, set()

    def put(self, path, scope):
        rel = path.relative_to(self.root).as_posix()
        if rel in self.manifest:
            return
        size = path.stat().st_size
        if path.suffix.lower() not in TEXT or size > MAX_TEXT:
            self.omitted[rel] = {'path': rel, 'bytes': size, 'reason': 'media/weight/feature or oversized derived table; not an uploaded input'}
            return
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            self.omitted[rel] = {'path': rel, 'bytes': size, 'reason': 'non-UTF8; not exported without a verified conversion'}
            return
        clean, redactions = sanitize(text)
        data = clean.encode('utf-8') if redactions else raw
        dest = self.destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        self.manifest[rel] = {'path': rel, 'bytes': len(data), 'source_sha256': sha(raw),
                              'export_sha256': sha(data), 'credential_redactions': redactions, 'scope': scope}
        normalized = re.sub(r'/+', '/', text.replace('\\', '/'))
        for value in re.findall(r'data/(?:metadata|process)/[^\s"\'`<>\[\]{},;()]+', normalized):
            value = value.rstrip('.:')
            if not re.search(r'/(?:frames|pixels|video_frames|video_windows|cache|secrets)(?:/|$)', value.lower()):
                self.metadata_refs.add(value)

    def metadata(self):
        visited = set()
        while self.metadata_refs - visited:
            pending = sorted(self.metadata_refs - visited)
            for rel in pending:
                visited.add(rel)
                if any(c in rel for c in '*?|') or len(rel) > 400:
                    continue
                p = (self.root / rel)
                if not p.is_file():
                    continue
                try:
                    p.resolve().relative_to(self.root)
                except ValueError:
                    continue
                # Only explicit files, never recursively traverse a referenced
                # data directory. Final input fingerprints identify exact files.
                candidates = [p]
                for source in candidates:
                    if source.suffix.lower() in TEXT and source.stat().st_size <= 128 * 1024:
                        self.put(source, 'input-metadata')
                    elif source.suffix == '.npz' and 'label' in source.name.lower() and source.stat().st_size <= MAX_TEXT:
                        self.metadata_arrays(source)

    def metadata_arrays(self, path):
        import numpy as np
        rel = path.relative_to(self.root).as_posix()
        if rel + '.metadata.json' in self.manifest:
            return
        with np.load(path, allow_pickle=True) as z:
            if any(re.search(r'(?i)feat|pixel|image|rgb|embedding', k) for k in z.files):
                return
            arrays = {k: {'dtype': str(z[k].dtype), 'shape': list(z[k].shape), 'values': z[k].tolist()} for k in z.files}
        data = (json.dumps({'original_path': rel, 'arrays': arrays}, ensure_ascii=False) + '\n').encode('utf-8')
        out = self.destination / (rel + '.metadata.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        self.manifest[rel + '.metadata.json'] = {'path': rel + '.metadata.json', 'source_path': rel,
             'source_sha256': sha(path.read_bytes()), 'export_sha256': sha(data), 'bytes': len(data),
             'scope': 'lossless label-array metadata conversion; not pixels/features'}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def finalize(out):
    """Deduplicate metadata, omit bulk per-epoch tensors, validate exported text."""
    import ast
    base = out / 'base'
    base_manifest = {x['path']: x for x in json.loads((base / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))}
    notices = []
    for folder in sorted((out / 'overlays').iterdir()):
        mp = folder / 'SOURCE_MANIFEST.json'
        records = json.loads(mp.read_text(encoding='utf-8'))
        omitted = json.loads((folder / 'ARTIFACT_INDEX.json').read_text(encoding='utf-8'))
        kept = []
        for item in records:
            rel = item['path']; p = folder / rel
            metadata = rel.startswith(('data/metadata/', 'data/process/'))
            bulk = (metadata and p.stat().st_size > 128 * 1024 and 'clip-labels.npz.metadata.json' not in rel)
            bulk |= bool(re.search(r'validation-epoch-\d+\.json$', rel))
            bulk |= (rel.startswith('data/results/') and p.suffix == '.json' and p.stat().st_size > 256 * 1024)
            old_narrative = rel.startswith('docs/process/experiments/') and '损失函数澄清' not in rel
            if bulk or old_narrative:
                omitted.append(dict(item, reason='bulk intermediate table or full historical narrative; final summary and run metrics retained'))
                p.unlink()
                continue
            if metadata:
                target = base / rel
                if rel in base_manifest:
                    assert sha(target.read_bytes()) == sha(p.read_bytes()), ('metadata conflict', rel)
                    p.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    p.replace(target)
                    base_manifest[rel] = item
                continue
            kept.append(item)
        dump(mp, kept)
        dump(folder / 'ARTIFACT_INDEX.json', omitted)
        info = json.loads((folder / 'EXPERIMENT.json').read_text(encoding='utf-8'))
        info['unuploaded_artifacts'] = omitted
        dump(folder / 'EXPERIMENT.json', info)
        num = info['experiment_id'][4:]
        entry = {'003':'experiments/common/convnext/run_pilot.py', '004':'experiments/common/convnext/smoke_d4.py',
                 '005':'experiments/common/convnext/run_full.py', '006':'experiments/common/temporal/train_temporal.py',
                 '008':'experiments/common/video/train_mil.py', '009':'experiments/common/video/train_ft.py',
                 '011':'experiments/common/video/train_mil.py', '012':'experiments/common/video/train_strong.py',
                 '014':'experiments/common/video/train_ft.py', '016':'experiments/runspecs/EXP-016/train.py',
                 '017':'experiments/runspecs/EXP-017/pipeline.py', '018':'experiments/runspecs/EXP-018/run.py',
                 '019':'experiments/runspecs/EXP-019/pipeline.py', '020':'experiments/runspecs/EXP-020/pipeline.py',
                 '021':'experiments/runspecs/EXP-021/pipeline.py', '022':'experiments/runspecs/EXP-022/pipeline.py',
                 '024':'experiments/runspecs/EXP-024/replay.py'}.get(num)
        code = '# 代码与工具\n\n'
        if entry:
            code += '主要入口：`' + entry + '`。\n\n'
        else:
            code += '本卡没有已完成训练入口，不为归档虚构train.py。设计文件见对应runspec目录（如存在）。\n\n'
        code += '共享骨干、损失、采样和评价：`experiments/common/`；可重用命令行工具：`experiments/tools/`；划分代码：`experiments/splits/code/`。保留所有共享源文件以覆盖跨实验依赖，分支README和EXPERIMENT.json限定本实验实际使用范围。\n\n当前迁移源码与历史执行代码的身份不同。只有run中实际保存的code快照及原指纹才能证明历史源码身份，不能把当前Git提交冒充旧训练提交。\n'
        (folder / 'CODE.md').write_text(code, encoding='utf-8')
        result = '# 最终结果\n\n' + info['status'] + '。\n\n' + info['conclusion'] + '\n\n'
        result += '## 正式run\n\n' + ('\n'.join('- `' + x + '`' for x in info['final_run_directories']) or '没有已完成的正式run。') + '\n\n'
        result += '上述目录保留原指标、预测及必要运行记录。SOURCE_MANIFEST.json区分原文件SHA256与发布文件SHA256；省略的中间表、未验证权重和大文件在ARTIFACT_INDEX.json登记。训练没有取得改进，仍可构成有效的阴性/中性结果。\n'
        (folder / 'RESULTS.md').write_text(result, encoding='utf-8')
        readme = folder / 'README.md'
        text = readme.read_text(encoding='utf-8').replace('## 从这里阅读\n\n', '## 从这里阅读\n\n- [代码入口与工具](CODE.md)\n- [最终结果](RESULTS.md)\n')
        if num == '008':
            text = text.replace('## 从这里阅读', '论文复现主版本请先读[主运行说明](PRIMARY_RUN.md)，其余运行是对照。\n\n## 从这里阅读')
        readme.write_text(text, encoding='utf-8')
    for rel, item in list(base_manifest.items()):
        p = base / rel
        if rel.startswith(('data/metadata/', 'data/process/')) and p.stat().st_size > 128 * 1024 and 'clip-labels.npz.metadata.json' not in rel:
            notices.append(dict(item, reason='bulk metadata; provenance hash retained instead of derived sample arrays'))
            p.unlink(); del base_manifest[rel]
    # The public exporter itself is the finalized version, not the earlier build copy.
    source = Path(__file__).resolve(); target = base / 'experiments/tools/export_experiment_archive.py'
    target.write_bytes(source.read_bytes())
    base_manifest['experiments/tools/export_experiment_archive.py'].update(source_sha256=sha(source.read_bytes()), export_sha256=sha(source.read_bytes()), bytes=source.stat().st_size)
    redacted = []
    for tree, manifest in [(base, base_manifest)] + [(f, {x['path']:x for x in json.loads((f/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))}) for f in sorted((out/'overlays').iterdir())]:
        for rel, item in manifest.items():
            p = tree / rel
            if p.suffix.lower() not in TEXT:
                continue
            text = p.read_text(encoding='utf-8-sig'); clean, n = sanitize(text)
            if n:
                p.write_text(clean, encoding='utf-8')
                item.update(export_sha256=sha(p.read_bytes()), bytes=p.stat().st_size,
                            credential_redactions=item.get('credential_redactions',0)+n)
                redacted.append({'path':rel,'count':n})
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=rel)
        dump(tree / 'SOURCE_MANIFEST.json', list(manifest.values()))
    primary = out/'overlays/EXP-008/PRIMARY_RUN.md'
    primary.write_text('''# 论文方法重实现：最终有效主版本

论文：Wu等（2025），Deep learning-based debris flow hazard detection and recognition system: a case study，Scientific Reports 15:6789。

## 代码与结构

1. `experiments/common/video/extract_feats.py`：冻结Kinetics400预训练R(2+1)D-18，提取每个16帧片段的512维特征。
2. `experiments/common/video/train_mil.py`：`MilHead`是512→128→64→32→1、ReLU/Dropout、末端Sigmoid；`mil_loss`实现三项损失。
3. 每条视频均匀取48个片段，分别评分，整条视频分数取最高值，0.5阈值输出正负。

原论文使用C3D及图像级复核级联；本仓库改变骨干并仅实现前两阶段，因此是方法重实现，不宣称复现其原始数字或得到作者原始代码。

## 三项损失

令正负视频片段分数为p、n，Rp=max(p)−min(p)，Rn=max(n)−min(n)。

- L1=mean(max(0,1−Rp+Rn))：比较正负包的峰谷差，不是正负最高分之差。
- L2=mean(Σ(p[k+1]−p[k])²)：正视频相邻片段分数平滑。
- L3=mean(Σp[k])：正视频分数稀疏约束。
- L=L1+8×10⁻⁵L2+8×10⁻⁵L3。

## 最终主运行

run_id：`mil-v2-150-full`；150条视频（113正、37负）；场次分组五折；种子7、Adam、学习率0.001、每折20轮，在本折验证AUC上选最好轮次。

逐折AUC：1.0000、0.8333、0.9545、0.9242、0.9938；逐折均值0.9412。整视频0.5阈值合并计数：TP=96、FP=23、FN=17、TN=14，Precision=0.8067、Recall=0.8496、F1=0.8276。均值AUC和合并混淆统计是不同聚合单位，不能混称同一个估计量。

权重及结果：`data/results/exp008/mil-v2-150-full/`。五个mil_f*.pt均经本轮保存特征前向复算，分数与原CSV最大差约6.02×10⁻⁷；逐折AUC/F1吻合。核对命令及全部指纹见main中的冻结头回放记录。

本次验证范围是保存特征→模型权重→预测→指标；没有重新从原视频生成特征，也没有从头重训。原始逐轮stdout未完整留存。该成绩带验证折选轮的乐观偏差，只是开发数据结果，不是独立测试或严格跨物理场景成绩。

`mil-v2-150-strict`是另一个筛选片段的对照，逐折均值AUC=0.8699，不与full主版本混同。其余对照及权重保留原run_id，均不含已作废运行。
''', encoding='utf-8')
    dump(base/'OMITTED_METADATA.json', notices)
    all_files = [p for p in base.rglob('*') if p.is_file()] + [p for p in (out/'overlays').rglob('*') if p.is_file()]
    blocked = [p for p in all_files if p.suffix.lower() in {'.npz','.npy','.jpg','.png','.mp4','.mov','.avi','.pth'}]
    assert not blocked, blocked[:3]
    receipt = {'status':'export_validated', 'experiments':len(list((out/'overlays').iterdir())),
               'file_occurrences':len(all_files), 'logical_bytes':sum(p.stat().st_size for p in all_files),
               'redacted_files':redacted, 'raw_data_files':0, 'python_syntax':'passed'}
    dump(out/'validation.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


def build(root, out, repo_url):
    if out.exists():
        raise SystemExit('Export directory exists; use a new version, never overwrite it')
    out.mkdir(parents=True)
    base = out / 'base'
    base.mkdir()
    common = Export(root, base)
    for prefix in ['experiments', 'docs/data']:
        for p in files(root / prefix):
            if p.suffix.lower() in TEXT and p.stat().st_size <= MAX_TEXT:
                common.put(p, 'current-source-snapshot' if prefix == 'experiments' else 'data-definition')
    for rel in ['AGENTS.md', '.codex/docs/workspace-layout.md', '.codex/docs/experiment-git-archive.md', 'data/数据清单.md']:
        common.put(root / rel, 'project-rules')
    for p in files(root / 'data/metadata/splits'):
        common.put(p, 'frozen-split')
    proof_path = root / 'data/metadata/experiment-archive/frozen-head-replay-20261110.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    common.put(proof_path, 'checkpoint-replay-proof')
    # Metadata paths explicitly referenced by code are not all historical final inputs;
    # branch DATA_USAGE and per-run fingerprints retain that distinction.
    common.metadata()
    (base / '.gitignore').write_text('data/raw/\n.venv/\n__pycache__/\n*.mp4\n*.mov\n*.avi\n*.jpg\n*.png\n*.npz\n*.npy\n.codex/secrets/\n', encoding='utf-8')
    restore = '''"""Restore NPZ annotation metadata only; no model input pixels/features."""\nimport argparse,json,numpy as np\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('metadata');p.add_argument('--out',required=True);a=p.parse_args()\nz=json.loads(Path(a.metadata).read_text(encoding='utf-8'));out=Path(a.out)\nif out.exists(): raise SystemExit('Refusing overwrite')\narrays={k:np.array(v['values'],dtype=v['dtype']).reshape(v['shape']) for k,v in z['arrays'].items()}\nout.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays)\n'''
    (base / 'experiments/tools/restore_archived_metadata.py').write_text(restore, encoding='utf-8')
    verified = {r['run_id']: r for r in proof['runs']}
    specs = dict(SPECS)
    # User-selected report scope: one final MIL/full run, no BCE comparisons.
    old = specs['008']
    specs['008'] = (old[0], old[1], old[2],
        'MIL full逐折均值AUC=0.9412；TP96/FP23/FN17/TN14。保存特征到权重预测及指标回放通过；验证折选轮，属于开发成绩。',
        ['exp008/mil-v2-150-full'])
    specs['011'] = specs['011'][:-1] + ([f'exp008/mil-v2-150-strict{s}' for s in ['-s7','-s8','-s9','-hnp1-s7','-hnp1-s8','-hnp1-s9']],)
    strong = ['strong-v2-150-strict-gate-s7', 'strong-v2-150-strict-s8', 'strong-v2-150-strict-s9']
    strong += [f'strong-v2-150-strict{suffix}-s{seed}' for suffix in ['-pen','-ctx-gated','-ctx-gated-pen','-ctx-selfattn','-ctx-selfattn-pen'] for seed in [7,8,9]]
    specs['012'] = specs['012'][:-1] + (['exp008/' + tag for tag in strong],)
    catalog = []
    for num, (title, status, usage, conclusion, runs) in sorted(specs.items()):
        eid = 'EXP-' + num
        branch = 'codex/' + eid
        folder = out / 'overlays' / eid
        overlay = Export(root, folder)
        folder.mkdir(parents=True)
        for rel in runs:
            p = root / 'data/results' / rel
            for source in files(p):
                name = source.name.lower()
                if any(x in name for x in ['failure', 'failed', 'smoke', 'before', '.bak']):
                    continue
                if source.suffix.lower() == '.pt' and p.name in verified:
                    evidence = next((x for x in verified[p.name]['folds'] if Path(x['checkpoint']).name == source.name), None)
                    if evidence and evidence['checkpoint_sha256'] == sha(source.read_bytes()):
                        dest = folder / source.relative_to(root)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        raw = source.read_bytes(); dest.write_bytes(raw)
                        r = source.relative_to(root).as_posix()
                        overlay.manifest[r] = {'path': r, 'bytes': len(raw), 'source_sha256': sha(raw), 'export_sha256': sha(raw), 'scope': 'checkpoint-to-prediction replay passed'}
                        continue
                overlay.put(source, 'final-run-evidence')
        # Final human-facing reports and mathematical clarification; omit old fault logs.
        for p in (root / 'docs/process/experiments').glob(eid + '*'):
            if p.is_file() and p.suffix == '.md' and any(w in p.name for w in ['训练报告', '双种子复核结果', '损失函数澄清', 'D通道诊断结果', '开发结果']):
                overlay.put(p, 'final-report-with-original-qualifications')
        if num in ['011', '012']:
            overlay.put(root / 'docs/process/experiments/EXP-011-012-训练报告.md', 'final-report')
        overlay.metadata()
        entry = {'experiment_id': eid, 'branch': branch, 'title': title, 'status': status,
                 'data_and_method': usage, 'conclusion': conclusion, 'final_run_directories': ['data/results/' + x for x in runs],
                 'code_identity': 'Current migrated/fixed source snapshot; historical executed sources are only those actually preserved inside final runs.',
                 'checkpoint_replay_verified_runs': [Path(x).name for x in runs if Path(x).name in verified],
                 'unuploaded_artifacts': list(overlay.omitted.values())}
        catalog.append({k:v for k,v in entry.items() if k != 'unuploaded_artifacts'})
        dump(folder / 'EXPERIMENT.json', entry)
        dump(folder / 'SOURCE_MANIFEST.json', list(overlay.manifest.values()))
        dump(folder / 'ARTIFACT_INDEX.json', list(overlay.omitted.values()))
        description = f'# {eid}：{title}\n\n状态：{status}。\n\n{conclusion}\n\n'
        description += '## 从这里阅读\n\n- [数据与训练方式](DATA_USAGE.md)\n- [实验与结果清单](EXPERIMENT.json)\n- [原文件与发布文件指纹](SOURCE_MANIFEST.json)\n- [未上传大文件与原因](ARTIFACT_INDEX.json)\n- 公共实现：`experiments/common/`；处理工具：`experiments/tools/`；独立入口：`experiments/runspecs/' + eid + '/`（如存在）。\n\n'
        description += '## 有效结果范围\n\n仅收本实验最终有效run及必要成功组件，不收作废/故障/冒烟成绩。阴性结果保留。原始指标、标签和预测不追改；旧标签下成绩不能冒充新N/P/T标签成绩。\n\n'
        description += '已验证小头权重见上述清单；回放验证范围是冻结特征到头预测和指标，不是从原视频重建骨干特征或从随机初始化重训。其他深模型权重未完成本轮独立前向核对，暂不上传、不标为已完整复现。\n\n'
        description += f'[总索引]({repo_url}/tree/main)；本分支是独立归档，不是新训练授权。\n'
        (folder / 'README.md').write_text(description, encoding='utf-8')
        (folder / 'DATA_USAGE.md').write_text('# 数据与训练方式\n\n' + usage + '\n\n输入元信息保留在`data/metadata/`及原run的`inputs/provenance/profile`等文件。成员、标签、采样、划分和选轮以该run绑定的文件为准，不使用今天的标签覆盖历史实验。代码中默认路径不是历史执行证据。\n\n原始视频、图片、像素和特征不上传；来源及获取方式见`data/数据清单.md`。NPZ标签数组转成`.npz.metadata.json`，可用`experiments/tools/restore_archived_metadata.py`恢复数组内容；压缩包字节哈希可能因环境不同而改变。\n\n' + conclusion + '\n', encoding='utf-8')
        print(eid, status, len(overlay.manifest), 'files', flush=True)
    dump(base / 'catalog.json', catalog)
    dump(base / 'SOURCE_MANIFEST.json', list(common.manifest.values()))
    lines = ['# 泥石流识别：实验代码与结果归档', '', '每个登记实验对应一个分支；仅收最终有效运行。没有训练结果的设计明确标记。原始数据不公开，部分历史材料缺失，不能据此宣称所有实验均已端到端复现。', '', '共享代码保持原相对路径；输入为元信息说明及冻结小清单。已回放通过的小头权重随分支保存；其余权重尚未完成本轮验证，未上传。', '', '## 实验入口', '']
    lines += [f"- [{e['experiment_id']}：{e['title']}]({repo_url}/tree/{e['branch']}) — {e['status']}" for e in catalog]
    lines += ['', '## 复算与环境', '', 'EXP-008/011/012冻结头回放命令：`python -B experiments/tools/verify_frozen_head_archive.py --out <新的核对记录.json>`。需先在本地恢复原特征缓存和NPZ标签；它们的指纹在`data/metadata/experiment-archive/frozen-head-replay-20261110.json`。回放不重新训练、不生成原视频。', '', '历史环境常见本地torch 2.4.1+cpu，云端运行环境以各原run保存文件为准；未提供一个未经验证的统一requirements冒充所有实验环境。', '', '源代码迁移后的当前快照与历史执行源码分开；已有原始代码快照才作历史代码证据。结果没有独立测试时明确称开发结果。', '', '总机器索引：`catalog.json`；共享源指纹：`SOURCE_MANIFEST.json`。']
    (base / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(out / 'build-receipt.json', {'repository': repo_url, 'experiments': len(catalog), 'base_files': len(common.manifest), 'branch_names': [x['branch'] for x in catalog], 'source_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root),text=True).strip(), 'source_git_not_modified': True})
    return catalog


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=REPO)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repo-url', default='https://github.com/heiferleaf/debris-flow-thesis')
    ap.add_argument('--finalize-existing', action='store_true')
    args = ap.parse_args()
    if not args.finalize_existing:
        build(args.root.resolve(), args.out.resolve(), args.repo_url)
    finalize(args.out.resolve())
    report = args.root.resolve() / 'docs/process/experiments/EXP-008-MIL复现实验报告.md'
    if report.exists():
        (args.out.resolve() / 'overlays/EXP-008/README.md').write_bytes(report.read_bytes())


if __name__ == '__main__':
    main()
