# 冻结划分清单

7份既有JSON从`experiments/splits/`原字节迁入，没有重新划分或改变视频归属。JSON继续进Git。

生成与校验代码在`experiments/splits/code/`。各生成器的适用范围见对应实验说明，不能把只支持train/val/test的检查器直接用于folds清单。

迁前后指纹见`.codex/logs/layout-migration-20261028.json`。新增划分使用新文件名并记录依据，不覆盖已引用版本。
