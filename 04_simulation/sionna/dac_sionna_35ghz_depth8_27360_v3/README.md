# dac_sionna_35ghz_depth8_27360_v3

当前固定基站 Stage 2 正式标签，3.5 GHz、depth 8、按高度自适应下倾角。已完成 27,360/27,360 瓦片，train/val/test = 21,789/2,840/2,731，合计 136,800 张传播图。

本目录已上传的顶层记录：[normalization_irt.json](normalization_irt.json)、[progress.json](progress.json)、[shard_manifest.csv](shard_manifest.csv)、[shard_metadata.json](shard_metadata.json)、[smoke_shard_manifest.csv](smoke_shard_manifest.csv)、[smoke_shard_metadata.json](smoke_shard_metadata.json)、[stage2a_residual_stats_smoke512.json](stage2a_residual_stats_smoke512.json)、[stage2a_residual_stats_train.json](stage2a_residual_stats_train.json)。完整仿真数组、逐 tile 日志与场景中间文件保存在本机。历史 JSON 中的 E 盘路径是生成时记录，现实际目录已迁入 dac。

[当前训练路线](../../../projects/irt_label_pipeline/SIONNA_CURRENT.md)。
