# WinProp 预测准备记录

这四份文件记录 2026-07-24 对 27,360 个 tile 的 WinProp `.odb` 文件进行检查和修复后的状态：

| 文件 | 用途 |
| --- | --- |
| `prediction_ready_summary.json` | 汇总 tile 数量、修复类别和替换结果 |
| `prediction_ready_all_odb.csv` | 全部 tile 的可用 `.odb` 清单，供后续脚本读取 |
| `prediction_ready_replacements.csv` | 502 个替换文件的来源、验证日志和哈希 |
| `prediction_smoke_10_tiles.csv` | 10 个代表性 tile 的快速测试名单 |

文件内部保留生成时的绝对路径，属于历史记录；本机原始 `.odb` 数据现位于 `../512mdata/`。在其他电脑使用清单前，需要将 `OdbPath` 映射到当地的数据目录。
