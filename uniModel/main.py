"""
统一入口：选择执行 train / test / 整图预测（predict_map）。

用法示例:
  py -3 main.py train
  py -3 main.py test --checkpoint best_model.pt
  py -3 main.py predict --checkpoint best_model.pt --dataset 2013houston --save_png

子命令后的参数会原样传给对应脚本（见各文件内的 argparse 说明）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 子命令 -> 脚本文件名（可多别名指向同一脚本）
SCRIPTS: dict[str, str] = {
    "train": "train.py",
    "test": "test.py",
    "predict": "predict_map.py",
    "predict_map": "predict_map.py",
}


def _print_help() -> None:
    lines = [
        "用法: python main.py <子命令> [该脚本的参数...]",
        "",
        "子命令:",
        "  train          运行 train.py（训练）",
        "  test           运行 test.py（验证 patch 评估）",
        "  predict        运行 predict_map.py（整图滑窗 + 可选 PNG）",
        "  predict_map    同 predict",
        "",
        "示例:",
        "  python main.py train",
        "  python main.py test --checkpoint best_model.pt",
        "  python main.py predict --checkpoint best_model.pt --save_png",
        "",
        f"脚本目录: {ROOT}",
    ]
    print("\n".join(lines))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        _print_help()
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in SCRIPTS:
        print(f"未知子命令: {cmd}\n", file=sys.stderr)
        _print_help()
        sys.exit(2)

    script_name = SCRIPTS[cmd]
    script_path = ROOT / script_name
    if not script_path.is_file():
        print(f"找不到脚本: {script_path}", file=sys.stderr)
        sys.exit(1)

    forwarded = sys.argv[2:]
    rc = subprocess.call([sys.executable, str(script_path), *forwarded])
    sys.exit(rc)


if __name__ == "__main__":
    main()
