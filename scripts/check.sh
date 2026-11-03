#!/usr/bin/env bash
set -euo pipefail
python -m pip check
python -m unittest discover -s tests -v
python -m examples.ce_graph_refine.offline_demo
python -m examples.ce_graph_refine.run_all_paper --help
