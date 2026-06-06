# Vehicle Flow Ascend

Vehicle Flow Ascend is a small Python project for experimenting with vehicle-flow counting on a PC demo backend and an Ascend OM backend. Task 1 establishes configuration loading and a dry-run command without importing PyTorch or CANN runtime packages.

## Dry-run usage

From this directory:

```bash
python -m pip install -e ".[dev]"
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --dry-run
```

Override selected values from the command line:

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --dry-run --source data/demo.mp4 --backend ascend_om --display false --max-frames 100
```
