@echo off
call conda activate mlops_env
python scripts\evaluation\run_all_models_fast.py
pause