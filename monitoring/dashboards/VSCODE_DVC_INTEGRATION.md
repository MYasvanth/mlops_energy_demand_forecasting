# DVC Integration in VS Code

## 1. Install DVC Extension

**Extension:** `DVC` by Iterative
- Open VS Code
- Go to Extensions (Ctrl+Shift+X)
- Search "DVC"
- Install "DVC" by Iterative

## 2. VS Code Features with DVC

### Data Version Control Panel
- **View:** Shows DVC-tracked files
- **Status:** Modified, staged, committed data
- **Actions:** Add, commit, push, pull data

### Command Palette Integration
```
Ctrl+Shift+P → DVC:
├── DVC: Show Experiments
├── DVC: Show Plots
├── DVC: Add File
├── DVC: Push
├── DVC: Pull
├── DVC: Status
└── DVC: Repro (run pipeline)
```

### File Explorer Integration
- **DVC files** show with special icons
- **Right-click menu** for DVC operations
- **Status indicators** for tracked files

## 3. Setup in Your Project

### Install DVC Extension Settings
```json
// .vscode/settings.json
{
    "dvc.dvcPath": "dvc",
    "dvc.pythonPath": "python",
    "dvc.experimentsTableHeadMaxHeight": 400
}
```

### Workspace Configuration
```json
// .vscode/extensions.json
{
    "recommendations": [
        "iterative.dvc",
        "ms-python.python",
        "ms-toolsai.jupyter"
    ]
}
```

## 4. DVC Operations in VS Code

### Track Data Files
1. Right-click file → "Add to DVC"
2. Or: Command Palette → "DVC: Add File"
3. Creates `.dvc` file automatically

### View Data Status
- **DVC Panel:** Shows tracked files status
- **Source Control:** Git + DVC changes
- **Terminal:** Integrated DVC commands

### Push/Pull Data
- **Command Palette:** DVC: Push/Pull
- **Terminal:** `dvc push` / `dvc pull`
- **Status Bar:** Shows DVC operations

## 5. Integrated Workflow

### Development Flow
```bash
# 1. Modify data processing
code src/data_processing.py

# 2. Run pipeline (VS Code terminal)
dvc repro

# 3. Check changes (DVC panel)
# Modified files show in DVC view

# 4. Add to DVC (right-click)
# Right-click → "Add to DVC"

# 5. Commit (Source Control panel)
git add . && git commit -m "Update data"

# 6. Push data (Command Palette)
# Ctrl+Shift+P → "DVC: Push"
```

### Visual Features
- **File icons:** DVC-tracked files have special icons
- **Status colors:** Modified (yellow), staged (green)
- **Diff view:** Compare data versions
- **Plots:** View metrics and plots inline

## 6. Debugging DVC Issues

### VS Code Terminal
```bash
# Check DVC status
dvc status

# View DVC config
dvc config list

# Check remote connection
dvc remote list
```

### Output Panel
- **View → Output → DVC**
- Shows DVC command outputs
- Error messages and logs

## 7. Benefits in VS Code

- ✅ **Visual interface** for DVC operations
- ✅ **Integrated terminal** for commands
- ✅ **File status indicators** 
- ✅ **Command palette** shortcuts
- ✅ **Git integration** (both Git + DVC)
- ✅ **Experiment tracking** visualization
- ✅ **Plot rendering** inline

## 8. Your Project Setup

```
mlops_energy_demand_forecasting/
├── .vscode/
│   ├── settings.json          ← DVC extension config
│   └── extensions.json        ← Recommended extensions
├── .dvc/
│   ├── config                 ← DVC remote config
│   └── cache/                 ← Local DVC cache
├── data/
│   ├── raw/
│   │   ├── energy_dataset.csv.dvc  ← DVC metadata
│   │   └── weather_features.csv.dvc
│   └── processed/
│       └── processed_energy_weather.csv.dvc
└── monitoring/dashboards/
    └── prediction_dashboard_lite.py
```

**Ready for DVC development in VS Code!** 🚀