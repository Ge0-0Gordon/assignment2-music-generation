## Project Scope and Environment

Work only inside this repository.

This project must use the Conda environment named `cse253`.

Do not use the `base` environment for this project.

Before running code, verify the current directory and Python environment:

```powershell
pwd
git status
conda info --envs
where python
python --version
```

Prefer running Python and pip commands through:

```powershell
conda run -n cse253 python ...
conda run -n cse253 pip ...
```

Examples:

```powershell
conda run -n cse253 python -m compileall .
conda run -n cse253 python script.py
conda run -n cse253 pip install package_name
```

Do not install packages globally or into `base`.

If a required package is missing, report it first and propose the minimal install command using the `cse253` environment.

Do not modify files outside this repository.

## Reply
Please reply in chinese