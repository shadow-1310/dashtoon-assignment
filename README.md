## Environment Setup
---
### Create virtual env
environment can be created either with Anaconda package manager or with python venv module
#### with anaconda
- create environment
```bash
conda create -n dashtoon python=3.10
```
- activate environment
```bash
conda activate dashtoon
```

#### with python venv module
- create environment
```bash
python3 -m venv dashtoon
```
- activate environment
```bash
source dashtoon/bin/activate
```

### Installing dependencies
```bash
pip install -r requirements.txt
```

#### installing the current environment as a kernel for Jupyter Notebook
```bash
ipython kernel install --user --name=dashtoon
```

#### finally launch jupyter notebook within current environment
```bash
jupyter notebook
```

## Running the streamlit app
```bash
streamlit run app.py
```
