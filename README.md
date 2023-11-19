## Streamlit webapp demo
##### home 
![home](https://github.com/shadow-1310/dashtoon-assignment/blob/main/streamlit-app_ui/ui_home.png)
#### modern art
![modern art](https://github.com/shadow-1310/dashtoon-assignment/blob/main/streamlit-app_ui/ui_modern_art.png)
##### Game Art
![game art](https://github.com/shadow-1310/dashtoon-assignment/blob/main/streamlit-app_ui/ui_game_art.png)
##### Manga Sketch
![sketch](https://github.com/shadow-1310/dashtoon-assignment/blob/main/streamlit-app_ui/ui_sketch.png)

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
