# AcademicNotesAI – Fused Dataset Dashboard Using Python 
AcademicNotesAI is an LLM-powered study assistant where students chat with an AI, search YouTube tutorials, upload diagrams for explanation take quizzes and save notes. The goal is to understand who uses the app, when, for which topics, and how well they perform, and to present those answers through 12 different chart types with filters.

## Run in VS Code
1. Open folder and in VS Code .
2. Open a terminal and create an environment:
   python -m venv venv
   venv\Scripts\activate  
3. pip install -r requirements.txt
4. python eda.py          -> full EDA printed in the terminal
5. python dashboard.py    -> open http://127.0.0.1:8050 in your browser
6. (optional) python export_figures.py -> PNG of every chart in /figures

## Files
data.xlsx          the fused dataset 
data_prep.py       cleaning + feature engineering (dates, users, topic_group, prompt_words ...)
eda.py             exploratory data analysis
dashboard.py       Dash app: 6 KPI cards, 12 charts, 4 filters + reset
assets/style.css   dashboard styling 
export_figures.py  saves charts as images
