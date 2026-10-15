"""
export_figures.py - saves all 12 dashboard charts as PNG files (for your report / PPT).
One-time setup for image export:   pip install -U kaleido   then   plotly_get_chrome
Run: python export_figures.py
"""
import os
from dashboard import DF, build_all

os.makedirs("figures", exist_ok=True)
names = ["01_activity_donut", "02_weekly_area", "03_weekday_heatmap", "04_user_stacked_bar",
         "05_topic_bar", "06_sunburst", "07_prompt_boxplot", "08_hour_polar", "09_scatter_notes",
         "10_quiz_bar", "11_timeline", "12_correlation"]
for n, fig in zip(names, build_all(DF)):
    fig.write_image(f"figures/{n}.png", width=1000, height=fig.layout.height, scale=2)
    print("saved", n)
