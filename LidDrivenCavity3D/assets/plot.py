import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from pathlib import Path


script_dir = Path(__file__).parent
post_pro_dir = script_dir / "../postProcessing/ogl_170"
json_file = post_pro_dir / "results.json"
df = pd.read_json(json_file)


x = "nCells"
y = "TimeStep"
color = "nProcs"
style = "solver_p"
plot_type = "line"


relplot = sb.relplot(
    x="nCells",
    y="TimeStep",
    hue=color,
    size=None,
    style=style,
    data=df,
    col="Host",
    kind="line",
    markers=True,
)

fig = relplot.get_figure()

fig.savefig(post_pro_dir / "TimeStep.png")
