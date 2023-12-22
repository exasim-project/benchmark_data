import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


script_dir = Path(__file__).parent
post_pro_dir = script_dir / "../postProcessing/ogl_170"
json_file = post_pro_dir / "results.json"
df = pd.read_json(json_file)

fig, axs = plt.subplots(figsize=(12, 4))
#df.plot(ax=axs)

fig.savefig(post_pro_dir / "TimeStep.png")
