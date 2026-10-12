"""Export the measured compact terrain profile and physical/numerical thickness."""
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


def load_configuration(*, input_path):
    return json.loads(input_path.read_text())["configuration"]


def plot_geometry(*, configuration, nx, ny, figure_size, dpi, output_path):
    x = np.arange(nx)*configuration["dx"]
    xn = np.arange(nx)/max(nx-1,1)
    yn = (ny//2)/max(ny-1,1)
    depths = np.asarray(configuration["representative_interfaces"])
    surface = depths[-1]+configuration["representative_relief"]*(xn+
        configuration["dem_wave_fraction"]*np.sin(np.pi*xn)*np.sin(2*np.pi*yn))
    figure,axes = plt.subplots(ncols=3,figsize=figure_size,gridspec_kw={"width_ratios":[3,1,1]})
    axes[0].fill_between(x/1000,surface-depths[-1],surface-depths[-2],color="slategray",label="low-K basement")
    axes[0].fill_between(x/1000,surface-depths[-2],surface,color="skyblue",label="permeable shallow profile")
    for depth in depths:
        axes[0].plot(x/1000,surface-depth,color="black",linewidth=.6)
    axes[0].set(xlabel="Distance [km]",ylabel="Physical elevation [m]",title="900 m relief, six depth layers")
    axes[0].legend(loc="upper left",fontsize="small")
    for index,(top,bottom) in enumerate(zip(depths[:-1],depths[1:])):
        color = "slategray" if index == len(depths)-2 else "skyblue"
        axes[1].fill_between([0,1],top,bottom,color=color,edgecolor="black",linewidth=.7)
    axes[1].set(ylim=(depths[-1],0),ylabel="Depth below terrain [m]",title="25 m local stack",xticks=[])
    bottom = depths[1]
    water_table_depth = configuration["shallow_depth"]
    numerical_top = bottom-min(configuration["default_floor"],bottom)
    axes[2].fill_between([0,1],0,bottom,color="whitesmoke",edgecolor="black")
    axes[2].fill_between([0,1],water_table_depth,bottom,color="royalblue",label="physical water")
    axes[2].plot([0,1],[numerical_top,numerical_top],linestyle="--",color="firebrick",label="conductance floor")
    axes[2].set(ylim=(bottom+.01,bottom-.04),ylabel="Depth below terrain [m]",title="Initial upper layer",xticks=[])
    axes[2].legend(loc="upper left",fontsize="x-small")
    figure.tight_layout()
    figure.savefig(output_path,dpi=dpi,bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    input_path = Path("docs/3d_terrain_benchmarks64.json")
    output_path = Path("docs/3d_terrain_geometry.png")
    nx,ny = 64,64
    figure_size = (12,4)
    dpi = 200
    configuration = load_configuration(input_path=input_path)
    plot_geometry(configuration=configuration,nx=nx,ny=ny,figure_size=figure_size,
                  dpi=dpi,output_path=output_path)
