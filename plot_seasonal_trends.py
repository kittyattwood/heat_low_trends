import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from matplotlib.lines import Line2D

boxes = ['box_A','box_B','box_C','box_D','box_E']
regions = ['North America', 'Sahara', 'West Asia', 'Southern Africa', 'Australia']

# read in the data
data_dir = ''

# plot the results ############################################################

# initialize figure with 5 subplots
fig, ax = plt.subplots(5, 1, figsize=(6, 10))

# loop through each region
for i, box in enumerate(boxes):
    color = sns.color_palette("Set2")[i]  # assign a unique color per region

    # read frequency data
    df_freq = pd.read_csv(f'{data_dir}/tables/{box}_freq_trend.csv')
    df_freq = df_freq.rename(columns={'Unnamed: 0': 'month'})
    print(df_freq)

    # read size data
    df_size = pd.read_csv(f'{data_dir}/tables/{box}_size_trend.csv')
    df_size = df_size.rename(columns={'Unnamed: 0': 'month'})

    # adjust ordering for Southern Hemisphere boxes (D & E)
    if box in ['box_D', 'box_E']:
        df_freq = pd.concat([df_freq[6:], df_freq[:6]]).reset_index(drop=True)
        df_size = pd.concat([df_size[6:], df_size[:6]]).reset_index(drop=True)
        ax[i].set_xticklabels([calendar.month_abbr[m] for m in range(7, 13)] + [calendar.month_abbr[m] for m in range(1, 7)])
    else:
        ax[i].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])

    x = np.arange(0, 14, 1)  # extended x-axis (0 to 13)
    
    # extend size data by adding 0s at the beginning and end
    size_values = np.concatenate(([0], df_size['slope'].values, [0]))

    # replace nans with 0s
    size_values = np.nan_to_num(size_values)

    # primary y-axis (Frequency)
    ax1 = ax[i]
    bars = ax1.bar(np.arange(1, 13, 1), df_freq['slope']*10, width=0.4, color=color, alpha=1, label="Freq.")

    # add stippling (hatch) to bars with significant trends
    for bar, trend in zip(bars, df_freq['trend']):
        if trend in ['increasing', 'decreasing']:
            bar.set_hatch('//')  # use hatch pattern when trend is significant
            
    ax1.set_ylabel("Freq. Trend \n(HL days/decade)")
    ax1.set_ylim(-1, 5)  # set y-axis limits for frequency

    # secondary y-axis (Size)
    ax2 = ax1.twinx()
    # determine marker colors based on significance
    marker_colors = ['black' if trend in ['increasing', 'decreasing'] else 'grey' for trend in df_size['trend']]
    marker_styles = ['o' if trend in ['increasing', 'decreasing'] else '' for trend in df_size['trend']]

    # extend the color list with grey at the beginning and end (to match the 0-padding in size_values)
    marker_colors = ['grey'] + marker_colors + ['grey']
    marker_styles = [''] + marker_styles + ['']

    # plot size trend line
    ax2.plot(x, size_values*10, linestyle="-", linewidth=0.8, color='k', label="Size")

    # plot markers separately with individual colors
    for xi, yi, c, s in zip(x, size_values*10, marker_colors, marker_styles):
        ax2.plot(xi, yi, marker=s, color=c, markersize=4)
    ax2.set_ylabel("Size Trend \n(km²/decade)")
    ax2.set_ylim(-65000, 320000)  # set y-axis limits for size

    # formatting
    ax1.set_xticks(np.arange(1, 13, 1))
    ax1.axhline(0, color='black', lw=0.5)  # horizontal zero line
    ax1.set_title(f'{regions[i]}', fontsize=12)

    ax1.set_xlim(0.5, 12.5)  # set x-axis limits

    # ensure the last subplot has x-axis labels
    if i == 4:
        ax1.set_xlabel("Month")

    # add vertical lines for equinoxes based on hemisphere
    if i < 3:  # Northern Hemisphere (box_A, box_B, box_C)
        ax1.axvline(x=3.5, color="black", linestyle="--", alpha=0.7, lw=0.8)  # Spring
        ax1.axvline(x=9.5, color="black", linestyle="--", alpha=0.7, lw=0.8)  # Autumn
    else:  # Southern Hemisphere (box_D, box_E)
        ax1.axvline(x=3.5, color="black", linestyle="--", alpha=0.7, lw=0.8)  # Autumn
        ax1.axvline(x=9.5, color="black", linestyle="--", alpha=0.7, lw=0.8)  # Spring

    ax1.legend(loc='upper left', edgecolor='white', facecolor='white')
    # ax2.legend(edgecolor='white', facecolor='white')

    # create custom legend handle for 'Size Trend' marker
    size_legend = Line2D([0], [0], color='k', linewidth=0.8,
                     marker='o', markersize=4, label='Size')

    # add to legend
    ax2.legend(handles=[size_legend], edgecolor='white', facecolor='white')

# adjust layout
plt.tight_layout()

# save the plot
plt.savefig(f'{data_dir}/plots/seasonal_trends.png', dpi=300)

