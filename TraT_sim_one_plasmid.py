import numpy as np
import matplotlib.pyplot as plt
import random, time
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from numba import jit, prange
import itertools
import scipy.stats as st
import pickle as pkl
import argparse, os

parser = argparse.ArgumentParser(description="Simulates the spreading of a single plasmid depending on the exclusion index (EI)")
parser.add_argument("-m", "--mode", default='movie', type=str, help="Either 'movie'/'m' or 'batch'/'b'. Default = 'movie'")
parser.add_argument("-ei", "--exclusion-index", default=1, type=int, help="Exclusion index in movie mode. Default = 1 (no exclusion).")
parser.add_argument("-eis", "--exclusion-indices", default=[1, 5, 200], nargs='+', type=int, help="Exclusion indices in batch mode, enter multiple values separated by a space. Default = 1 5 200")
parser.add_argument("-r", "--repeat", default=50, type=int, help="Number of repeats in batch mode. Default = 50")
parser.add_argument("-d", "--dead-cutoff", default=5, type=int, help="Dead cutoff. Minimum number of simultaneous mating into the same recipient which would result in the death of the recipient (lethal zygosis). Default = 5")
parser.add_argument("-s", "--frame-size", default=100, type=int, help="Size of the canvas. Default = 100 (100x100 pixels)")
parser.add_argument("-l", "--sim-len", default=500, type=int, help="Simulation length (number of time steps). Default = 500")
parser.add_argument("-o", "--output", default='', type=str, help="Output directory. If a valid path is set, saves the movie ('movie' mode - mp4) or plots ('batch' mode - svg) to the set path.")
args = parser.parse_args()

# Check validity of inputs
if args.mode not in ('movie', 'm', 'batch', 'b'):
    raise ValueError("Mode must be either 'movie' or 'batch'")

if args.exclusion_index < 1 :
    raise ValueError("Exclusion index must be greater than or equal to 1")

if type(args.exclusion_indices) is int :
    if args.exclusion_indices < 1:
        raise ValueError("Exclusion index must be greater than or equal to 1")
    else :
        args.exclusion_indices = [args.exclusion_indices]
else :
    # Check it's an iterable (list)
    try :
        len(args.exclusion_indices)
    except :
        raise TypeError("Wrong format for exclusion indices")
    for ei in args.exclusion_indices :
        if ei < 1:
            raise ValueError("Exclusion index must be greater than or equal to 1")

if args.repeat <= 0:
    raise ValueError("Number of repeats must be greater than zero")

if args.dead_cutoff <= 0:
    raise ValueError("Dead cutoff must be greater than zero")

if args.frame_size < 1:
    raise ValueError("Frame size must be greater or equal to 1")

if args.sim_len < 1:
    raise ValueError("Simulation length must be greater or equal to 1")

if args.output.replace(' ', '') == '':
    print('No output path specified. Output will not be saved.')
elif not os.path.isdir(args.output) :
    raise TypeError("Output path does not exist")

mpl.rcParams['font.family'] = 'Helvetica Neue'

# Generate list of all possible combinations of surroundding pixels order
base_surround_coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ALL_SURROUND_COORDS = np.array(list(itertools.permutations(base_surround_coords)))

# Constants representing cell types
RECIPIENT = 0
DONOR1 = 1
DEAD = 100

DEAD_CUTOFF = args.dead_cutoff


@jit(nopython=True, parallel=True)
def update_frames(N, current_frame, intermediate_frame_donor1, EI1, base_mating_success_rate_donor1):
    for x in prange(N):
        for y in range(N):
            if current_frame[x, y] in [RECIPIENT, DEAD]:
                continue

            surround_coords = ALL_SURROUND_COORDS[random.randint(0, len(ALL_SURROUND_COORDS)-1)]

            for (i, j) in surround_coords:
                ni, nj = x + i, y + j
                if ni < 0 or ni >= N or nj < 0 or nj >= N:
                    continue

                if current_frame[ni, nj] == DEAD:
                    continue

                if current_frame[ni, nj] == RECIPIENT and np.random.random() < base_mating_success_rate_donor1:
                    intermediate_frame_donor1[ni, nj] += 1
                    break
                if current_frame[ni, nj] == DONOR1 and np.random.random() < (base_mating_success_rate_donor1 / EI1):
                    intermediate_frame_donor1[ni, nj] += 1
                    break

@jit(nopython=True, parallel=True)
def apply_updates(N, current_frame, intermediate_frame_donor1, dead_cutoff):
    next_frame = np.copy(current_frame)
    for x in prange(N):
        for y in range(N):
            donor1_mating = intermediate_frame_donor1[x, y]

            if donor1_mating == 0 :
                continue
            if donor1_mating >= dead_cutoff:
                next_frame[x, y] = DEAD
                continue

            if donor1_mating != 0:
                next_frame[x, y] = DONOR1
    return next_frame

def simulate_single_repeat(k, EI1, N, max_steps, base_mating_success_rate_donor1, return_full_movie, dead_cutoff, print_repeat):
    if print_repeat :
        print(f'Repeat : {k+1}')

    frames = []

    current_frame = np.zeros((N, N), dtype=int)
    first_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    current_frame[first_donor_coords] = DONOR1

    for step in range(max_steps):
        intermediate_frame_donor1 = np.zeros((N, N), dtype=int)

        update_frames(N, current_frame, intermediate_frame_donor1, EI1, base_mating_success_rate_donor1)
        next_frame = apply_updates(N, current_frame, intermediate_frame_donor1, dead_cutoff)

        current_frame = np.copy(next_frame)
        frames.append(np.copy(current_frame))

    # Only set to True if need the full movie for visualization, since it consumes a lot of RAM
    if return_full_movie :
        return frames

    sim_len = len(frames)
    frames_results = np.zeros((max_steps, 4), dtype=float)

    plasmid1_percentage = np.array(
        [np.count_nonzero(frame == DONOR1) for frame in frames]) / N ** 2
    dead_percentage = np.array(
        [np.count_nonzero(frame == DEAD) for frame in frames]) / N ** 2

    frames_results[:sim_len, 0] = plasmid1_percentage
    frames_results[:sim_len, 1] = dead_percentage
    frames_results[:sim_len, 2] = sim_len

    return frames_results

def main_loop(repeats=1, EI1=1, N=100, max_steps=500, base_mating_success_rate_donor1=0.8, return_full_movie=False, dead_cutoff=DEAD_CUTOFF, print_repeat=False) -> list:
    all_repeats = []
    for k in range(repeats) :
        results = simulate_single_repeat(k=k, EI1=EI1, N=N, max_steps=max_steps, base_mating_success_rate_donor1=base_mating_success_rate_donor1, return_full_movie=return_full_movie, dead_cutoff=dead_cutoff, print_repeat=print_repeat)
        all_repeats.append(results)
    return all_repeats


if args.mode in ('batch', 'b'):
    start = time.time()
    # MANUAL PARAMETERS
    EI_possible_values = args.exclusion_indices
    N = args.frame_size #length and width of the canvas
    repeats = args.repeat
    max_steps = args.sim_len
    num_channels = 2 # Types of data to plot (plasmid 1, plasmid 2, dead percentages)
    alpha=1 #opacity of lines for plots

    all_results = np.empty([len(EI_possible_values), num_channels, max_steps, repeats], dtype=np.float32)
    sim_lengths = np.zeros([len(EI_possible_values), repeats])

    # EI_table[k, l, :] = combinations of EI values
    for i, EI in enumerate(EI_possible_values):
        all_repeats = main_loop(repeats=repeats, EI1=EI, N=N, max_steps=max_steps, dead_cutoff=DEAD_CUTOFF, return_full_movie=False, print_repeat=True)
        for r, frames in enumerate(all_repeats):
            all_results[i, 0, :, r] = frames[:, 0]
            all_results[i, 1, :, r] = frames[:, 1]

    print(f'Time elapsed : {time.time() - start} s')

    with open('one_donor_all_results_250.pkl', 'wb') as file:
        pkl.dump(all_results, file)

    with open('one_donor_all_results_250.pkl', 'rb') as file:
        all_results = pkl.load(file)

    # Data for bar plots
    categories = ['Plasmid 1', 'Dead']
    colors = ['#1f77b4', '#3BA941', '#FFAA3B']

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=500)
    plt.subplots_adjust(bottom=0.25)
    axs = axs.flatten()

    def plot_prct(ax, prct, label, color):
        prct_mean = prct.mean(axis=1)
        prct_std = prct.std(axis=1)

        ci_upper = prct_mean + prct_std
        ci_lower = prct_mean - prct_std

        x = list(range(len(prct_mean)))

        ax.plot(x, prct_mean, label=label, color=color, alpha=1, linewidth=3)
        ax.fill_between(x=x, y1=ci_upper, y2=ci_lower, alpha=0.2, color=color)


    # Create each bar plot in the grid

    axs[0].set_title(f'Proportion of donors', fontsize=16)
    axs[1].set_title(f'Proportion of dead cells', fontsize=16)

    for i, EI in enumerate(EI_possible_values):
        color = colors[i]
        sim_len = max_steps
        plot_prct(axs[0], all_results[i, 0, :, :], f'EI = {EI}', color)
        plot_prct(axs[1], all_results[i, 1, :, :], f'EI = {EI}', color)

    for k, ax in enumerate(axs):
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, sim_len)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Frame', fontsize=16)

        if k % 3 != 0:  # Subplots not on the leftmost column
            ax.set_yticks([])
        else:
            ax.set_ylabel('Proportion', fontsize=16)

    plt.legend(loc='lower right', bbox_to_anchor=(1, 0., 0.5, 0.5), fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if args.output == '':
        plt.show()
    else :
        plt.savefig(f'one_donor_batch_{repeats}.svg')

    print('Done')

if args.mode in ('movie', 'm') :
    from matplotlib.widgets import Slider, Button
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    mpl.use("TkAgg")

    # Define custom colormap values and colors
    cmap_values = [0, 1, 100]
    cmap_colors = ['white', 'tab:blue', 'black']

    # Normalize values and to allow non-equal spacing between the cmap values
    norm = Normalize(vmin=min(cmap_values), vmax=max(cmap_values))
    colors = [(norm(val), color) for val, color in zip(cmap_values, cmap_colors)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    EI=args.exclusion_index
    frames = main_loop(N=args.frame_size, EI1=EI, return_full_movie=True)[0]
    num_frames = len(frames)

    # Set up the figure and plot elements
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    cax = ax.matshow(frames[0], cmap=custom_cmap, vmin=min(cmap_values), vmax=max(cmap_values))
    plt.title(f'EI = {EI}')

    # Add a slider for controlling the frame
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    # Add play/pause button
    ax_play = plt.axes([0.8, 0.1, 0.1, 0.04])
    button = Button(ax_play, 'Play/Pause')

    # Variables to control the animation
    playing = True
    current_frame = 0

    # Update function for the slider
    def update(val):
        frame = int(slider.val)
        cax.set_array(frames[frame])
        fig.canvas.draw_idle()

    # Function to play/pause the animation
    def play_pause(event):
        global playing
        playing = not playing
        if playing:
            ani.event_source.start()
        else:
            ani.event_source.stop()

    # Animation function
    def animate(i):
        global current_frame
        if playing:
            current_frame += 1
            if current_frame >= num_frames:
                current_frame = 0
            slider.set_val(current_frame)

    # Connect the slider to the update function
    slider.on_changed(update)

    # Connect the button to the play/pause function
    button.on_clicked(play_pause)

    # Create the animation object
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)

    # Create custom legend for defined colors
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='gray', markersize=10, label='Recipient', markerfacecolor='white', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='tab:blue', markersize=10, label='Donor', markerfacecolor='tab:blue', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Dead', markerfacecolor='black', linestyle='None')
    ]

    plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.5, 5))

    if args.output == '' :
        # Display the plot with the slider and play/pause button
        plt.show()
    else :
        # Save animation as mp4
        ani.save(f'TraT_sim_one_donor_EI{EI}_overmating{DEAD_CUTOFF}.mp4', writer='ffmpeg', fps=30)


# # =============================
# # Output movie snapshots, uncomment the section below to activate
# # =============================
#
# from matplotlib.colors import LinearSegmentedColormap, Normalize
# # Define custom colormap values and colors
# cmap_values = [0, 1, 100]  # Define the values at which the colors change
# cmap_colors = ['white', 'tab:blue', 'black']  # Corresponding colors for the values
# cmap_name = 'custom_cmap'
#
# # Normalize values to be between 0 and 1
# norm = Normalize(vmin=min(cmap_values), vmax=max(cmap_values))
#
# # Create the custom colormap
# colors = [(norm(val), color) for val, color in zip(cmap_values, cmap_colors)]
# custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
#
# EI=1
# frames = main_loop(N=100, EI1=EI, return_full_movie=True)[0]
#
# # Set up the figure and plot elements
# fig, axs = plt.subplots(1, 3, figsize=(8, 6))
# plt.subplots_adjust(bottom=0.25)
#
# plot_frames = [20, 50, 150]
# for i, frame_no in enumerate(plot_frames):
#     axs[i].matshow(frames[frame_no], cmap=custom_cmap)
#     axs[i].set_yticks([])
#     axs[i].set_xticks([])
# # plt.title(f'EI = {EI}')
#
# # Create custom legend for defined colors
# legend_elements = [
#     plt.Line2D([0], [0], marker='s', color='gray', markersize=10, label='Recipient', markerfacecolor='white', linestyle='None'),
#     plt.Line2D([0], [0], marker='s', color='tab:blue', markersize=10, label='Donor', markerfacecolor='tab:blue', linestyle='None'),
#     plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Dead', markerfacecolor='black', linestyle='None')
# ]
#
# plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 1))
#
# # Display the plot with the slider and play/pause button
# plt.show()
# # plt.savefig('one_donor_animation.svg')