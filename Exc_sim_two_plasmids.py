import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random, itertools, time, math
import pandas as pd
from numba import jit, prange
import seaborn as sns
import scipy.stats as st
import pickle as pkl

plot_animation = True
mpl.rcParams['font.family'] = 'Hecvetica Neue'

# Generate list of all possible combinations of surroundding pixels order
base_surround_coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ALL_SURROUND_COORDS = np.array(list(itertools.permutations(base_surround_coords)))

# Constants representing cell types
RECIPIENT = 0
DONOR1 = 1
DONOR2 = 2
DONOR_MIX = 3
DEAD = 100

cell_type = {
    'recipient': 0,
    'donor1' : 1,
    'donor2' : 2,
    'donor_mix' : 3,
    'dead' : 100
}

@jit(nopython=True, parallel=True)
def update_frames(*, N:int, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, sfx1_self:int, sfx1_comp:int, sfx2_self:int, sfx2_comp:int, base_mating_success_rate_donor1:float, base_mating_success_rate_donor2:float):
    for x in prange(N):
        for y in prange(N):
            if current_frame[x, y] in [RECIPIENT, DEAD]:
                continue

            donor_plasmid1, donor_plasmid2 = False, False
            if current_frame[x, y] == DONOR_MIX:
                donor_plasmid1, donor_plasmid2 = True, True
            elif current_frame[x, y] == DONOR1:
                donor_plasmid1 = True
            elif current_frame[x, y] == DONOR2:
                donor_plasmid2 = True

            surround_coords = ALL_SURROUND_COORDS[random.randint(0, len(ALL_SURROUND_COORDS) - 1)]

            if donor_plasmid1 :
                for (i, j) in surround_coords:
                    ni, nj = x + i, y + j
                    if not ((0 <= ni < N) and (0 <= nj < N)): # ignore out of bounds
                        continue

                    if current_frame[ni, nj] == DEAD: # dead cells cannot receive any plasmid
                        continue
                    if current_frame[ni, nj] == RECIPIENT and random.random() < base_mating_success_rate_donor1:
                        intermediate_frame_donor1[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR1 and random.random() < (base_mating_success_rate_donor1 / sfx1_self):
                        intermediate_frame_donor1[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR2 and random.random() < (base_mating_success_rate_donor1 / sfx2_comp):
                        intermediate_frame_donor1[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR_MIX and random.random() < (base_mating_success_rate_donor1 / (sfx1_self * sfx2_comp)):
                        intermediate_frame_donor1[ni, nj] += 1
                        break

            if donor_plasmid2 :
                for (i, j) in surround_coords:
                    ni, nj = x + i, y + j
                    if not ((0 <= ni < N) and (0 <= nj < N)): # ignore out of bounds
                        continue

                    if current_frame[ni, nj] == DEAD: # dead cells cannot receive any plasmid
                        continue
                    if current_frame[ni, nj] == RECIPIENT and random.random() < base_mating_success_rate_donor2:
                        intermediate_frame_donor2[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR2 and random.random() < (base_mating_success_rate_donor2 / sfx2_self):
                        intermediate_frame_donor2[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR1 and random.random() < (base_mating_success_rate_donor2 / sfx1_comp):
                        intermediate_frame_donor2[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR_MIX and random.random() < (base_mating_success_rate_donor2 / (sfx2_self * sfx1_comp)):
                        intermediate_frame_donor2[ni, nj] += 1
                        break

@jit(nopython=True, parallel=True)
def apply_updates(*, N, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, dead_cutoff):
    next_frame = np.copy(current_frame)
    for x in prange(N):
        for y in prange(N):
            donor1_mating = intermediate_frame_donor1[x, y]
            donor2_mating = intermediate_frame_donor2[x, y]

            if donor1_mating == 0 and donor2_mating == 0:
                continue
            if donor1_mating + donor2_mating >= dead_cutoff :
                next_frame[x, y] = DEAD
                continue

            if donor1_mating != 0:
                if current_frame[x, y] == DONOR2:
                    next_frame[x, y] = DONOR_MIX
                if current_frame[x, y] == RECIPIENT:
                    next_frame[x, y] = DONOR1

            if donor2_mating != 0:
                if current_frame[x, y] == DONOR1:
                    next_frame[x, y] = DONOR_MIX
                if current_frame[x, y] == RECIPIENT:
                    next_frame[x, y] = DONOR2

    return next_frame

def simulate_single_repeat(*, k, sfx1_self, sfx1_comp, sfx2_self, sfx2_comp, N, max_steps, base_mating_success_rate_donor1, base_mating_success_rate_donor2, print_steps, dead_cutoff, return_full_movie=False):
    print(f'Repeat : {k+1}')
    frames = []

    current_frame = np.zeros((N, N), dtype=int)

    first_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    current_frame[first_donor_coords] = DONOR1

    second_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    while second_donor_coords == first_donor_coords : #just in case they end up having the same coords
        second_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    current_frame[second_donor_coords] = DONOR2

    frames.append(np.copy(current_frame))

    for step in range(max_steps-1):
        if print_steps and step % 10 == 0:
            print(f'Step {step}')

        intermediate_frame_donor1 = np.zeros((N, N), dtype=int)
        intermediate_frame_donor2 = np.zeros((N, N), dtype=int)

        update_frames(N, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, sfx1_self, sfx1_comp, sfx2_self, sfx2_comp, base_mating_success_rate_donor1, base_mating_success_rate_donor2)

        next_frame = apply_updates(N, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, dead_cutoff)

        current_frame = np.copy(next_frame)
        frames.append(np.copy(current_frame))

    # Only set to True if need the full movie for visualization, since it consumes a lot of RAM
    if return_full_movie :
        return frames

    sim_len = len(frames)
    frames_results = np.zeros((max_steps, 4), dtype=float)

    plasmid1_percentage = np.array(
        [np.count_nonzero(frame == DONOR1) + np.count_nonzero(frame == DONOR_MIX) for frame in frames]) / N ** 2
    plasmid2_percentage = np.array(
        [np.count_nonzero(frame == DONOR2) + np.count_nonzero(frame == DONOR_MIX) for frame in frames]) / N ** 2
    dead_percentage = np.array(
        [np.count_nonzero(frame == DEAD) for frame in frames]) / N ** 2

    frames_results[:sim_len, 0] = plasmid1_percentage
    frames_results[:sim_len, 1] = plasmid2_percentage
    frames_results[:sim_len, 2] = dead_percentage
    frames_results[:sim_len, 3] = sim_len

    return frames_results

def main_loop(*, repeats=1, sfx1_self=1, sfx1_comp=1, sfx2_self=1, sfx2_comp=2, N=100, max_steps=500, base_mating_success_rate_donor1=0.8, base_mating_success_rate_donor2=0.8, print_steps=False, dead_cutoff=5, return_full_movie=False) -> list:
    '''
    :param return_full_movie: can operate in two modes : ***True*** -> returns array[repeat, frame, x, y, cell_type], ***False*** -> returns array[repeat, frame, int:0-3 (0: plasmid1 percnt, 1: plasmid2 percnt, 2: dead percnt, 3: sim_length)]
    '''
    all_repeats = []
    for k in range(repeats) :
        result = simulate_single_repeat(k=k, sfx1_self=sfx1_self, sfx1_comp=sfx1_comp, sfx2_self=sfx2_self, sfx2_comp=sfx2_comp, N=N, max_steps=max_steps, base_mating_success_rate_donor1=base_mating_success_rate_donor1, base_mating_success_rate_donor2=base_mating_success_rate_donor2, print_steps=print_steps, dead_cutoff=dead_cutoff, return_full_movie=return_full_movie)
        all_repeats.append(result)
    return all_repeats






# # SPEED BENCHMARK
# import time
# start = time.time()
#
# # MANUAL PARAMETERS
# DEAD_CUTOFF = 8 # How many matings in one step to kill the recipient? (1-8, if set >16 then no death by overmating)
# N = 100 #length and width of the canvas
# repeats = 50
# max_steps = 500
#
# sfx1_self, sfx1_comp, sfx2_self, sfx2_comp = 1,1,1,1
# all_repeats = main_loop(repeats=repeats, sfx1_self=sfx1_self, sfx1_comp=sfx1_comp, sfx2_self=sfx2_self, sfx2_comp=sfx2_comp, N=N, max_steps=max_steps, dead_cutoff=DEAD_CUTOFF)
#
# print(f'Time : {time.time()-start}')
# # time to beat : 17-18s
#
# for frames in all_repeats :
#     plasmid1_percentage = frames[:, 0]
#     plasmid2_percentage = frames[:, 1]
#     dead_percentage = frames[:, 2]
#
#     len_sim = len(frames)
#     plt.plot(np.arange(len_sim), plasmid1_percentage, color='red', linewidth=1, alpha=0.1)
#     plt.plot(np.arange(len_sim), plasmid2_percentage, color='blue', linewidth=1, alpha=0.1)
#     plt.plot(np.arange(len_sim), dead_percentage, color='black', linewidth=1, alpha=0.1)
#
# plt.show()
#













start = time.time()
# mpl.rcParams['figure.dpi'] = 500

# MANUAL PARAMETERS
sfx_possible_values = [1, 20000]
DEAD_CUTOFF = 8 # How many matings in one step to kill the recipient? (1-8, if set >16 then no death by overmating)
N = 100 #length and width of the canvas
repeats = 50
max_steps = 500
num_channels = 3 # Types of data to plot (plasmid 1, plasmid 2, dead percentages)
alpha=1 #opacity of lines for plots
#
# sfx_comb = list(itertools.product(sfx_possible_values, repeat=2))
# all_results = np.empty([len(sfx_comb)**2, num_channels, max_steps, repeats], dtype=np.float32)
#
# # Table for sfx values
# sfx_table = np.zeros([len(sfx_comb), len(sfx_comb), 4], dtype=int)
# for i, (sfx_self, sfx_comp) in enumerate(sfx_comb):
#     sfx1_self, sfx1_comp = sfx_self, sfx_comp
#     sfx_table[i, :, :2] = (sfx_self, sfx_comp)
# for j, (sfx_self, sfx_comp) in enumerate(sfx_comb):
#     sfx2_self, sfx2_comp = sfx_self, sfx_comp
#     sfx_table[:, j, 2:] = (sfx_self, sfx_comp)
#
# comb_records = []
# sim_lengths = np.zeros([len(sfx_comb), len(sfx_comb), repeats])
#
# # sfx_table[k, l, :] = combinations of sfx values
# i = 0
# for k in range(sfx_table.shape[0]):
#     for l in range(sfx_table.shape[1]):
#         sfx1_self, sfx1_comp, sfx2_self, sfx2_comp = sfx_table[k, l, :]
#         if [sfx2_self, sfx2_comp, sfx1_self, sfx1_comp] in comb_records:
#             i += 1
#             continue
#         comb_records.append([sfx1_self, sfx1_comp, sfx2_self, sfx2_comp])
#
#         all_repeats = main_loop(repeats=repeats, sfx1_self=sfx1_self, sfx1_comp=sfx1_comp, sfx2_self=sfx2_self, sfx2_comp=sfx2_comp, N=N, max_steps=max_steps, dead_cutoff=DEAD_CUTOFF)
#
#         # all_repeats[repeat, frame, channels]
#         #     channels :
#         #         0: plasmid1 percnt
#         #         1: plasmid2 percnt
#         #         2: dead percnt
#         #         3: sim_length
#
#         for r, frames in enumerate(all_repeats):
#             all_results[i, 0, :, r] = frames[:, 0]
#             all_results[i, 1, :, r] = frames[:, 1]
#             all_results[i, 2, :, r] = frames[:, 2]
#         i += 1
#
# print(f'Time elapsed : {time.time() - start} s')
#
# with open('two_donors_all_results.pkl', 'wb') as file:
#     pkl.dump(all_results, file)
#
#
# with open('two_donors_all_results_250.pkl', 'rb') as file:
#     all_results = pkl.load(file)
#
# print('Plotting...')
#
# # Data for bar plots
# categories = ['Plasmid 1', 'Plasmid 2', 'Dead']
#
# fig, axs = plt.subplots(4, 4, figsize=(12, 12))
# axs = axs.flatten()
#
# def plot_prct(ax, prct, label, color):
#     prct_mean = prct.mean(axis=1)
#     prct_std = prct.std(axis=1)
#     # n = prct.shape[1]
#
#     # diff = np.array([st.t.interval(0.9999999, len(a) - 1, loc=np.mean(a), scale=st.sem(a)) for a in prct])
#     # diff[diff > 1] = 1.0
#     # diff[diff < 0] = 0.0
#
#     ci_upper = prct_mean + prct_std
#     ci_lower = prct_mean - prct_std
#
#     x = list(range(len(prct_mean)))
#
#     ax.plot(x, prct_mean, label=label, color=color, alpha=1, linewidth=4)
#     ax.fill_between(x=x, y1=ci_upper, y2=ci_lower, alpha=0.2, color=color)
#
#
# # Create each bar plot in the grid
# for i, ax in enumerate(axs):
#     sim_len = max_steps
#
#     ax.set_ylim(-0.1, 1.1)
#     ax.set_xlim(0, sim_len)
#     ax.tick_params(axis='both', which='major', labelsize=15)
#
#     if i % 4 != 0: # Subplots not on the leftmost column
#         ax.set_yticks([])
#     else :
#         ax.set_ylabel('Proportion', fontsize=16)
#
#     if i < 12: # Subplots not on the bottommost row
#         ax.set_xticks([])
#     else :
#         ax.set_xlabel('Frame', fontsize=16)
#
#     if np.all(all_results[i, :, :, :] == 0) :
#         continue
#
#     # print(i)
#     plot_prct(ax, all_results[i, 0, :, :], 'Plasmid 1', 'tab:blue')
#     plot_prct(ax, all_results[i, 1, :, :], 'Plasmid 2', 'tab:orange')
#     plot_prct(ax, all_results[i, 2, :, :], 'Dead', 'black')
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
#
# # Display the plot
# # plt.show()
# plt.savefig('two_donors_all_results_250.svg')
#
# print('Done')















if plot_animation :
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # Define custom colormap values and colors
    cmap_values = cell_type.values()  # Define the values at which the colors change
    cmap_colors = ['white', '#1f77b4', '#ff7f0f', '#af60cb', 'black']  # Corresponding colors for the values
    cmap_name = 'custom_cmap'

    # Normalize values to be between 0 and 1
    norm = Normalize(vmin=min(cmap_values), vmax=max(cmap_values))

    # Create the custom colormap
    colors = [(norm(val), color) for val, color in zip(cmap_values, cmap_colors)]
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    sfx1_self=1
    sfx1_comp=1
    sfx2_self=20000
    sfx2_comp=20000
    N=200
    max_steps=500
    DEAD_CUTOFF=8

    frames = main_loop(repeats=1, sfx1_self=sfx1_self, sfx1_comp=sfx1_comp, sfx2_self=sfx2_self, sfx2_comp=sfx2_comp, N=N, max_steps=max_steps, dead_cutoff=DEAD_CUTOFF, print_steps=True, return_full_movie=True)[0]

    # Set up the figure and plot elements
    fig, axs = plt.subplots(1, 5, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    plot_frames = [20, 50, 100, 200, 500]
    for i, frame_no in enumerate(plot_frames):
        axs[i].matshow(frames[frame_no-1], cmap=custom_cmap, vmin=0, vmax=100)
        axs[i].set_yticks([])
        axs[i].set_xticks([])

    # Create custom legend for defined colors
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Recipient', markerfacecolor='white', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='#1f77b4', markersize=10, label='Donor 1', markerfacecolor='#1f77b4', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='#ff7f0f', markersize=10, label='Donor 2', markerfacecolor='#ff7f0f', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='#af60cb', markersize=10, label='Donor mix', markerfacecolor='#af60cb', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Dead', markerfacecolor='black', linestyle='None')
    ]

    plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 1))

    # Display the plot with the slider and play/pause button
    # plt.show()
    plt.savefig('two_donors_animation.svg')

    # Display the plot with the slider and play/pause button
    plt.show()
















#
#
# if plot_animation :
#     from matplotlib.widgets import Slider, Button
#     import matplotlib.animation as animation
#     from matplotlib.colors import LinearSegmentedColormap, Normalize
#
#     mpl.use("TkAgg")
#
#     # Define custom colormap values and colors
#     cmap_values = cell_type.values()  # Define the values at which the colors change
#     cmap_colors = ['white', '#1f77b4', '#ff7f0f', '#af60cb', 'black']  # Corresponding colors for the values
#     cmap_name = 'custom_cmap'
#
#     # Normalize values to be between 0 and 1
#     norm = Normalize(vmin=min(cmap_values), vmax=max(cmap_values))
#
#     # Create the custom colormap
#     colors = [(norm(val), color) for val, color in zip(cmap_values, cmap_colors)]
#     custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
#
#     sfx1_self=20000
#     sfx1_comp=20000
#     sfx2_self=1
#     sfx2_comp=1
#     N=200
#     max_steps=1000
#     DEAD_CUTOFF=8
#
#     frames = main_loop(repeats=1, sfx1_self=sfx1_self, sfx1_comp=sfx1_comp, sfx2_self=sfx2_self, sfx2_comp=sfx2_comp, N=N, max_steps=max_steps, dead_cutoff=DEAD_CUTOFF, print_steps=True, return_full_movie=True)[0]
#     num_frames = len(frames)
#
#     # Set up the figure and plot elements
#     fig, ax = plt.subplots(figsize=(8, 6))
#     plt.subplots_adjust(bottom=0.25)
#     cax = ax.matshow(frames[0], cmap=custom_cmap, vmin=0, vmax=100)
#     plt.title(f'sfx1_self = {sfx1_self}, sfx1_comp = {sfx1_comp}, sfx2_self = {sfx2_self}, sfx2_comp = {sfx2_comp}', fontsize=12)
#
#     # Add a slider for controlling the frame
#     ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
#     slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
#
#     # Add play/pause button
#     ax_play = plt.axes([0.8, 0.1, 0.1, 0.04])
#     button = Button(ax_play, 'Play/Pause')
#
#     # Variables to control the animation
#     playing = False
#     current_frame = 0
#
#     # Update function for the slider
#     def update(val):
#         frame = int(slider.val)
#         cax.set_array(frames[frame])
#         fig.canvas.draw_idle()
#
#     # Function to play/pause the animation
#     def play_pause(event):
#         global playing
#         playing = not playing
#         if playing:
#             ani.event_source.start()
#         else:
#             ani.event_source.stop()
#
#     # Animation function
#     def animate(i):
#         global current_frame
#         if playing:
#             current_frame += 1
#             if current_frame >= num_frames:
#                 current_frame = 0
#             slider.set_val(current_frame)
#
#     # Connect the slider to the update function
#     slider.on_changed(update)
#
#     # Connect the button to the play/pause function
#     button.on_clicked(play_pause)
#
#     # Create the animation object
#     ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
#
#     # Create custom legend for defined colors
#     legend_elements = [
#         plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Recipient', markerfacecolor='white',
#                    linestyle='None'),
#         plt.Line2D([0], [0], marker='s', color='#1f77b4', markersize=10, label='Donor 1', markerfacecolor='#1f77b4',
#                    linestyle='None'),
#         plt.Line2D([0], [0], marker='s', color='#ff7f0f', markersize=10, label='Donor 2', markerfacecolor='#ff7f0f',
#                    linestyle='None'),
#         plt.Line2D([0], [0], marker='s', color='#af60cb', markersize=10, label='Donor mix', markerfacecolor='#af60cb',
#                    linestyle='None'),
#         plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Dead', markerfacecolor='black',
#                    linestyle='None')
#     ]
#
#     plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.5, 5))
#
#     # Display the plot with the slider and play/pause button
#     plt.show()
