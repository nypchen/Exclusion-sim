import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random, itertools
from numba import jit, prange
import pickle as pkl

mpl.rcParams['font.family'] = 'Helvetica Neue'
mpl.use("TkAgg") # For IDEs, display plots in a separate window

# Run time optimization
# Generate list of all possible combinations of surroundding pixels order
base_surround_coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ALL_SURROUND_COORDS = np.array(list(itertools.permutations(base_surround_coords)))

# Constants representing cell types
RECIPIENT = 0
DONOR1 = 1
DONOR2 = 2
DONOR_MIX = 3
DEAD = 100

@jit(nopython=True, parallel=True)
def update_frames(*, N:int, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, EI1self:int, EI1comp:int, EI2self:int, EI2comp:int, base_mating_success_rate_donor1:float, base_mating_success_rate_donor2:float):
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
                    if not ((0 <= ni < N) and (0 <= nj < N)): # Ignore out of bounds
                        continue

                    if current_frame[ni, nj] == DEAD: # Dead cells cannot receive any plasmid
                        continue
                    if current_frame[ni, nj] == RECIPIENT and random.random() < base_mating_success_rate_donor1:
                        intermediate_frame_donor1[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR1 and random.random() < (base_mating_success_rate_donor1 / EI1self):
                        intermediate_frame_donor1[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR2 and random.random() < (base_mating_success_rate_donor1 / EI2comp):
                        intermediate_frame_donor1[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR_MIX and random.random() < (base_mating_success_rate_donor1 / (EI1self * EI2comp)):
                        intermediate_frame_donor1[ni, nj] += 1
                        break

            if donor_plasmid2 :
                for (i, j) in surround_coords:
                    ni, nj = x + i, y + j
                    if not ((0 <= ni < N) and (0 <= nj < N)): # Ignore out of bounds
                        continue

                    if current_frame[ni, nj] == DEAD: # Dead cells cannot receive any plasmid
                        continue
                    if current_frame[ni, nj] == RECIPIENT and random.random() < base_mating_success_rate_donor2:
                        intermediate_frame_donor2[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR2 and random.random() < (base_mating_success_rate_donor2 / EI2self):
                        intermediate_frame_donor2[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR1 and random.random() < (base_mating_success_rate_donor2 / EI1comp):
                        intermediate_frame_donor2[ni, nj] += 1
                        break
                    if current_frame[ni, nj] == DONOR_MIX and random.random() < (base_mating_success_rate_donor2 / (EI2self * EI1comp)):
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

def simulate_single_repeat(*, EI1self, EI1comp, EI2self, EI2comp, N, max_steps, base_mating_success_rate_donor1, base_mating_success_rate_donor2, print_steps, dead_cutoff, return_full_movie=False):

    frames = []
    current_frame = np.zeros((N, N), dtype=int)

    first_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    current_frame[first_donor_coords] = DONOR1

    second_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)

    # In the rare case both donors end up having the same coords
    while second_donor_coords == first_donor_coords :
        second_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    current_frame[second_donor_coords] = DONOR2

    percentages = np.zeros((max_steps, 4), dtype=float)

    # To save memory, only return the frames when a movie is requested
    # Otherwise, percentage data is extracted and saved, while full frame data is discarded after each step
    for step in range(max_steps):
        intermediate_frame_donor1 = np.zeros((N, N), dtype=int)
        intermediate_frame_donor2 = np.zeros((N, N), dtype=int)

        update_frames(N, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, EI1self, EI1comp, EI2self, EI2comp, base_mating_success_rate_donor1, base_mating_success_rate_donor2)

        next_frame = apply_updates(N, current_frame, intermediate_frame_donor1, intermediate_frame_donor2, dead_cutoff)

        current_frame = np.copy(next_frame)

        if return_full_movie:
            frames.append(np.copy(current_frame))

        plasmid1_percentage = (np.count_nonzero(current_frame == DONOR1) + np.count_nonzero(current_frame == DONOR_MIX)) / N ** 2
        plasmid2_percentage = (np.count_nonzero(current_frame == DONOR2) + np.count_nonzero(current_frame == DONOR_MIX)) / N ** 2
        dead_percentage = np.count_nonzero(current_frame == DEAD) / N ** 2

        percentages[step, 0] = plasmid1_percentage
        percentages[step, 1] = plasmid2_percentage
        percentages[step, 2] = dead_percentage
        percentages[step, 3] = max_steps

    return percentages, frames


def main_loop(*, n_repeats=1, EI1self=1, EI1comp=1, EI2self=1, EI2comp=2, N=100, max_steps=500, base_mating_success_rate_donor1=0.8, base_mating_success_rate_donor2=0.8, print_steps=False, dead_cutoff=5, return_full_movie=False) -> list:

    all_repeats = []

    for k in range(n_repeats) :
        result, frames = simulate_single_repeat(EI1self=EI1self, EI1comp=EI1comp, EI2self=EI2self, EI2comp=EI2comp, N=N, max_steps=max_steps, base_mating_success_rate_donor1=base_mating_success_rate_donor1, base_mating_success_rate_donor2=base_mating_success_rate_donor2, print_steps=print_steps, dead_cutoff=dead_cutoff, return_full_movie=return_full_movie)
        all_repeats.append(result)

    return all_repeats, frames


def run_simulation(N, n, d, EI1self_values, EI1comp_values, EI2self_values, EI2comp_values, max_steps, show_movie, save_data, save_path=''):
    print("* Two plasmids simulation started. *\n")
    print("Parameters:")
    print("Array size (N):", N)
    print("Repeats (n):", n)
    print("Dead cutoff (d):", d)
    print("EI1self:", str(EI1self_values).strip('[]'))
    print("EI1comp:", str(EI1comp_values).strip('[]'))
    print("EI2self:", str(EI2self_values).strip('[]'))
    print("EI2comp:", str(EI2comp_values).strip('[]'))
    print("Max steps (l):", max_steps)

    if show_movie:
        print("Display movie? Yes")
    else:
        print("Display movie? No")
    if save_data:
        print("Save data? Yes")
        print("Save path:", save_path)
    else:
        print("Save data? No")
    print('')

    n_repeats = n
    n_channels = 7  # Number of values to keep track of (plasmid 1, plasmid 2, dead percentages, EI1self, EI1comp, EI2self, EI2comp)
    base_mating_success_rate_donor = 0.8

    for EI_list in (EI1self_values, EI1comp_values, EI2self_values, EI2comp_values):
        if type(EI_list) != list:
            print('type(EI_list):', type(EI_list))
            raise TypeError('Input EI is not a list')

    n_conditions = len(EI1self_values) * len(EI1comp_values) * len(EI2self_values) * len(EI2comp_values)
    alpha = 1  # opacity of lines for plots

    EI1self_values.sort(reverse=False)
    EI1comp_values.sort(reverse=False)
    EI2self_values.sort(reverse=False)
    EI2comp_values.sort(reverse=False)

    EI_comb = list(itertools.product(EI1self_values, EI1comp_values, EI2self_values, EI2comp_values, repeat=1))

    n_cols = len(EI2self_values) * len(EI2comp_values)
    n_rows = len(EI1self_values) * len(EI1comp_values)

    # Table for EI values
    EI_table = np.zeros([n_rows, n_cols, 4], dtype=int)

    i_comb = 0
    for row in range(n_rows):
        for col in range(n_cols):
            EI1self, EI1comp, EI2self, EI2comp = EI_comb[i_comb]
            EI_table[row, col, 0] = EI1self
            EI_table[row, col, 1] = EI1comp
            EI_table[row, col, 2] = EI2self
            EI_table[row, col, 3] = EI2comp
            i_comb += 1

    # all_results = np.empty([len(EI_comb), n_channels, max_steps, n_repeats], dtype=np.float32)
    all_results = np.empty([n_rows, n_cols, n_channels, max_steps, n_repeats], dtype=np.float32)
    # all_results[EI combination, channels, frame, repeats]
    #     channels :
    #         0: plasmid1 percnt
    #         1: plasmid2 percnt
    #         2: dead percnt
    #         3: simulation length

    # EI_table[k, l, :] = combinations of EI values
    for row1 in range(n_rows):
        for col1 in range(n_cols):
            EI1self, EI1comp, EI2self, EI2comp = EI_table[row1, col1, :]

            print(f'Running {n_repeats} simulation{"s" if n_repeats > 1 else ""} ({row1 * n_cols + col1 + 1}/{n_conditions} EI combinations)...')

            percentages_list, all_frames = main_loop(n_repeats=n_repeats, EI1self=EI1self, EI1comp=EI1comp, EI2self=EI2self, EI2comp=EI2comp, N=N, max_steps=max_steps, dead_cutoff=d, return_full_movie=show_movie)

            # percentages_list: list of np arrays [max_steps, percentages (n_channels)]
            # Combine the data into a single np array [EI value index, channel, max_steps, percentage (n_repeats)]
            for r, frames in enumerate(percentages_list):
                all_results[row1, col1, 0, :, r] = frames[:, 0] # plasmid 1
                all_results[row1, col1, 1, :, r] = frames[:, 1] # plasmid 2
                all_results[row1, col1, 2, :, r] = frames[:, 2] # dead

            all_results[row1, col1, 3, 0, 0] = EI1self
            all_results[row1, col1, 4, 0, 0] = EI1comp
            all_results[row1, col1, 5, 0, 0] = EI2self
            all_results[row1, col1, 6, 0, 0] = EI2comp

    print("\nSimulation finished.")

    save_dict = {
        'run_type': 'two_plasmids',
        'run_parameters': (N, n, d, EI1self_values, EI1comp_values, EI2self_values, EI2comp_values, max_steps, show_movie, save_path),
        'percentages': all_results,
        'frames': all_frames
    }

    if save_data:
        with open(save_path, 'wb') as file:
            pkl.dump(save_dict, file)
        print(f'Data saved to {save_path}')

    plot_data(save_dict)

    if show_movie:
        plot_movie(save_dict)


def plot_percentages(ax, prct, label, color):
    prct_mean = prct.mean(axis=1)
    prct_std = prct.std(axis=1)

    ci_upper = prct_mean + prct_std
    ci_lower = prct_mean - prct_std

    x = list(range(len(prct_mean)))

    ax.plot(x, prct_mean, label=label, color=color, alpha=1, linewidth=3)
    ax.fill_between(x=x, y1=ci_upper, y2=ci_lower, alpha=0.2, color=color)


def plot_data(save_dict):
    N, n, d, EI1self_values, EI1comp_values, EI2self_values, EI2comp_values, max_steps, show_movie, save_path = save_dict['run_parameters']
    percentages = save_dict['percentages']

    n_plot_rows = len(EI1self_values) * len(EI1comp_values)
    n_plot_cols = len(EI2self_values) * len(EI2comp_values)

    title_fontsize = 12
    ax_label_fontsize = (12, 12) # (y-axis, x-axis)
    legend_fontsize = 12
    tick_label_fontsize = 10

    if n_plot_rows == 1 and n_plot_cols == 1:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, max_steps)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

        EI1self, EI1comp, EI2self, EI2comp = percentages[0, 0, 3:, 0, 0]
        EI1self = int(EI1self)
        EI1comp = int(EI1comp)
        EI2self = int(EI2self)
        EI2comp = int(EI2comp)

        # Annotate axes
        plot_title = r"$EI_{1}^{self}=$" + str(EI1self) + r", $EI_{1}^{comp}=$" + str(EI1comp) + '\n'
        plot_title += r"$EI_{2}^{self}=$" + str(EI2self) + r", $EI_{2}^{comp}=$" + str(EI2comp)
        ax.set_title(plot_title, fontsize=title_fontsize)
        ax.set_ylabel('Proportion', fontsize=ax_label_fontsize[0])
        ax.set_xlabel('Timestep', fontsize=ax_label_fontsize[1])

        plot_percentages(ax, percentages[0, 0, 0, :, :], 'Plasmid 1', 'tab:blue')
        plot_percentages(ax, percentages[0, 0, 1, :, :], 'Plasmid 2', 'tab:orange')
        plot_percentages(ax, percentages[0, 0, 2, :, :], 'Dead', 'black')

        # bbox_to_anchor = [x0, y0, 0, 0]
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0., 0.5, 0.5), fontsize=legend_fontsize)
        fig.subplots_adjust(left=0.1, right=0.7, top=0.85)
    else:
        fig, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=(2.3 * n_plot_cols, 2 * n_plot_rows))
        axs = np.atleast_2d(axs)

        # Create each bar plot in the grid
        for row in range(n_plot_rows):
            for col in range(n_plot_cols):
                axs[row, col].set_ylim(-0.1, 1.1)
                axs[row, col].set_xlim(0, max_steps)
                axs[row, col].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

                # Annotate plots
                EI1self, EI1comp, EI2self, EI2comp = percentages[row, col, 3:, 0, 0]
                EI1comp = int(EI1comp)
                EI2comp = int(EI2comp)

                if row == 0: # Plots on the 1st row
                    axs[row, col].set_title(r"$EI_{2}^{comp}=$" + str(EI2comp) + '\n', fontsize=title_fontsize)

                if col % n_plot_cols == 0: # y-axis label for plots on the 1st column
                    yaxis_label_text = r"$EI_{1}^{comp}=$" + str(EI1comp) + '\nProportion'
                    axs[row, col].set_ylabel(yaxis_label_text, fontsize=ax_label_fontsize[0])
                else: # Remove ticks for all other plots
                    axs[row, col].set_yticks([])

                if row == n_plot_rows-1: # x-axis label for plots on the last row
                    axs[row, col].set_xlabel('Timestep', fontsize=ax_label_fontsize[1])
                else:
                    axs[row, col].set_xticks([])

                plot_percentages(axs[row, col], percentages[row, col, 0, :, :], 'Plasmid 1', 'tab:blue')
                plot_percentages(axs[row, col], percentages[row, col, 1, :, :], 'Plasmid 2', 'tab:orange')
                plot_percentages(axs[row, col], percentages[row, col, 2, :, :], 'Dead', 'black')

        TL = (axs[0, 0].get_position().x0 * 1.06, axs[0, 0].get_position().y1)
        TR = (axs[0, -1].get_position().x1 * 0.85, axs[0, -1].get_position().y1)
        BL = (axs[-1, 0].get_position().x0 * 1.06, axs[-1, 0].get_position().y0)

        plot_grid_width = TR[0] - TL[0]
        plot_grid_height = TL[1] - BL[1]

        col_group_width = plot_grid_width / len(EI2self_values)
        col_group_center = col_group_width / 2

        row_group_width = plot_grid_height / len(EI1self_values)
        row_group_center = row_group_width / 2

        for i, EI2self in enumerate(EI2self_values):
            fig.text(x=TL[0] + i * col_group_width + col_group_center - 0.03, y=TL[1] + 0.08, s=r'$EI_2^{self}=$'+str(EI2self), fontsize=title_fontsize)

        for j, EI1self in enumerate(EI1self_values):
            fig.text(x=TL[0] - 0.12, y=TL[1] - j * row_group_width - row_group_center - 0.03, s=r'$EI_1^{self}=$'+str(EI1self), fontsize=title_fontsize, rotation=90)

        plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0., 0.5, 0.5), fontsize=legend_fontsize)
        fig.subplots_adjust(wspace=0.08, hspace=0.08, left=0.15, right=0.8)

    plt.show()
    plt.close()

currently_playing = False
current_frame_playing = 0

def plot_movie(save_dict):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    frames = save_dict['frames']

    cmap_values = (RECIPIENT, DONOR1, DONOR2, DONOR_MIX, DEAD)
    cmap_colors = ['white', '#1f77b4', '#ff7f0f', '#af60cb', 'black']
    cmap_name = 'two_plasmids_std'

    # Normalize values to be between 0 and 1
    norm = Normalize(vmin=min(cmap_values), vmax=max(cmap_values))

    # Create the custom colormap
    colors = [(norm(val), color) for val, color in zip(cmap_values, cmap_colors)]
    two_plasmids_std_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    N, n, d, EI1self_values, EI1comp_values, EI2self_values, EI2comp_values, max_steps, show_movie, save_path = save_dict['run_parameters']
    frames = save_dict['frames']

    if len(EI1self_values) > 1 or len(EI1comp_values) > 1 or len(EI2self_values) > 1 or len(EI2comp_values) > 1:
        raise ValueError("Movie can only be saved for a single EI value AND a single repeat.")

    if len(frames) == 0:
        raise ValueError("No movie saved for this simulation.")

    # Set up figure
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0, right=0.9)
    cax = ax.matshow(frames[0], cmap=two_plasmids_std_cmap, vmin=min(cmap_values), vmax=max(cmap_values))

    fig.suptitle("Two plasmids simulation", y=0.95, x=0.45, fontsize=16, weight='bold')
    ax.set_title("© Nicolas Chen", pad=35, x=0.5, fontsize=12)

    annotation_text = r"$EI_{1}^{self}=$" + str(EI1self_values[0]) + r", $EI_{1}^{comp}=$" + str(EI1comp_values[0])
    annotation_text += r", $EI_{2}^{self}=$" + str(EI2self_values[0]) + r", $EI_{2}^{comp}=$" + str(EI2comp_values[0]) + '\n'
    annotation_text += f"d={d}, l={max_steps}"

    plt.text(x=round(N * 0.5), y=round(N * 1.12), s=annotation_text, fontsize=11, horizontalalignment='center')

    # Slider
    ax_slider = plt.axes([0.15, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, 'Timestep', 1, len(frames), valinit=0, valstep=1)

    # Add a fancy-looking Play/Pause button
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import matplotlib.patches as mpatches

    def button_on_hover(event):
        if fancybox.contains_point([event.x, event.y]):
            fancybox.set_facecolor("#5c8bc1")  # Hover color
        else:
            fancybox.set_facecolor("#2061a5")  # Default color
        fig.canvas.draw_idle()

    button_ax = plt.axes([0.83, 0.08, 0.08, 0.03])
    button_ax.set_frame_on(False)

    button = Button(button_ax, 'Play')
    button.label.set_color('white')
    button.label.set_fontsize(12)
    button.label.set_fontweight('bold')

    fancybox = mpatches.FancyBboxPatch((0, 0), 1, 1, linewidth=0, facecolor="#2061a5", boxstyle="round, pad=0.1, rounding_size=0.1", mutation_aspect=3, transform=button_ax.transAxes, clip_on=False)
    button_ax.add_patch(fancybox)
    fig.canvas.mpl_connect("motion_notify_event", button_on_hover)

    # Animation state
    state = {"playing": False, "current_frame": 0}

    def update(val):
        frame = int(slider.val)-1
        cax.set_array(frames[frame])
        fig.canvas.draw_idle()

    def play_pause(event):
        if state["playing"]:
            ani.event_source.stop()
            button.label.set_text("Play")
        else:
            state["current_frame"] = int(slider.val) - 1 # Sync animation frame with slider value
            ani.event_source.start()
            button.label.set_text("Pause")
        state["playing"] = not state["playing"]

    def animate(frame):
        if state["playing"]:
            state["current_frame"] = (state["current_frame"] + 1) % len(frames)
            slider.set_val(state["current_frame"])

    # Connect UI
    slider.on_changed(update)
    button.on_clicked(play_pause)

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=100)

    # Create custom legend for defined colors
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Recipient', markerfacecolor='white', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='#1f77b4', markersize=10, label='Donor 1', markerfacecolor='#1f77b4', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='#ff7f0f', markersize=10, label='Donor 2', markerfacecolor='#ff7f0f', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='#af60cb', markersize=10, label='Donor mix', markerfacecolor='#af60cb', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Dead', markerfacecolor='black', linestyle='None')
    ]

    plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.5, 5))
    plt.show()