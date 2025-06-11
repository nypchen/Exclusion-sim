import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from numba import jit, prange
import itertools
import pickle as pkl

mpl.rcParams['font.family'] = 'Arial'
mpl.use("TkAgg") # For IDEs, display plots in a separate window

# Run time optimization
# Generate list of all possible combinations of surroundding pixels order
base_surround_coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ALL_SURROUND_COORDS = np.array(list(itertools.permutations(base_surround_coords)))

# Constants representing cell types
RECIPIENT = 0
donor = 1
DEAD = 100

@jit(nopython=True, parallel=True)
def update_frames(N, current_frame, intermediate_frame_donor, EI, base_mating_success_rate_donor):
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

                if current_frame[ni, nj] == RECIPIENT and np.random.random() < base_mating_success_rate_donor:
                    intermediate_frame_donor[ni, nj] += 1
                    break
                if current_frame[ni, nj] == donor and np.random.random() < (base_mating_success_rate_donor / EI):
                    intermediate_frame_donor[ni, nj] += 1
                    break

@jit(nopython=True, parallel=True)
def apply_updates(N, current_frame, intermediate_frame_donor, dead_cutoff):
    next_frame = np.copy(current_frame)
    for x in prange(N):
        for y in range(N):
            donor_mating = intermediate_frame_donor[x, y]

            if donor_mating == 0 :
                continue
            if donor_mating >= dead_cutoff:
                next_frame[x, y] = DEAD
                continue

            if donor_mating != 0:
                next_frame[x, y] = donor
    return next_frame

def simulate_single_repeat(EI, N, max_steps, base_mating_success_rate_donor, return_full_movie, dead_cutoff):
    frames = []

    # Initialize simulation
    current_frame = np.zeros((N, N), dtype=int)
    first_donor_coords = random.randint(0, N - 1), random.randint(0, N - 1)
    current_frame[first_donor_coords] = donor # Turn a random pixel into a donor
    percentages = np.zeros((max_steps, 3), dtype=float)

    # To save memory, only return the frames when a movie is requested
    # Otherwise, percentage data is extracted and saved, while full frame data is discarded after each step
    for step in range(max_steps):
        intermediate_frame_donor = np.zeros((N, N), dtype=int)

        update_frames(N, current_frame, intermediate_frame_donor, EI, base_mating_success_rate_donor)
        next_frame = apply_updates(N, current_frame, intermediate_frame_donor, dead_cutoff)

        current_frame = np.copy(next_frame)

        if return_full_movie:
            frames.append(np.copy(current_frame))

        plasmid_percentage = np.count_nonzero(current_frame == donor) / N**2
        dead_percentage = np.count_nonzero(current_frame == DEAD) / N**2

        percentages[step, 0] = plasmid_percentage
        percentages[step, 1] = dead_percentage
        percentages[step, 2] = max_steps

    return percentages, frames

def main_loop(n_repeats, EI, N, max_steps, base_mating_success_rate_donor, return_full_movie, dead_cutoff):
    print(f'Running {n_repeats} simulation{"s" if n_repeats > 1 else ""} for EI = {EI}...')
    percentages_all_repeats  = [] # Container for all repeats

    for k in range(n_repeats) :
        percentages, frames = simulate_single_repeat(EI=EI, N=N, max_steps=max_steps, base_mating_success_rate_donor=base_mating_success_rate_donor, return_full_movie=return_full_movie, dead_cutoff=dead_cutoff)

        percentages_all_repeats.append(percentages)

    return percentages_all_repeats, frames

def run_simulation(N, n, d, EI_values, max_steps, show_movie, save_data, save_path=''):
    print("* One plasmid simulation started. *\n")
    print("Parameters:")
    print("Array size (N):", N)
    print("Repeats (n):", n)
    print("Dead cutoff (d):", d)
    print("Exclusion index (EI):", str(EI_values).strip("[]"))
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
    n_channels = 2 # Number of values to keep track of [plasmid percentages, dead cells percentages]
    base_mating_success_rate_donor = 0.8

    all_results = np.empty([len(EI_values), n_channels, max_steps, n_repeats], dtype=np.float32)

    # EI_table[k, l, :] = combinations of EI values
    for i, EI in enumerate(EI_values):
        percentages_list, all_frames = main_loop(n_repeats=n_repeats, EI=EI, N=N, max_steps=max_steps, dead_cutoff=d, base_mating_success_rate_donor=base_mating_success_rate_donor, return_full_movie=show_movie)

        # percentages_list: list of np arrays [max_steps, percentages (n_channels)]
        # Combine the data into a single np array [EI value index, channel, max_steps, percentage (n_repeats)]
        for r, frames in enumerate(percentages_list):
            all_results[i, 0, :, r] = frames[:, 0]
            all_results[i, 1, :, r] = frames[:, 1]

    print("\nSimulation finished.")

    save_dict = {
        'run_type': 'one_plasmid',
        'run_parameters': (N, n, d, EI_values, max_steps, show_movie, save_path),
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
    N, n, d, EI_values, max_steps, show_movie, save_path = save_dict['run_parameters']
    percentages = save_dict['percentages']

    # Randomly generate colors for >3 lines
    colors = ['#1f77b4', '#3BA941', '#FFAA3B']
    while len(EI_values) > len(colors):
        red = format(random.randint(0, 255), 'x')
        green = format(random.randint(0, 255), 'x')
        blue = format(random.randint(0, 255), 'x')
        h = '#'
        for c in (red, green, blue):
            if len(c) == 1:
                h += '0' + c
            else:
                h += c
        colors.append(h)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    plt.subplots_adjust(bottom=0.25)
    axs = axs.flatten()

    # Create each bar plot in the grid
    axs[0].set_title(f'Donors', fontsize=16)
    axs[1].set_title(f'Dead cells', fontsize=16)

    for i, EI in enumerate(EI_values):
        color = colors[i]
        sim_len = max_steps
        plot_percentages(axs[0], percentages[i, 0, :, :], f'{EI}', color)
        plot_percentages(axs[1], percentages[i, 1, :, :], f'{EI}', color)

    for k, ax in enumerate(axs):
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, sim_len)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Timestep', fontsize=16)

        if k % 3 != 0:  # Subplots not on the leftmost column
            ax.set_yticks([])
        else:
            ax.set_ylabel('Proportion of total cells', fontsize=16)

    plt.legend(title="Exclusion index (EI)", loc='lower right', bbox_to_anchor=(1, 0., 0.5, 0.5), fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    plt.close()

currently_playing = False
current_frame_playing = 0


def plot_movie(save_dict):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    N, n, d, EI_values, max_steps, show_movie, save_path = save_dict['run_parameters']
    frames = save_dict['frames']

    if len(EI_values) > 1 or n > 1:
        raise ValueError("Movie can only be saved for a single EI value AND a single repeat.")

    if len(frames) == 0:
        raise ValueError("No movie saved for this simulation.")

    # Set up colormap
    cmap_values = [0, 1, 100]
    cmap_colors = ['white', 'tab:blue', 'black']
    norm = Normalize(vmin=min(cmap_values), vmax=max(cmap_values))
    colors = [(norm(val), color) for val, color in zip(cmap_values, cmap_colors)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    num_frames = len(frames)

    # Set up figure
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0, right=0.9)
    cax = ax.matshow(frames[0], cmap=custom_cmap, vmin=min(cmap_values), vmax=max(cmap_values))
    fig.suptitle("One plasmid simulation", y=0.95, x=0.45, fontsize=16, weight='bold')
    ax.set_title("© Nicolas Chen", pad=35, x=0.5, fontsize=12)
    plt.text(x=round(N*0.5), y=round(N*1.08), s=f"EI={EI_values[0]}, d={d}, l={max_steps}", fontsize=12, horizontalalignment='center')

    # Slider
    ax_slider = plt.axes([0.15, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, 'Timestep', 1, num_frames, valinit=0, valstep=1)

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
            state["current_frame"] = (state["current_frame"] + 1) % num_frames
            slider.set_val(state["current_frame"])

    # Connect UI
    slider.on_changed(update)
    button.on_clicked(play_pause)

    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='gray', markersize=10, label='Recipient', markerfacecolor='white',
                   linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='tab:blue', markersize=10, label='Donor', markerfacecolor='tab:blue',
                   linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='black', markersize=10, label='Dead', markerfacecolor='black',
                   linestyle='None')
    ]

    plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.5, 5))
    plt.show()
