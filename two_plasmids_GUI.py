import customtkinter as ctk
from two_plasmids_core import run_simulation, plot_data, plot_movie
import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import pickle as pkl
import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("700x800")
app.title("Two plasmids - main")

title_font = ctk.CTkFont(family="Arial", size=22, weight="bold")
normal_font = ctk.CTkFont(family="Arial", size=16)
checkbox_label_font = ctk.CTkFont(family="Arial", size=14)
frame_title_font = ctk.CTkFont(family="Arial", size=12)
valid_range_font = ctk.CTkFont(family="Arial", size=12)

# Visual feedback when clicking in and out of an input text field
def click_clear_focus(event):
    widget = event.widget
    # Clear focus when user clicks on background (not an entry)
    if type(widget) is not tkinter.Entry:
        app.focus_set()

app.bind("<Escape>", lambda event: app.focus_set())
app.bind("<Return>", lambda event: app.focus_set())
app.bind("<Button-1>", click_clear_focus)

def focusin(event):
    event.widget.select_range(0, ctk.END)
    event.widget.configure(justify="center")
    event.widget.xview_moveto(1)

def focusout(event, entry_id, input_dict):
    input_value = event.widget.get()
    event.widget.configure(justify="left")
    update_values(event, entry_id, input_dict, input_value)

# Function to show/hide tooltip
def show_tooltip(event):
    global tooltip
    if show_movie_checkbox.cget("state") == "disabled":
        tooltip = ctk.CTkLabel(app, text="Only available for n=1 and a single EI value", fg_color="gray20", corner_radius=5)
        tooltip.place(x=event.x_root - app.winfo_rootx() + 20,
                      y=event.y_root - app.winfo_rooty() + 10)

def hide_tooltip(event):
    global tooltip
    if tooltip is not None:
        tooltip.destroy()
        tooltip = None

def verify_movie_condition():
    single_EI_values = len(input_dict["EI1self"]) == 1 and len(input_dict["EI2self"]) == 1 and len(input_dict["EI1comp"]) == 1 and len(input_dict["EI2comp"]) == 1

    if input_dict["Repeats (n)"] != 1 or not single_EI_values:
        show_movie_checkbox.deselect()
        show_movie_checkbox.configure(state="disabled")
    else:
        show_movie_checkbox.configure(state="normal")

def update_values(event, entry_id, input_dict, input_value):
    format_valid, stored_value, disp_value = format_check(entry_id, input_value)
    event.widget.delete(0, ctk.END)

    if format_valid:
        # Update the stored and displayed value
        input_dict[list(input_dict.keys())[entry_id]] = stored_value
        event.widget.insert(0, disp_value)
    else:
        # Revert to the old value
        if entry_id in (3, 4, 5, 6): # EI values (list)
            event.widget.insert(0, str(input_dict[input_keys[entry_id]]).strip("[]"))
        else:
            event.widget.insert(0, str(input_dict[input_keys[entry_id]]))

    verify_movie_condition()


def format_check(input_id, input_value):
    if type(input_value) is not str :
        return False, None, None

    if input_value.replace(" ", "") == "":
        return False, None, None

    # For array size (L), repeats (n), dead cutoff (d), max steps (l)"
    # Check whether values are valid integers and within range
    if input_id in (0, 1, 2, 7):
        try:
            input_int = int(input_value)
        except ValueError: # input is not integer
            try:
                input_float = float(input_value)
                input_int = int(input_float)

                if input_float != input_int:
                    return False, None, None # reject non-integer float
            except ValueError: # input is not integer or float
                return False, None, None

        if values_lim[input_id][0] <= input_int <= values_lim[input_id][1]:
            return True, input_int, str(input_int)
        else:
            return False, None, None

    # For exclusion index/indices (EI)
    # Always returns a list of integers (size of list can be 1 = a single EI value)
    if 3 <= input_id <= 6:
        output_list = []
        input_list = input_value.replace(' ', '').split(',')

        for input_value in input_list:
            if input_value != '':
                try:
                    input_int = int(input_value)
                    if input_int in output_list:
                        continue
                except ValueError:
                    try:
                        input_float = float(input_value)
                        input_int = int(input_float)

                        if input_float != int(input_float):
                            continue

                        if input_float in output_list:
                            continue
                    except ValueError:
                        continue

                if values_lim[input_id][0] <= input_int <= values_lim[input_id][1]:
                    output_list.append(input_int)

            else: # empty string
                continue

        if len(output_list) < 1 or len(output_list) > 10:
            return False, None, None
        else:
            output_list.sort()
            return True, output_list, str(output_list).strip('[]')

def checkbox_event(event=None):
    if save_data_checkbox.get() == 'on':
        save_path_label.grid(row=2, column=0, padx=10, pady=(10, 10))
        save_path_entry.grid(row=2, column=1, padx=10, pady=(10, 10))
        savepath_browse_button.grid(row=2, column=2, padx=(10, 10), pady=(10, 10))

        save_path_entry.delete(0, ctk.END)
        save_path_entry.insert(0, input_dict['Save path']+default_filename())
        save_path_entry.bind("<FocusIn>", focusin)
        save_path_entry.bind("<FocusOut>", lambda event: update_save_path(event))
        save_path_entry.xview_moveto(1)
    else:
        if save_path_label is not None:
            save_path_label.grid_remove()
        if save_path_entry is not None:
            save_path_entry.grid_remove()
        if savepath_browse_button is not None:
            savepath_browse_button.grid_remove()

def default_filename(parent_path=os.getcwd()+'/'):
    current_date = datetime.datetime.now().strftime("%y%m%d")
    i = 0
    while os.path.exists(parent_path + f"Saved_data_2P_{current_date}{'' if i==0 else '_'+str(i)}.pkl"):
        i += 1
        if i > 100:
            raise FileExistsError("Too many saved files in the selected directory")

    return f"Saved_data_2P_{current_date}{'' if i==0 else '_'+str(i)}.pkl"

def check_filepath(input_path):
    # Check for empty path
    if input_path.replace(' ', '') == '':
        return False, '', '', "Empty path"

    parent_path = os.path.dirname(input_path)
    filename = os.path.basename(input_path)

    if parent_path == '':
        parent_path = os.path.abspath(os.getcwd()) # Only filename inputted, default to current working directory
    else:
        parent_path = os.path.abspath(parent_path)

    if parent_path != "/": # Add a slash unless working in the root directory
        parent_path += "/"

    if filename.endswith('.pkl'):
        if filename[:-4].replace(' ', '') == '':
            return False, '', '', 'Empty filename'
    else:
        if filename.replace('.', '') == '':
            return False, '', '', 'Empty filename'

    for char in '''!"#$%&'()*+,:;<=>?@[]^`{|}~''':
        if char in input_path:
            return False, '', '', 'Special characters'

    # If file already exists
    if (os.path.exists(input_path) and not os.path.isdir(input_path)) or (os.path.exists(input_path+'.pkl') and not input_path.endswith(".pkl")):
        return False, '', '', "File exists"

    # Valid parent folder path
    if os.path.isdir(parent_path):
        # Add '.pkl' extension if not inputted by the user
        if filename.endswith('.pkl'):
            return True, parent_path, filename[:-4].replace(".", "")+'.pkl', '' # Remove all "." in filename to avoid bugs
        else:
            return True, parent_path, filename.replace(".", "") + '.pkl', ''
    else:
        return False, '', '', 'Invalid path'

warnings = {
    # Warning name: (warning type, message)
    "Empty path": ('Error', ''),
    "Empty filename": ('Error', 'No filename entered or selected.'),
    "Special characters": ('Error', 'Special characters are not allowed in the filename.'),
    "File exists": ('Warning', 'Filename already exists. Saving now will overwrite existing data.'),
    "Invalid path": ('Warning', 'Invalid path.')
}

def update_save_path(event, checkbox_value=None):
    global warning_label
    global run_button

    # Check if function is called from save_path_entry or savepath_browse_button
    if type(checkbox_value) != tkinter.StringVar:
        input_path = event.widget.get()
    else:
        input_path = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=input_dict["Save path"],
            initialfile=default_filename()
        )

    path_valid, parent_path, filename, error_info = check_filepath(input_path)

    if warning_label is not None:
        warning_label.destroy()

    if path_valid:
        if checkbox_value is not None:
            checkbox_value.set('on')
            save_data_checkbox.update()

        input_dict["Save path"] = parent_path
        input_dict["Filename"] = filename

        save_path_entry.delete(0, ctk.END)
        save_path_entry.insert(0, parent_path+filename)
        run_button.configure(fg_color='#2061a5', state='normal')
    else:
        warning_type, warning_message = warnings[error_info]

        if warning_message != '':
            warning_label = ctk.CTkLabel(run_options_frame, font=checkbox_label_font, text_color="orange")
            warning_label.configure(text=warning_message)
            warning_label.grid(row=3, column=0, columnspan=3, padx=10, pady=(5, 10))

        if warning_type == 'Error':
            save_path_entry.delete(0, ctk.END)
            save_path_entry.insert(0, input_dict["Save path"] + input_dict["Filename"])

        if error_info == 'Invalid path':
            run_button.configure(fg_color='gray', state='disabled')

    save_path_entry.xview_moveto(1)

def run():
    save_path = save_path_entry.get()
    answer = True

    if (os.path.exists(save_path) and not os.path.isdir(save_path)) or (
            os.path.exists(save_path + '.pkl') and not save_path.endswith(".pkl")):
        answer = messagebox.askyesno("File exists", f"The file already exists:\n{save_path}\n\nDo you want to overwrite it?")

    if answer:
        run_simulation(
            input_dict['Array size (N)'],
            input_dict['Repeats (n)'],
            input_dict['Dead cutoff (d)'],
            input_dict["EI1self"],
            input_dict["EI1comp"],
            input_dict["EI2self"],
            input_dict["EI2comp"],
            input_dict["Max steps (l)"],
            show_movie.get() == 'on',
            save_data.get() == 'on',
            save_path
        )

current_dir_pkl_files = []
user_selected_pkl_files = []
pkl_files = []
data_dict = None
load_data_warning_label = None
max_size_allowed = 200 * 1024 * 1024 # 200 MB

def refresh_pkl_files():
    global current_dir_pkl_files
    global user_selected_pkl_files
    global pkl_files
    global load_data_option_menu
    global loaded_data_params_label
    global loaded_data_params_frame

    if loaded_data_params_label is not None:
        loaded_data_params_label.destroy()

    if loaded_data_params_frame is not None:
        loaded_data_params_frame.destroy()

    working_dir = os.getcwd()

    current_dir_pkl_files = [working_dir + '/' + file for file in os.listdir(working_dir) if file.endswith('.pkl')]

    current_dir_pkl_files_filtered = []

    for file in current_dir_pkl_files:
        if os.path.getsize(file) > max_size_allowed:
            continue

        with open(file, 'rb') as f:
            data_dict = pkl.load(f)

        if list(data_dict.keys()) != ['run_type', 'run_parameters', 'percentages', 'frames']:
            continue

        if data_dict['run_type'] != 'two_plasmids':
            continue

        current_dir_pkl_files_filtered.append(file)

    pkl_files = current_dir_pkl_files_filtered + user_selected_pkl_files
    load_data_option_menu.configure(values=pkl_files)

    if len(pkl_files) >= 1:
        load_data_option_menu.configure(values=pkl_files)
        load_data_option_menu.set(pkl_files[0])
    else:
        load_data_option_menu.set('')

def browse_saved_data(event):
    global data_dict

    selected_file = filedialog.askopenfilename(
        title="Select data file",
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl")],
        initialdir=input_dict["Save path"]
    )

    if selected_file == '':
        return 0

    if not os.path.isfile(selected_file):
        show_load_data_warning_message("Error file does not exist")
        return 0

    if not selected_file.endswith(".pkl"):
        show_load_data_warning_message("Error file format unvalid")
        return 0

    filesize = os.path.getsize(selected_file)

    if filesize > max_size_allowed:
        override_file_size = tkinter.messagebox.askyesno(
            f"The file you selected is {filesize // (1024 * 1024)} MB. Loading it may take up a significant amount of your computer resources. Continue anyways?")
        if not override_file_size:
            print("Large data file not loaded.")
            return 0

    if selected_file not in pkl_files:
        # Add option in dropdown list if not already in the list
        user_selected_pkl_files.append(selected_file)
        refresh_pkl_files()

    # Select corresponding option in dropdown list
    show_load_data_warning_message('')
    load_data_option_menu.set(selected_file)

def load_data():
    filepath = load_data_option_menu.get()

    with open(filepath, "rb") as f:
        data_dict = pkl.load(f)

    if list(data_dict.keys()) != ['run_type', 'run_parameters', 'percentages', 'frames']:
        show_load_data_warning_message("Error invalid file format.")
        return 0
    else:
        if len(data_dict['run_parameters']) != 10:
            show_load_data_warning_message(f"Error invalid file selected: 11 parameters expected, {len(data_dict['run_parameters'])} found.")
            return 0
        else:
            show_load_data_warning_message("")

    disp_data_parameters(data_dict)
    plot_data(data_dict)

    if data_dict['run_parameters'][5]:
        plot_movie(data_dict)

loaded_data_params_label = None
loaded_data_params_frame = None
reload_params_button = None

def show_load_data_warning_message(warning_message=''):
    global load_data_warning_label
    global loaded_data_params_label
    global loaded_data_params_frame
    global reload_params_button

    if load_data_warning_label is not None:
        load_data_warning_label.destroy()

    if warning_message != '':
        load_data_warning_label = ctk.CTkLabel(load_data_frame, font=checkbox_label_font, text_color="orange")
        load_data_warning_label.configure(text=warning_message)
        load_data_warning_label.pack(pady=(10, 10))

    if loaded_data_params_label is not None:
        loaded_data_params_label.destroy()

    if loaded_data_params_frame is not None:
        loaded_data_params_frame.destroy()

    if reload_params_button is not None:
        reload_params_button.destroy()


def clear_load_data_warning_message(event):
    show_load_data_warning_message('')

def disp_data_parameters(data_dict):
    global loaded_data_params_label
    global loaded_data_params_frame
    global reload_params_button

    if loaded_data_params_label is not None:
        loaded_data_params_label.destroy()

    if loaded_data_params_frame is not None:
        loaded_data_params_frame.destroy()

    if reload_params_button is not None:
        reload_params_button.destroy()

    loaded_data_params_label = ctk.CTkLabel(tabview.tab("Analyze"), text="Loaded data parameters", font=frame_title_font, justify="center")
    loaded_data_params_label.pack(pady=0)

    loaded_data_params_frame = ctk.CTkFrame(tabview.tab("Analyze"), border_width=1, border_color="#a8a8a8", corner_radius=5, width=420, fg_color='#1d1e1f')
    loaded_data_params_frame.pack(pady=(0, 10), padx=(80, 80), fill="x")

    N1, n1, d1, EI1self_values, EI1comp_values, EI2self_values, EI2comp_values, max_steps1, show_movie1, save_path1 = data_dict['run_parameters']

    loaded_params = ctk.CTkTextbox(loaded_data_params_frame, font=normal_font, width=250, height=150, fg_color='transparent', wrap='word', state='normal')
    loaded_params.pack(pady=(5, 5))

    params_output = 'Loaded data parameters:\n'
    params_output += f"Array size (N): {N1}\n"
    params_output += f"Repeats (n): {n1}\n"
    params_output += f"Dead cutoff (d): {d1}\n"
    params_output += f"EI1self: {str(EI1self_values).strip('[]')}\n"
    params_output += f"EI1comp: {str(EI1comp_values).strip('[]')}\n"
    params_output += f"EI2self: {str(EI2self_values).strip('[]')}\n"
    params_output += f"EI2comp: {str(EI2comp_values).strip('[]')}\n"
    params_output += f"Max steps (l): {max_steps1}\n"
    params_output += f"Show movie: {'Yes' if show_movie1 else 'No'}\n"

    loaded_params.insert("0.0", params_output)
    print(params_output)

    reload_params_button = ctk.CTkButton(loaded_data_params_frame, text="Load parameters to new run", fg_color="#2061a5", hover_color="#5c8bc1")
    reload_params_button.pack(pady=(5, 10), padx=(20, 20))
    reload_params_button.bind("<Button-1>", command=lambda event, N=N1, n=n1, d=d1, EI1self=EI1self_values, EI1comp=EI1comp_values, EI2self=EI2self_values, EI2comp=EI2comp_values, max_steps=max_steps1, show_movie=show_movie1, save_path=save_path1: reload_params(event, N, n, d,  EI1self, EI1comp, EI2self, EI2comp, max_steps, show_movie, save_path))


def reload_params(event, N, n, d, EI1self, EI1comp, EI2self, EI2comp, max_steps, show_movie, save_path):
    tabview.set("Run")
    loaded_params = (N, n, d, EI1self, EI1comp, EI2self, EI2comp, max_steps)

    for i, param in enumerate(loaded_params):
        input_entries[i].delete(0, "end")

        if i not in (3, 4, 5, 6): # Not EI values
            input_entries[i].insert(0, param)
        else: # EI values are stored in a list (even if only a single EI value)
            if len(param) == 1:
                EI_formatted = str(param[0]).strip('[]')
            elif len(param) >= 2:
                EI_formatted = str(param).strip('[]')
            input_entries[i].insert(0, EI_formatted)

    if show_movie:
        show_movie_checkbox.select()
    else:
        show_movie_checkbox.deselect()

    save_data_checkbox.select()
    checkbox_event()

    current_dir = os.path.abspath(os.getcwd())
    base_new_filename = os.path.basename(save_path).split('.')[-2]

    new_filename = base_new_filename + '_rerun.pkl'
    k=1
    while os.path.isfile(current_dir + '/' + new_filename):
        new_filename = base_new_filename + '_rerun_' + str(k) + '.pkl'
        k += 1
        if k > 50:
            raise ValueError('Too many reruns')

    save_path_entry.delete(0, "end")
    save_path_entry.insert(0, current_dir + '/' + new_filename)

    input_dict['Array size (N)'] = N
    input_dict['Repeats (n)'] = n
    input_dict['Dead cutoff (d)'] = d
    input_dict['EI1self'] = EI1self
    input_dict['EI1comp'] = EI1comp
    input_dict['EI2self'] = EI2self
    input_dict['EI2comp'] = EI2comp
    input_dict['Max_steps (l)'] = max_steps
    input_dict['Save path'] = current_dir
    input_dict['Filename'] = new_filename


# ==== Define GUI elements ====

main_title = ctk.CTkLabel(app, text="Two plasmids simulation", font=title_font)
main_title.pack(pady=(30, 20))

tabview = ctk.CTkTabview(master=app)
tabview.pack(padx=(20, 20), pady=(0, 10), fill="x")
tabview.add("Run")
tabview.add("Analyze")

# ==== "Run" tab ====

tabview.set("Run") # Selected tab by default

# SIMULATION PARAMETERS frame
input_frame_label = ctk.CTkLabel(tabview.tab("Run"), text="Simulation parameters", font=frame_title_font, justify="center")
input_frame_label.pack(pady=(15, 0))
input_frame = ctk.CTkFrame(tabview.tab("Run"), border_width=1, border_color="#a8a8a8", corner_radius=5, width=420)
input_frame.pack(pady=(0, 10), padx=(80, 80), fill="x", expand=True)

# Initialize values
input_keys = ["Array size (N)", "Repeats (n)", "Dead cutoff (d)", "EI1self", "EI1comp", "EI2self", "EI2comp", "Max steps (l)", "Save path", "Filename"]
input_values_default = [100, 50, 8, [1, 30000], [1, 30000], [1, 30000], [1, 30000], 500, os.getcwd()+'/', default_filename()]
input_dict = dict(zip(input_keys, input_values_default))

# Define valid range for integer values
values_lim = {
    0: (10, 500),  # array size (N)
    1: (1, 500),  # repeats (n)
    2: (0, 100),  # dead cutoff (d)
    3: (1, 1000000),  # EI
    4: (1, 1000000),  # EI
    5: (1, 1000000),  # EI
    6: (1, 1000000),  # EI
    7: (10, 1000)  # max steps (l)
}

# Add input entries
input_entries = {} # Store entry widgets into a dictionary to allow automatic loading of parameters from saved data (in "Analyze tab")

for i, label_text in enumerate(input_dict.keys()):
    if i not in (8, 9): # Not the checkboxes, only the input fields
        label = ctk.CTkLabel(input_frame, text=label_text, font=normal_font)
        label.grid(row=i, column=0, padx=10, pady=5, sticky="ew")

        entry = ctk.CTkEntry(input_frame, width=150, justify="left", font=normal_font)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")

        valid_range = f"[{values_lim[i][0]}-{values_lim[i][1]}]"
        valid_range_label = ctk.CTkLabel(input_frame, text=valid_range, font=valid_range_font)
        valid_range_label.grid(row=i, column=2, padx=10, pady=5, sticky="ew")

        if 3 <= i <= 6:
            entry.insert(0, str(input_dict[label_text]).strip('[]'))
        else:
            entry.insert(0, str(input_dict[label_text]))

        entry.bind("<FocusIn>", focusin)
        entry.bind("<FocusOut>", command=lambda event, entry_id=i: focusout(event, entry_id, input_dict))

        input_entries[i] = entry

for i in range(3):
    input_frame.grid_columnconfigure(i, weight=1) # Make columns expand equally

# RUN OPTIONS frame
run_options_label = ctk.CTkLabel(tabview.tab("Run"), text="Run options", font=frame_title_font, justify="center")
run_options_label.pack(pady=(10, 0))
run_options_frame = ctk.CTkFrame(tabview.tab("Run"), border_width=1, border_color="#a8a8a8", corner_radius=5)
run_options_frame.pack(pady=(0, 10), padx=(80, 80), fill="x", expand=True)

# "Show movie" checkbox
show_movie = ctk.StringVar(value="off")
show_movie_checkbox = ctk.CTkCheckBox(run_options_frame, text="Show movie", font=checkbox_label_font, variable=show_movie, onvalue="on", offvalue="off")
show_movie_checkbox.grid(row=0, column=0, columnspan=3, padx=0, pady=(10, 10))

# "Save data" checkbox
save_data = ctk.StringVar(value="off")
save_data_checkbox = ctk.CTkCheckBox(run_options_frame, text="Save data", font=checkbox_label_font, command=lambda: checkbox_event(), variable=save_data, onvalue="on", offvalue="off")
save_data_checkbox.grid(row=1, column=0, columnspan=3, padx=0, pady=(10, 10))

# Hidden until the user clicks on the "Sava data" checkbox
save_path_label = ctk.CTkLabel(run_options_frame, text="Save path", font=normal_font)
save_path_entry = ctk.CTkEntry(run_options_frame, width=250, justify="left", font=normal_font)
savepath_browse_button = ctk.CTkButton(run_options_frame, text="Browse", width=80, command=lambda event, checkbox_value=save_data: update_save_path(event, checkbox_value))
# savepath_browse_button.bind("<Button-1>", lambda event, checkbox_value=save_data: update_save_path(event, checkbox_value))

warning_label = ctk.CTkLabel(run_options_frame, font=checkbox_label_font, text_color="orange")

for i in range(3):
    run_options_frame.grid_columnconfigure(i, weight=1) # Make columns expand equally

tooltip = ctk.CTkLabel(tabview.tab("Run"), text="Only available for n=1 and a single EI value", fg_color="gray20", corner_radius=5)
show_movie_checkbox.bind("<Enter>", show_tooltip)
show_movie_checkbox.bind("<Leave>", hide_tooltip)

# Update UI depending on checkbox values
verify_movie_condition()
checkbox_event()

run_button = ctk.CTkButton(tabview.tab("Run"), text="Run!", command=run, fg_color="#2061a5", hover_color="#5c8bc1")
run_button.pack(pady=(10, 30))

# ==== "Analyze" tab ====

# LOAD DATA frame
load_data_frame_label = ctk.CTkLabel(tabview.tab("Analyze"), text="Load data", font=frame_title_font, justify="center")
load_data_frame_label.pack(pady=0)

load_data_frame = ctk.CTkFrame(tabview.tab("Analyze"), border_width=1, border_color="#a8a8a8", corner_radius=5, width=420)
load_data_frame.pack(pady=(0, 10), padx=(80, 80), fill="x")

# load_data_option_menu.pack(pady=20)
load_buttons_frame = ctk.CTkFrame(load_data_frame, fg_color="transparent")
load_buttons_frame.pack(pady=(1, 5))

load_data_browse_button = ctk.CTkButton(load_buttons_frame, text="Browse", fg_color="#2061a5", hover_color="#5c8bc1", width=90)
load_data_refresh_button = ctk.CTkButton(load_buttons_frame, text="Refresh", command=refresh_pkl_files, fg_color="#2061a5", hover_color="#5c8bc1", width=90)
load_data_browse_button.bind("<Button-1>", lambda event: browse_saved_data(event))

load_data_browse_button.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), sticky="n")
load_data_refresh_button.grid(row=0, column=1, padx=(10, 10), pady=(10, 10), sticky="n")

load_data_option_menu = ctk.CTkOptionMenu(
    master=load_data_frame,
    button_hover_color="#5c8bc1",
    command=clear_load_data_warning_message
)

load_data_option_menu.pack(pady=(0, 10), padx=(20, 20), fill="x", expand=True)

load_data_button = ctk.CTkButton(load_data_frame, text="Load & plot", command=load_data, fg_color="#2061a5", hover_color="#5c8bc1", width=90)
load_data_button.pack(pady=(5, 10), padx=(20, 20))

refresh_pkl_files()

app.mainloop()
