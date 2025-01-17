import os
import ipywidgets as widgets
# Initialize the results dictionary
results = {}
# Get the list of image files
target_folder = os.path.dirname(__file__)
image_files =  [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.jpg') or f.endswith('.png')]

from IPython.display import display, Image
import json
image_widget = widgets.Image(format='jpg')
yes_button = widgets.Button(description="Yes")
no_button = widgets.Button(description="No")
output = widgets.Output()

# Function to update the image
def update_image(index):
    if index < len(image_files):
        with open(image_files[index], "rb") as file:
            image_widget.value = file.read()
    else:
        image_widget.value = None

# Function to handle button clicks
def on_button_click(b):
    global current_index
    results[image_files[current_index]] = b.description
    current_index += 1
    if current_index < len(image_files):
        update_image(current_index)
    else:
        with output:
            print("Annotation complete!")
            print(results)

# Attach button click events
yes_button.on_click(on_button_click)
no_button.on_click(on_button_click)

# Display the UI
current_index = 0
update_image(current_index)
display(image_widget, yes_button, no_button, output)
# Create text input widgets
description_input = widgets.Text(placeholder='Enter image description')
coding_input = widgets.Text(placeholder='Enter visual coding')

# Function to handle button clicks
def on_button_click(b):
    global current_index
    results[image_files[current_index]] = {
        "description": description_input.value,
        "coding": coding_input.value
    }
    # Save the results to a json file
    json_filename = os.path.splitext(image_files[current_index])[0] + '.json'
    with open(json_filename, 'w') as json_file:
        json.dump(results[image_files[current_index]], json_file)
    
    current_index += 1
    if current_index < len(image_files):
        update_image(current_index)
        description_input.value = ''
        coding_input.value = ''
    else:
        with output:
            print("Annotation complete!")
            print(results)

# Attach button click events
yes_button.on_click(on_button_click)
no_button.on_click(on_button_click)

# Display the UI
current_index = 0
update_image(current_index)
display(image_widget, description_input, coding_input, yes_button, no_button, output)