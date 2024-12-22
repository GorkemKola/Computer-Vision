import matplotlib.pyplot as plt
import json
import os
from matplotlib.colors import ListedColormap
import argparse

custom_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'pink', 'purple', 'brown']
custom_cmap = ListedColormap(custom_colors)
def plot_metrics(axes, epochs, metric, label, linestyle='-', color='b'):
    axes.plot(epochs, metric, linestyle, label=label, color=color)

def plot(all_data, path):
    plt.figure(figsize=(25, 25))
    for i, data in enumerate(all_data):
        part, name, p, lr, batch_size, data = data
        epochs = range(1, len(data['loss']) + 1)
        for j, (k, v) in enumerate(data.items()):
            # Loss plot
            plt.subplot(3, 3, j+1)
            try:
                if int(k[-1]):
                    k = k[:-2]
            except:
                k = k
            plot_metrics(plt, epochs, v, f'{part}_{name}_{p}_{lr}_{batch_size}', linestyle='o-', color=custom_cmap(i))

            # Customize the plot
            plt.xlabel('Epochs')
            plt.ylabel(f'{k}')
            plt.legend()

            plt.tight_layout()

            # Save the plot if save_path is provided
    if path:
        plt.savefig(path)
    else:
        plt.show()

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def filter_path(path, model_name=None, 
                 dropout_rates=False,
                 learning_rate=None, 
                 batch_size=None):
    if dropout_rates and '_0.0_' in path:
        return False
    
    if model_name and model_name not in path:
        return False
    
    if learning_rate and learning_rate not in path:
        return False
    
    if batch_size and batch_size not in path:
        return False
        
    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("--history_dir", 
                        default='src/histories', 
                        help="Path to the Image Dataset")
    
    parser.add_argument("--save_to", 
                        default='src/plots/output_plot.png', 
                        help="Path to the Image Dataset")
 
    parser.add_argument("--model_name", 
                    help="model name")

    parser.add_argument("--learning_rate", 
                    help="Learning Rate")
    
    parser.add_argument("--batch_size", 
                    help="Batch Size")
    
    # Replace 'path_to_json_files' with the actual path to your JSON files
    parser.add_argument("--all", 
                        action="store_true",
                        default=False, 
                        help="Plot all")
    
    parser.add_argument("--dropout_rates", 
                        action="store_true",
                        default=False, 
                        help="Plot Dropout Rates")
    args = parser.parse_args()

    history_dir = args.history_dir
    model_name = args.model_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_to = args.save_to
    paths = []

    if args.all:
        paths = os.listdir(history_dir)

    else:
        for path in os.listdir(history_dir):
            if not filter_path(path,
                           model_name,
                           args.dropout_rates,
                           learning_rate,
                           batch_size):
                continue
            paths.append(path)
        
    all_data = []

    # Loop through each JSON file in the specified directory
    for filename in paths:
        if not filename.endswith('.json'):
            continue

        name = filename[:-5]
        part, name, p, lr, batch_size = name.split('_')
        file_path = os.path.join(history_dir, filename)
        data = read_json_file(file_path)
        all_data.append((part, name, p, lr, batch_size, data))
    # Example usage of the function to plot and save metrics
    # The second argument is the file path where you want to save the plot, e.g., "output_plot.png"
    plot(all_data, path=save_to)
