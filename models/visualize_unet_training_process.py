import matplotlib.pyplot as plt

def read_epoch_losses_from_file(file_path):
    """
    Reads epoch loss values from a text file.

    Parameters:
        file_path (str): Path to the text file containing loss values.

    Returns:
        list of float: List of loss values.
    """
    with open(file_path, 'r') as file:
        epoch_losses = [float(line.strip()) for line in file if line.strip()]
    return epoch_losses

def visualize_epoch_loss(epoch_losses):
    """
    Visualizes the epoch loss as a line plot.

    Parameters:
        epoch_losses (list of float): List of loss values for each epoch.
    """
    epochs = range(1, len(epoch_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, marker='o', linestyle='-', color='orange', label='Loss')
    plt.title('Epoch Loss Visualization', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('epoch_loss_visualization.png', dpi=300)



# Example usage
file_path = '/home/yushiran/BYU_Locating_Bacterial_Flagellar_Motors_2025/models/3dunet/metric_values.txt'  # Replace with the actual file path
epoch_losses = read_epoch_losses_from_file(file_path)
visualize_epoch_loss(epoch_losses)
