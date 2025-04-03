import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *

def describe_dataset(train_labels: pd.DataFrame):
    """
    Generates a description of the dataset and writes it to a text file.

    This function provides basic information about the dataset, including its shape,
    column names, and basic statistics for numerical columns. The information is 
    printed to the console and saved to a file located at 
    `outputs/data_description/dataset_description.txt`.

    Args:
        train_labels (pd.DataFrame): A pandas DataFrame containing the training dataset.

    Outputs:
        - A text file with dataset information and statistics.
        - Console output of the same information.

    Note:
        Ensure that the `config.BASE_DIR` is correctly set to the base directory of the project
        before calling this function.

    """

    with open(f'{config.BASE_DIR}/outputs/data_description/dataset_description.txt', 'w') as f:
        # Display basic dataset information
        info = "Training dataset shape: {}\n".format(train_labels.shape)
        print(info)
        f.write(info)

        columns_info = "\nColumns in the dataset:\n{}".format(train_labels.columns.tolist())
        print(columns_info)
        f.write(columns_info)

        # Display basic statistics for numerical columns
        stats_info = "\nBasic statistics:\n{}".format(train_labels.describe())
        print(stats_info)
        f.write(stats_info)


def motor_distribution(train_labels: pd.DataFrame):
    """
    Analyzes and visualizes the distribution of motors per tomogram in the given dataset.
    This function performs the following tasks:
    1. Counts the number of unique tomograms in the dataset and writes the result to a text file.
    2. Computes the distribution of motors per tomogram and writes the result to the same text file.
    3. Generates a bar plot visualizing the distribution of motors per tomogram and saves it as an image.
    Args:
        train_labels (pd.DataFrame): A pandas DataFrame containing the dataset labels. 
                                     It must include the columns 'tomo_id' and 'Number of motors'.
    Outputs:
        - A text file at `outputs/data_description/motor_distribution.txt` containing:
          - The number of unique tomograms.
          - The distribution of motors per tomogram.
        - A bar plot saved as `outputs/data_description/motors_per_tomo_distribution.png`.
    Note:
        Ensure that the `config.BASE_DIR` is correctly set to the base directory of the project 
        and that the required output directories exist before calling this function.
    """

    with open(f'{config.BASE_DIR}/outputs/data_description/motor_distribution.txt', 'w') as f:
        # Count unique tomograms in the dataset
        unique_tomo_count = train_labels['tomo_id'].nunique()
        unique_tomo_info = f"\nNumber of unique tomograms: {unique_tomo_count}\n"
        print(unique_tomo_info)
        f.write(unique_tomo_info)

        # Compute distribution of motors per tomogram
        motors_per_tomo = train_labels.groupby('tomo_id')['Number of motors'].first().value_counts().sort_index()
        motors_distribution_info = "\nDistribution of motors per tomogram:\n{}".format(motors_per_tomo)
        print(motors_distribution_info)
        f.write(motors_distribution_info)

    # Visualize the distribution with a bar plot
    plt.figure(figsize=(8, 5))
    motors_per_tomo.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Motors per Tomogram')
    plt.xlabel('Number of Motors')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{config.BASE_DIR}/outputs/data_description/motors_per_tomo_distribution.png')


def additional_insight(train_labels: pd.DataFrame):
    """
    Analyzes and provides additional insights into the training labels dataset.
    This function performs the following tasks:
    1. Displays and writes a few sample rows from the training labels to a file.
    2. Checks for missing values in each column and writes the results to a file.
    3. Explores the range of tomogram sizes along each axis (Z, X, Y) and writes the findings to a file.
    4. Displays and writes the distribution of voxel spacing values to a file.
    The results of the analysis are saved to a text file located at:
    '{config.BASE_DIR}/outputs/data_description/additional_insight.txt'.
    Args:
        train_labels (pd.DataFrame): A DataFrame containing the training labels dataset. 
            Expected columns include:
            - 'Array shape (axis 0)': Represents the size along the Z-axis.
            - 'Array shape (axis 1)': Represents the size along the X-axis.
            - 'Array shape (axis 2)': Represents the size along the Y-axis.
            - 'Voxel spacing': Represents the voxel spacing values.
    Note:
        Ensure that the `train_labels` DataFrame contains the required columns 
        before calling this function to avoid KeyError.
    加一个注释: This function is designed for exploratory data analysis and 
    assumes the presence of specific columns in the input DataFrame.
    """

    with open(f'{config.BASE_DIR}/outputs/data_description/additional_insight.txt', 'w') as f:
        # Display a few sample rows from the training labels
        sample_rows_info = "\nSample rows from training labels:\n{}".format(train_labels.head())
        print(sample_rows_info)
        f.write(sample_rows_info)

        # Check for missing values in each column
        missing_values_info = "\nMissing values per column:\n{}".format(train_labels.isnull().sum())
        print(missing_values_info)
        f.write(missing_values_info)

        # Explore the range of tomogram sizes along each axis
        tomogram_size_info = "\nTomogram size ranges:\nZ-axis (slices): {} to {}\nX-axis (width): {} to {}\nY-axis (height): {} to {}\n".format(
            train_labels['Array shape (axis 0)'].min(), train_labels['Array shape (axis 0)'].max(),
            train_labels['Array shape (axis 1)'].min(), train_labels['Array shape (axis 1)'].max(),
            train_labels['Array shape (axis 2)'].min(), train_labels['Array shape (axis 2)'].max())
        print(tomogram_size_info)
        f.write(tomogram_size_info)

        # Display voxel spacing distribution
        voxel_spacing_counts = train_labels['Voxel spacing'].value_counts().sort_index()
        voxel_spacing_info = "\nVoxel spacing distribution:\n{}".format(voxel_spacing_counts)
        print(voxel_spacing_info)
        f.write(voxel_spacing_info)

def show_descriptive_statistics(train_labels: pd.DataFrame):
    """
    Generate and display descriptive statistics for the given dataset, save the statistics to a text file, 
    and create histograms for feature distributions.
    Args:
        train_labels (pd.DataFrame): A pandas DataFrame containing the dataset labels or features.
    Functionality:
        1. Computes descriptive statistics (mean, min, max) for the dataset and saves them to a text file.
        2. Displays the descriptive statistics in the console.
        3. Generates histograms for the feature distributions with improved design and saves the plot as an image.
    Outputs:
        - Text file: Saves descriptive statistics to 
          `outputs/data_description/descriptive_statistics.txt`.
        - Image file: Saves feature distribution histograms to 
          `outputs/data_description/feature_distributions.png`.
    Note:
        Ensure that the `config.BASE_DIR` is correctly set to the base directory of the project 
        and the `train_labels` DataFrame is properly formatted before calling this function.
    加一个注释: 此函数用于生成数据集的描述性统计信息，并保存统计结果和特征分布图表。
    """

    with open(f'{config.BASE_DIR}/outputs/data_description/descriptive_statistics.txt', 'w') as f:
        # Show descriptive statistics
        descriptive_stats = train_labels.describe().loc[['mean', 'min', 'max']].T
        print(descriptive_stats)
        f.write(descriptive_stats.to_string())

    # Improved histogram design
    train_labels.hist(
        bins=30, 
        figsize=(14, 8), 
        layout=(3, 4), 
        edgecolor="black", 
        color="#4CAF50"  # Greenish color theme
    )
    plt.suptitle("Feature Distributions", fontsize=16, fontweight='bold', color="darkblue")
    plt.tight_layout()
    plt.savefig(f'{config.BASE_DIR}/outputs/data_description/feature_distributions.png')

def show_correlation_matrix(train_labels: pd.DataFrame):
    """
    Generates and saves a heatmap visualizing the correlation matrix of the given DataFrame.
    This function computes the correlation matrix for the numeric columns in the provided 
    DataFrame and displays it as a heatmap. The heatmap uses a "coolwarm" colormap to 
    highlight positive and negative correlations, with annotations showing the correlation 
    coefficients.
    Args:
        train_labels (pd.DataFrame): A pandas DataFrame containing the data for which the 
                                     correlation matrix will be computed.
    Saves:
        A PNG image of the correlation heatmap to the path specified in the configuration 
        (config.BASE_DIR/outputs/data_description/correlation_heatmap.png).
    Note:
        Ensure that the `config.BASE_DIR` variable is correctly set in your configuration 
        file to avoid file saving issues.
    """

    plt.figure(figsize=(9, 5), facecolor="white")
    sns.heatmap(
        data=train_labels.corr(numeric_only=True),
        cmap="coolwarm",  # Strong contrast for positive/negative correlations
        vmin=-1, vmax=1,
        linecolor="white", linewidth=0.6,
        annot=True,
        fmt=".2f"
    )
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', color="black")
    plt.savefig(f'{config.BASE_DIR}/outputs/data_description/correlation_heatmap.png')

def integrate_dataset_info(train_labels: pd.DataFrame):
    # Display basic dataset information
    describe_dataset(train_labels)

    # Motor distribution analysis
    motor_distribution(train_labels)

    # Additional insights
    additional_insight(train_labels)

    # Show descriptive statistics and histograms
    show_descriptive_statistics(train_labels)

    # Show correlation matrix
    show_correlation_matrix(train_labels)


if __name__ == '__main__':
    # Load the training labels CSV into a pandas DataFrame
    train_labels = pd.read_csv(config.TRAIN_CSV)
    integrate_dataset_info(train_labels)