import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *

def describe_dataset(train_labels: pd.DataFrame):
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


if __name__ == '__main__':
    # Load the training labels CSV into a pandas DataFrame
    train_labels = pd.read_csv(config.TRAIN_CSV)
    integrate_dataset_info(train_labels)