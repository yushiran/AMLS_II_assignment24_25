import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *

def distance_metric(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    thresh_ratio: float,
    min_radius: float,
):
    coordinate_cols = ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']
    label_tensor = solution[coordinate_cols].values.reshape(len(solution), -1, len(coordinate_cols))
    predicted_tensor = submission[coordinate_cols].values.reshape(len(submission), -1, len(coordinate_cols))
    # Find the minimum euclidean distances between the true and predicted points
    solution['distance'] = np.linalg.norm(label_tensor - predicted_tensor, axis=2).min(axis=1)
    # Convert thresholds from angstroms to voxels
    solution['thresholds'] = solution['Voxel spacing'].apply(lambda x: (min_radius * thresh_ratio) / x)
    solution['predictions'] = submission['Has motor'].values
    solution.loc[(solution['distance'] > solution['thresholds']) & (solution['Has motor'] == 1) & (submission['Has motor'] == 1), 'predictions'] = 0
    return solution['predictions'].values


def score(solution: pd.DataFrame, submission: pd.DataFrame, min_radius: float, beta: float) -> float:
    """
    Parameters:
    solution (pd.DataFrame): DataFrame containing ground truth motor positions.
    submission (pd.DataFrame): DataFrame containing predicted motor positions.

    Returns:
    float: FBeta score.

    Example
    --------
    >>> solution = pd.DataFrame({
    ...     'tomo_id': [0, 1, 2, 3],
    ...     'Motor axis 0': [-1, 250, 100, 200],
    ...     'Motor axis 1': [-1, 250, 100, 200],
    ...     'Motor axis 2': [-1, 250, 100, 200],
    ...     'Voxel spacing': [10, 10, 10, 10],
    ...     'Has motor': [0, 1, 1, 1]
    ... })
    >>> submission = pd.DataFrame({
    ...     'tomo_id': [0, 1, 2, 3],
    ...     'Motor axis 0': [100, 251, 600, -1],
    ...     'Motor axis 1': [100, 251, 600, -1],
    ...     'Motor axis 2': [100, 251, 600, -1]
    ... })
    >>> score(solution, submission, 1000, 2)
    0.3571428571428571
    """
    solution = solution.sort_values('tomo_id').reset_index(drop=True)
    submission = submission.sort_values('tomo_id').reset_index(drop=True)

    filename_equiv_array = solution['tomo_id'].eq(submission['tomo_id'], fill_value=0).values

    if np.sum(filename_equiv_array) != len(solution['tomo_id']):
        raise ValueError('Submitted tomo_id values do not match the sample_submission file')

    submission['Has motor'] = 1
    # If any columns are missing an axis, it's marked with no motor
    select = (submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] == -1).any(axis='columns')
    submission.loc[select, 'Has motor'] = 0

    required_columns = ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']
    missing_columns = [col for col in required_columns if col not in submission.columns]
    if missing_columns:
        raise ValueError(f"Submission is missing the following required columns: {missing_columns}")

    # Calculate a label of 0 or 1 using the 'has motor', and 'motor axis' values
    predictions = distance_metric(
        solution,
        submission,
        thresh_ratio=1.0,
        min_radius=min_radius,
    )

    return sklearn.metrics.fbeta_score(solution['Has motor'].values, predictions, beta=beta)


def parse_csvpath_pd(submission_path: str, solution_path: str):
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)

    # Ensure the tomo_id columns are of the same type
    submission['tomo_id'] = submission['tomo_id'].astype(str)
    solution['tomo_id'] = solution['tomo_id'].astype(str)

    # Merge the solution and submission dataframes on 'tomo_id'
    merged = pd.merge(solution, submission, on='tomo_id', suffixes=('_solution', '_submission'))
    # pd.set_option('display.max_columns', None)
    solution = merged[['tomo_id', 'Motor axis 0_solution', 'Motor axis 1_solution', 'Motor axis 2_solution', 'Voxel spacing']].rename(columns={
        'Motor axis 0_solution': 'Motor axis 0',
        'Motor axis 1_solution': 'Motor axis 1',
        'Motor axis 2_solution': 'Motor axis 2'
    })

    # If any of the Motor axis columns have a value of -1, set 'Has motor' to 0, otherwise set it to 1
    solution['Has motor'] = np.where(
        (solution['Motor axis 0'] == -1) | (solution['Motor axis 1'] == -1) | (solution['Motor axis 2'] == -1),
        0,
        1
    )

    submission = merged[['tomo_id', 'Motor axis 0_submission', 'Motor axis 1_submission', 'Motor axis 2_submission']].rename(columns={
        'Motor axis 0_submission': 'Motor axis 0',
        'Motor axis 1_submission': 'Motor axis 1',
        'Motor axis 2_submission': 'Motor axis 2'
    })

    return solution,submission

if __name__ == '__main__':
    parsed_solution, parsed_submission = parse_csvpath_pd(solution_path=config.TRAIN_CSV,submission_path=os.path.join(config.SUBMISSION_DIR, 'submission.csv'))    
    final_score = score(parsed_solution, parsed_submission, 1000, 2)
    print('final_score',final_score)

