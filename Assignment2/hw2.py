import numpy as np
import pandas as pd

def fill_missing_features(existing_data, test_data, k, distance_method, distance_threshold, weighted_voting):
    def calculate_distance(x1, x2, features):
        if distance_method.lower() == 'euclidean':
            return np.sqrt(np.sum((x1[features] - x2[features]) ** 2))
        elif distance_method.lower() == 'manhattan':
            return np.sum(np.abs(x1[features] - x2[features]))
        elif distance_method.lower() == 'chebyshev':
            return np.max(np.abs(x1[features] - x2[features]))
        else:
            raise ValueError("Unsupported distance calculation method: {}".format(distance_method))

    imputed_data = test_data.copy()

    feature_columns = existing_data.columns[1:] 

    for index, row in test_data.iterrows():
        if row.isnull().any():
            missing_feature = row[row.isnull()].index[0]
            available_features = [feat for feat in feature_columns if feat in row.dropna().index]
            
            distances = existing_data.apply(lambda x: calculate_distance(x, row, available_features), axis=1)
            
            if distance_threshold is not None:
                distances = distances[distances <= distance_threshold]
            
            nearest_indices = distances.nsmallest(k).index
            nearest_neighbors = existing_data.loc[nearest_indices]

            if weighted_voting:
                weights = 1 / (distances.loc[nearest_indices] ** 2 + 0.0001)
                if np.isclose(weights.sum(), 0):
                    imputed_value = nearest_neighbors[missing_feature].mean()
                else:
                    imputed_value = np.average(nearest_neighbors[missing_feature], weights=weights)
            else:
                imputed_value = nearest_neighbors[missing_feature].mean()

            imputed_data.at[index, missing_feature] = imputed_value
        else:
            pass

    return imputed_data

def knn(existing_data, test_data, k, distance_method, re_training, distance_threshold, weighted_voting):

    feature_columns = existing_data.columns[1:]

    existing_data[feature_columns] = existing_data[feature_columns].apply(pd.to_numeric, errors='coerce')
    test_data[feature_columns] = test_data[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    def calculate_distance(x1, x2):
        if distance_method.lower() == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif distance_method.lower() == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif distance_method.lower() == 'chebyshev':
            return np.max(np.abs(x1 - x2))

    predictions = []
    
    for index, test_row in test_data.iterrows():
        distances = existing_data.apply(lambda row: calculate_distance(row[feature_columns], test_row[feature_columns]), axis=1)

        if distance_threshold is not None:
            distances = distances[distances <= distance_threshold]

        if distances.empty:
            prediction = existing_data.iloc[:, 0].mode()[0]
        else:
            nearest_indices = distances.nsmallest(k).index
            nearest_neighbors = existing_data.iloc[nearest_indices]

            if weighted_voting:
                weights = 1 / (distances.loc[nearest_indices] ** 2 + 0.0001)
                weighted_sum = np.sum(nearest_neighbors.iloc[:, 0] * weights)
                total_weight = np.sum(weights)
                if total_weight > 0:
                    weighted_vote = weighted_sum / total_weight
                    prediction = int(round(weighted_vote))
                else:
                    prediction = nearest_neighbors.iloc[:, 0].mode()[0]
            else:
                prediction = nearest_neighbors.iloc[:, 0].mode()[0]

        predictions.append(prediction)

        if re_training:
            new_sample_data = [prediction] + test_row[feature_columns].tolist()
            new_sample = pd.DataFrame([new_sample_data], columns=existing_data.columns)
            existing_data = pd.concat([existing_data, new_sample], ignore_index=True)

    return predictions

