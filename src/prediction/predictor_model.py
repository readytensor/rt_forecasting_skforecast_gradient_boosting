import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.ensemble import GradientBoostingRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the GradientBoosting Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "GradientBoosting Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        loss: str = "squared_error",
        criterion: str = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: int = 1,
        lags: Union[int, List[int]] = 7,
        random_state: int = 0,
        history_length: int = None,
    ):
        """Construct a new GradientBoosting Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of the data used for training.

            n_estimators (int): The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually
                results in better performance. Values must be in the range [1, inf).

            learning_rate (float):
                Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
                Values must be in the range [0.0, inf).

            loss (str): {'squared_error', 'absolute_error', 'huber', 'quantile'},
                Loss function to be optimized. 'squared_error' refers to the squared error for regression.
                'absolute_error' refers to the absolute error of regression and is a robust loss function. 'huber' is a combination of the two.
                'quantile' allows quantile regression (use alpha to specify the quantile).

            criterion (str): The function to measure the quality of a split.
                Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman,
                “squared_error” for mean squared error.
                The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.

            min_samples_split (Union[int, float]): The minimum number of samples required to split an internal node:
                If int, values must be in the range [2, inf).
                If float, values must be in the range (0.0, 1.0] and min_samples_split will be ceil(min_samples_split * n_samples).


            min_samples_leaf (Union[int, float]): The minimum number of samples required to be at a leaf node.
                 A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
                 If int, values must be in the range [1, inf).
                 If float, values must be in the range (0.0, 1.0) and min_samples_leaf will be ceil(min_samples_leaf * n_samples).

            lags (Union[int, List[int]]): Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                - int: include lags from 1 to lags (included).
                - list, 1d numpy ndarray or range: include only lags present in lags, all elements must be int.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.n_estimators = n_estimators
        self.leaarning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.lags = lags
        self.history_length = history_length
        self._is_trained = False
        self.models = {}
        self.data_schema = data_schema
        self.end_index = {}

        self.base_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.leaarning_rate,
            loss=self.loss,
            criterion=self.criterion,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )

    def _prepare_data(
        self, history: pd.DataFrame, data_schema: ForecastingSchema
    ) -> pd.DataFrame:
        """
        Puts the data into the expected shape by the forecaster.
        Drops the time column and puts all the target series as columns in the dataframe.

        Args:
            history (pd.DataFrame): The provided training data.
            data_schema (ForecastingSchema): The schema of the training data.

        Returns:
            pd.DataFrame: The processed data.
        """
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.all_ids = all_ids

        base = all_series[0][[data_schema.time_col, data_schema.target]]
        base.rename(
            columns={data_schema.target: f"{all_ids[0]}_{data_schema.target}"},
            inplace=True,
        )
        for id, series in zip(all_ids[1:], all_series[1:]):
            series = series[[data_schema.time_col, data_schema.target]]
            series.rename(
                columns={data_schema.target: f"{id}_{data_schema.target}"}, inplace=True
            )
            base = pd.merge(
                left=base, right=series, on=data_schema.time_col, how="inner"
            )
        base.drop(columns=data_schema.time_col, inplace=True)
        return base

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate GradientBoosting model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
            history_length (int): The length of the series used for training.
        """

        np.random.seed(self.random_state)

        history = self._prepare_data(history=history, data_schema=data_schema)

        if self.history_length:
            history = history.iloc[-self.history_length :]

        forecaster = ForecasterAutoregMultiSeries(
            regressor=self.base_model, lags=self.lags
        )
        self.model = forecaster
        forecaster.fit(series=history)

        self._is_trained = True
        self.data_schema = data_schema

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The predictions dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        time_col = test_data[self.data_schema.time_col]
        id_col = test_data[self.data_schema.id_col]

        predictions = self.model.predict(steps=self.data_schema.forecast_length)

        flattened_predictions = []
        for c in predictions.columns:
            flattened_predictions += predictions[c].values.tolist()

        result = pd.DataFrame(
            {
                self.data_schema.id_col: id_col,
                self.data_schema.time_col: time_col,
                prediction_col_name: flattened_predictions,
            }
        )

        return result

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(history=history, data_schema=data_schema)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
