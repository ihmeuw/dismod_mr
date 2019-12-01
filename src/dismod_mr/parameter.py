from pathlib import Path
from typing import Callable, Union

from loguru import logger
import numpy as np
import pandas as pd


DATA_TYPES = [
    'i',      # Incidence rate in susceptible population.
    'p',      # Prevalence proportion in population.
    'r',      # Remission rate in the with condition population.
    'f',      # Excess mortality rate in the with condition population.
    'rr',     # Relative risk of death in the with condition population as compared with the susceptible population.
    'X',      # Condition duration.
    'pf',     # Excess mortality scaled to the full population.  Roughly the cause specific mortality rate.
    'm_all',  # The all cause mortality rate in the full population.
    'csmr',   # Cause specific mortality rate in the full population.
    'smr',    # Some mortality ratio, not sure.
]


class InputDataSet:

    expected_columns = [
        'data_type',  # Effectively, the measure this data represents.
        'area',  # The location the data set is associated with.
        'year_start', 'year_end',  # Each pair defines the year range the value represents.
        'age_start', 'age_end',  # Each pair defines the age range the value represents.
        'age_weights',  # Something something age standardization.
        'sex',  # The sex represented by the data.
        'value',  # The data.
        'standard_error', 'lower_ci', 'upper_ci', 'effective_sample_size', # Parameters about data uncertainty
    ]

    def __init__(self, data: pd.DataFrame):
        self._original_data = data
        self._data, self._data_type = self.clean_data(data.copy())
        self._data = data

    @classmethod
    def from_file(cls, file_path: str):
        """Loads this data set from a csv."""
        path = Path(file_path).expanduser().resolve()
        if path.suffix != '.csv':
            raise NotImplementedError('Can only read data sets from csv files.')
        return cls(pd.read_csv(path))

    @property
    def original_data(self) -> pd.DataFrame:
        """Returns original data before cleaning or dropping rows."""
        return self._original_data

    @property
    def data(self) -> pd.DataFrame:
        """A copy of the underlying data represented by this data set."""
        return self._data.copy()

    @property
    def data_type(self) -> str:
        """The measure this data set represents."""
        return self._data_type

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans up input data sets where possible, otherwise errors."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data sets must be provided as pandas DataFrames.  You provided {type(data)}.')

        if isinstance(data.index, pd.MultiIndex):
            logger.warning('Provided data set contains a multi-index. This index will be dropped.')
            data = data.reset_index(drop=True)

        # extra_columns = data.columns.difference(self.expected_columns)
        # covariate_columns = [c for c in extra_columns if c.startswith('x_')]
        # if any([c for c in extra_columns]):
        #     logger.warning(f'Dropping extra columns {extra_columns} found in the data set.')
        #     data = data.loc[:, self.expected_columns]

        data, data_type = _clean_data_type_column(data)
        data = _clean_area_column(data, data_type)
        data = _clean_year_columns(data, data_type)
        data = _clean_age_columns(data, data_type)
        data = _clean_value_column(data, data_type)
        data = _clean_uncertainty_columns(data, data_type)
        return data, data_type

    def keep(self, relevant_row: Callable):
        """Keeps only relevant rows in the data set.

        Parameters
        ----------
        relevant_row
            A callable that accepts a series indexed by this data set's
            columns and returns a boolean value.

        """
        data = self.data
        mask = self.data.apply(relevant_row, axis=1)
        logger.info(f'Dropping {len(mask) - mask.sum()} rows and keeping {mask.sum()} rows from '
                    f'{self.data_type} data set.')
        self._data = data.loc[mask]

    def invalid_precision(self):
        data = self.data
        rows = (data['effective_sample_size'].isnull()
                & data['standard_error'].isnull()
                & (data['lower_ci'].isnull() | data['upper_ci'].isnull()))
        return data.loc[rows]


def _clean_data_type_column(data: pd.DataFrame) -> (pd.DataFrame, str):
    column = 'data_type'
    _check_column_exists(data, column, data_type='unknown', error=True)
    _check_one_value_column(data, column, data_type='unknown')
    data = _clean_missing(data, column, data_type='unknown')
    data = _clean_column_dtype(data, column, str, data_type='unknown')
    data_type = data.at[0, column]
    if data_type not in DATA_TYPES:
        raise ValueError(f'Data type {data_type} specified in input data is unknown. '
                         f'Data type must be one of {DATA_TYPES}')

    return data, data_type


def _clean_area_column(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    column = 'area'
    _check_column_exists(data, column, data_type, error=True)
    data = _clean_missing(data, column, data_type, error=False, fill_value='all')
    data = _clean_column_dtype(data, 'area', str, data_type)
    return data


def _clean_year_columns(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    # FIXME: Any assumptions here?  Contiguity probably. Equal widths?
    for column, fill in [('year_start', 1990), ('year_end', 2020)]:
        _check_column_exists(data, column, data_type, error=True)
        data = _clean_missing(data, column, data_type, error=False, fill_value=fill)
        data = _clean_column_dtype(data, column, int, data_type)

    return data


def _clean_age_columns(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    # FIXME: Any assumptions here? Contiguity probably. Equal widths?
    for column in ['age_start', 'age_end']:
        _check_column_exists(data, column, data_type, error=True)
        data = _clean_missing(data, column, data_type)
        data = _clean_column_dtype(data, column, float, data_type)

    column = 'age_weights'
    if not _check_column_exists(data, column, data_type):
        # TODO: Warn about what this affects.
        logger.warning(f'Column {column} not found in data for {data_type}. Filling with 1.')
        data[column] = 1.0
    if np.all(data[column].isnull()):
        logger.warning(f'Column {column} present in data for {data_type}, but all NaNs. Filling with 1.')
        data[column] = 1.0
    elif np.any(data[column].isnull()):
        raise ValueError(f'Some values and some NaNs found in {column} in {data_type} data set.')
    data = _clean_missing(data, column, data_type)
    data = _clean_column_dtype(data, column, float, data_type)

    return data


def _clean_value_column(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    column = 'value'
    _check_column_exists(data, column, data_type, error=True)
    data = _clean_missing(data, column, data_type)
    data = _clean_column_dtype(data, column, float, data_type)

    max_rate = 1e6
    bounds_map = {
        'i': (0, max_rate),
        'p': (0, 1),
        'r': (0, max_rate),
        'f': (0, max_rate),
        'rr': (1, max_rate),
        'pf': (0, max_rate),
        'm_all': (0, max_rate),
        'csmr': (0, max_rate),
    }

    lower, upper = bounds_map[data_type]
    if np.any(data[column] < lower):
        raise ValueError(f'Values below {lower} found in data for {data_type}.')
    if np.any(data[column] > upper):
        raise ValueError(f'Values above {upper} found in data for {data_type}.')

    return data


def _clean_uncertainty_columns(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    # Fill any missing columns with nans
    for column in ['standard_error', 'upper_ci', 'lower_ci', 'effective_sample_size']:
        if column not in data.columns:
            logger.warning(f'No {column} column found in data for {data_type}. Will approximate if possible.')
            data[column] = np.nan
        _clean_column_dtype(data, column, float, data_type)

    # FIXME: Should we be using se <= 0?  It's data so zero probably means missing?
    # Try to use CI to estimate missing standard error
    missing_se = data['standard_error'].isnull() | (data['standard_error'] < 0)
    if np.any(missing_se):
        logger.warning(f'Filling {missing_se.sum()} rows missing a standard error estimate for {data_type} '
                       f'using confidence interval.')
        upper, lower = data.loc[missing_se, 'upper_ci'], data.loc[missing_se, 'lower_ci']
        scale = 2 * 1.96  # Assume CI is symmetric and covers 97.5 percent of the data.
        data.loc[missing_se, 'standard_error'] = (upper - lower) / scale

    # If that fails, use a stand-in value.
    se_fill_value = 1.e6
    still_missing_se = data['standard_error'].isnull() | (data['standard_error'] < 0)
    if np.any(still_missing_se):
        logger.warning(f'{still_missing_se.sum()} rows of data with invalid uncertainty quantification for '
                       f'{data_type}. Filling with a standard error of {se_fill_value}.')
        data.loc[still_missing_se, 'standard_error'] = se_fill_value

    missing_ess = data['effective_sample_size'].isnull() | data['effective_sample_size'].isnull()
    if np.any(missing_ess):
        logger.warning(f'{missing_ess.sum()} rows of data have no measure of sample size for {data_type}. Filling with '
                       f'data value and standard error.')
        value, se = data.loc[missing_ess, 'value'], data.loc[missing_ess, 'standard_error']
        # FIXME: No idea where this estimate comes from.  Should document.
        data.loc[missing_ess, 'effective_sample_size'] = value * (1 - value) / se **2

    # Should only be spots where standard error is zero.
    still_missing_ess = data['effective_sample_size'].isnull() | data['effective_sample_size'].isnull()
    if np.any(still_missing_se):
        logger.warning(f'{still_missing_ess.sum()} rows missing sample size and have no uncertainty for {data_type}. '
                       f'Filling sample size with zero.')
        data.loc[still_missing_ess, 'effective_sample_size'] = 0

    return data


def _clean_column_dtype(data: pd.DataFrame, column_name: str, expected_dtype: type, data_type: str) -> pd.DataFrame:
    if expected_dtype == str:
        _check_string_column(data, column_name, data_type)
    else:
        dtype = data[column_name].dtype
        if expected_dtype == int and dtype == float:
            logger.warning(f'Truncating {column_name} in {data_type} data from float to int.')
            data[column_name] = data[column_name].astype(int)
        elif expected_dtype == float and dtype == int:
            # Don't need to warn about expanding data type.
            data[column_name] = data[column_name].astype(float)
        elif dtype != expected_dtype:
            raise TypeError(f'Column {column_name} must be filled with {expected_dtype}. '
                            f'You specified {dtype} in the provided data set for {data_type}.')
    return data


def _check_string_column(data: pd.DataFrame, column_name: str, data_type: str):
    dtype = data[column_name].dtype
    if dtype != object:
        raise TypeError(f'Column {column_name} must be filled with strings. '
                        f'You specified {dtype} in the provided data set for {data_type}.')


def _check_column_exists(data: pd.DataFrame, column_name: str, data_type: str, error: bool = False) -> bool:
    """Returns whether a column exists in the data or errors by request."""
    if column_name not in data.columns:
        if error:
            raise ValueError(f'Column {column_name} must be present in data set {data_type} but was not found.')
        return False
    return True


def _check_one_value_column(data: pd.DataFrame, column_name: str, data_type: str):
    """Ensures there is exactly one value in a dataframe column."""
    if len(data[column_name].unique()) != 1:
        raise ValueError(f'A data set may represent exactly one {column_name}.  The provided '
                         f'data set {data_type} contains the {column_name}s {list(data[column_name].unique())}')


def _clean_missing(data: pd.DataFrame, column_name: str, data_type: str,
                   error: bool = True, fill_value: Union[str, float, int] = 0.0) -> pd.DataFrame:
    missing = data[column_name].isnull()
    if missing.any():
        if error:
            raise ValueError(f'Data set {data_type} contains null values in column {column_name}.')
        else:
            logger.warning(f'Data set {data_type} contains null values in column {column_name}. '
                           f'Filling missing values with {fill_value}.')
            data.loc[missing, column_name] = fill_value
    return data
