"""Summary
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod
import os


class TrainTestBase(object):

    """Summary

    Attributes:
        data (TYPE): Description
        export_to_file (TYPE): Description
        import_from_file (TYPE): Description
        output_dir (TYPE): Description
        test (TYPE): Description
        test_size (TYPE): Description
        train (TYPE): Description
    """

    __metaclass__ = ABCMeta

    def __init__(self, test_size, import_from_file=True, export_to_file=True,
                 output_dir='', concat_features_label=True):
        """Summary

        Args:
            test_size (TYPE): Description
            import_from_file (bool, optional): Description
            export_to_file (bool, optional): Description
            output_dir (str, optional): Description
        """
        self.test_size = test_size
        self.import_from_file = import_from_file
        self.export_to_file = export_to_file
        self.output_dir = output_dir
        self.concat_features_label = concat_features_label

    def generate_train_test(self):
        """Summary

        No Longer Raises:
            NotImplementedError: Description

        No Longer Returned:
            TYPE: Description
        """
        if self.import_from_file:
            self.data = self.load_data()

        X, y = self.data.iloc[:, :-1], self.data.iloc[:, -1]

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size)

        if self.export_to_file:
            self.export_data(X_train, X_test, y_train, y_test)

    def export_data(self, X_train, X_test, y_train, y_test):
        """Summary
        """
        print 'Export train test data to csv'
        if self.concat_features_label:
            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)
            train.to_csv(os.path.join(
                self.output_dir, 'train.csv'), index=False)
            test.to_csv(os.path.join(
                self.output_dir, 'test.csv'), index=False)
        else:
            X_train.to_csv(os.path.join(
                self.output_dir, 'X_train.csv'), index=False)
            y_train.to_csv(os.path.join(
                self.output_dir, 'y_train.csv'), index=False)
            X_test.to_csv(os.path.join(
                self.output_dir, 'X_test.csv'), index=False)
            y_test.to_csv(os.path.join(
                self.output_dir, 'y_test.csv'), index=False)

    @abstractmethod
    def load_data(self):
        """Summary

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError()


class TrainTestFromFile(TrainTestBase):

    """Summary

    Attributes:
        delimiter (TYPE): Description
        excluded_cols (TYPE): Description
        file_path (TYPE): Description
        include_header (TYPE): Description
    """

    def __init__(self, test_size, file_path, excluded_cols, delimiter=',',
                 include_header=True, export_to_file=True, output_dir='',
                 concat_features_label=True):
        """Summary

        Args:
            test_size (TYPE): the percentage of test dataset
            file_path (TYPE): file path
            excluded_cols (TYPE): array of excluded column names
            delimiter (str, optional): Description
            include_header (bool, optional): Description
            export_to_file (bool, optional): Description
            output_dir (str, optional): Description
        """
        TrainTestBase.__init__(self, test_size=test_size,
                               import_from_file=True,
                               export_to_file=export_to_file,
                               output_dir=output_dir,
                               concat_features_label=concat_features_label)
        self.file_path = file_path
        self.excluded_cols = excluded_cols
        self.delimiter = delimiter
        self.include_header = include_header

    def load_data(self):
        """Generate train & test subset from one dataset

        Deleted Parameters:
            file_path (TYPE): Description
            delimiter (str, optional): Description
            include_header (bool, optional): Description

        Returns:
            TYPE: Description
        """
        print 'Load data from csv'
        data = pd.read_csv(self.file_path, delimiter=self.delimiter)
        data = data.drop(self.excluded_cols, axis=1)
        return data
