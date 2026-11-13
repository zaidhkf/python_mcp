"""Shared dataset loading and metadata models."""

from data_manager.models import HeaderDetection, TableResult
from data_manager.table_reader import read_table

__all__ = ['HeaderDetection', 'TableResult', 'read_table']
