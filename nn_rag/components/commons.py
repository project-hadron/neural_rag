from typing import Any
import pandas as pd
import pyarrow as pa
from ds_core.components.core_commons import CoreCommons


class Commons(CoreCommons):

    @staticmethod
    def list_formatter(value: Any) -> list:
        if isinstance(value, pd.Series):
            return value.to_list()
        if isinstance(value, pd.DataFrame):
            return value.iloc[0].to_list()
        return CoreCommons.list_formatter(value)

    @staticmethod
    def report(canonical: pd.DataFrame, index_header: [str, list]=None, bold: [str, list]=None,
               large_font: [str, list]=None, precision: int=None):
        """ generates a stylised report

        :param canonical: the DataFrame to report on
        :param index_header: the header to index on
        :param bold: any columns to make bold
        :param large_font: any columns to enlarge
        :param precision: a numeric precision for floating points
        :return: stylised report DataFrame
        """
        precision = precision if isinstance(precision, dict) else 4
        index_header = Commons.list_formatter(index_header)
        pd.set_option('max_colwidth', 200)
        pd.set_option('expand_frame_repr', True)
        bold = Commons.list_formatter(bold)
        bold += index_header
        large_font = Commons.list_formatter(large_font)
        large_font += index_header
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        for header in index_header:
            prev = ''
            for idx in range(len(canonical[header])):
                if canonical[header].iloc[idx] == prev:
                    canonical[header].iloc[idx] = ''
                else:
                    prev = canonical[header].iloc[idx]
        canonical = canonical.reset_index(drop=True)
        df_style = canonical.style.set_table_styles(style)
        _ = df_style.format(precision=precision)
        _ = df_style.set_properties(**{'text-align': 'left'})
        if len(bold) > 0:
            _ = df_style.set_properties(subset=bold, **{'font-weight': 'bold'})
        if len(large_font) > 0:
            _ = df_style.set_properties(subset=large_font, **{'font-size': "120%"})
        return df_style

    @staticmethod
    def table_report(t: pa.Table, top: int=None, headers: [str, list]=None, d_type: [str, list]=None,
                     regex: [str, list]=None, drop: bool=None, index_header: [str, list]=None, bold: [str, list]=None,
                     large_font: [str, list]=None):
        """ generates a stylised version of the pyarrow table """
        top = top if isinstance(top, int) else 10
        report = Commons.filter_columns(t.slice(0, top), headers=headers, d_types=d_type, regex=regex, drop=drop)
        df = report.to_pandas()
        return Commons.report(df, index_header=index_header, bold=bold, large_font=large_font)
