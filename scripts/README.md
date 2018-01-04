The option "GUI" requires the [PandasDataFrameGUI](https://github.com/bluenote10/PandasDataFrameGUI)

```bash
usage: summarize_results.py [-h] [-p PERFORMANCE] [-f FILTER_ROWS]
                            [-v VERBOSE] [-g]
                            PATH SUMMARY

Generates a summary of all the experiments in the subfolders of the specified
path

positional arguments:
  PATH                  Path with the result folders to summarize.
  SUMMARY               Path to store the summary.

optional arguments:
  -h, --help            show this help message and exit
  -p PERFORMANCE, --performance PERFORMANCE
                        Filter all the datasets with supervised performance
                        lower than the specified value
  -f FILTER_ROWS, --filter FILTER_ROWS
                        Dictionary with columns and filters.
  -v VERBOSE, --verbose VERBOSE
                        Dictionary with columns and filters.
  -g, --gui             Open a GUI with the dataframe.
```
