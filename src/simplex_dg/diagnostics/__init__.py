from simplex_dg.diagnostics.errors import (
    ErrorReport,
    error_report,
    l2_error,
    linf_error,
    relative_l2_error,
)
from simplex_dg.diagnostics.convergence import (
    ConvergenceRow,
    estimate_log2_rates,
    format_convergence_table,
    rows_to_dicts_with_rates,
    write_convergence_csv,
)

__all__ = [
    "ErrorReport",
    "l2_error",
    "relative_l2_error",
    "linf_error",
    "error_report",
    "ConvergenceRow",
    "estimate_log2_rates",
    "rows_to_dicts_with_rates",
    "format_convergence_table",
    "write_convergence_csv",
]