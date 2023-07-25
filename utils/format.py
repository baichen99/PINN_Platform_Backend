from rich.console import Console
from rich.table import Table
from rich.style import Style


def format_to_scientific_notation(number, decimals=4):
    return "{:.{}e}".format(number, decimals)

def print_table(cols, data):
    # 示例颜色样式
    header_style = Style(color="white", bgcolor="blue", bold=True)
    odd_row_style = Style(bgcolor="#eeeeee")
    even_row_style = Style(bgcolor="#e0e0e0")

    table = Table(title="Logs", show_header=True, header_style=header_style)

    # 添加表头
    for col in cols:
        table.add_column(col)

    # 添加数据行，并根据奇偶行应用不同的样式
    for i, row in enumerate(data):
        row_style = even_row_style if i % 2 == 0 else odd_row_style
        table.add_row(*row, style=row_style)

    console = Console()
    console.print(table)