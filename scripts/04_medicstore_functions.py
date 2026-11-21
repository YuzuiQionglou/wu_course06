"""MedicStore - Sales Reporting (script with functions).

Same analysis as the notebook `notebooks/02_medicstore_notebook.ipynb`, organised in
functions: each step of the pipeline is a function, and `main()` calls them in order.

Run it from the project root:

    uv run scripts/04_medicstore_functions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.drawing.image import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_DIR = DATA_DIR / "input_medicstore"  # the 3 raw files are here
OUTPUT_DIR = DATA_DIR / "output_medicstore_v3"


# --- Objective 1: consolidation ----------------------------------------------


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the two parts of the sales and the mapping table."""
    # The CSV files use ";" as separator
    sales_part1 = pd.read_csv(input_dir / "Sales_MedicStore_part1.csv", sep=";")
    sales_part2 = pd.read_csv(input_dir / "Sales_MedicStore_part2.csv", sep=";")
    # Note: the file name contains a space before ".xlsx"
    mapping = pd.read_excel(input_dir / "Mapping .xlsx")
    return sales_part1, sales_part2, mapping


def consolidate(
    sales_part1: pd.DataFrame, sales_part2: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate the two parts, merge them with the mapping table, compute total_sales."""
    sales = pd.concat([sales_part1, sales_part2], ignore_index=True)
    df_final = sales.merge(mapping, on="Sub-Category", how="left")

    assert len(df_final) == len(sales), "The merge changed the number of lines"
    assert df_final["Category"].notna().all(), (
        "Some sub-categories are not in the mapping table"
    )

    # total_sales = Quantity x Unit Price
    df_final["total_sales"] = df_final["Quantity"] * df_final["Unit Price"]

    # Convert the dates (format dd/mm/yyyy) so that Excel reads them as real dates
    for column in ["Order Date", "Ship Date"]:
        df_final[column] = pd.to_datetime(df_final[column], format="%d/%m/%Y")
    return df_final


# --- Objective 2: questions ----------------------------------------------------


def compute_shapes(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Technical Q1: number of lines and columns of each dataset."""
    return pd.DataFrame(
        {
            "dataset": list(datasets),
            "nb_lines": [df.shape[0] for df in datasets.values()],
            "nb_columns": [df.shape[1] for df in datasets.values()],
        }
    )


def answer_business_questions(df: pd.DataFrame) -> dict[str, object]:
    """Business Q1 to Q5: return a dictionary {question: answer}."""
    # One order can contain several lines (one per product): count the unique Order IDs
    total_orders = df["Order ID"].nunique()
    total_sales_amount = df["total_sales"].sum()

    top3_customers = (
        df.groupby("Customer Name")["Order ID"]
        .nunique()
        .sort_values(ascending=False)
        .head(3)
    )
    top3_text = ", ".join(f"{name} ({n} orders)" for name, n in top3_customers.items())

    total_sales_france = df.loc[df["Country"] == "France", "total_sales"].sum()
    ratio_sales_france = total_sales_france / total_sales_amount

    return {
        "How many orders do we have": total_orders,
        "What is the total sales amount": total_sales_amount,
        "Who are the top 3 customers (order count)": top3_text,
        "What is the total Sales for France": total_sales_france,
        "How much does the total Sales for France represent compare to the total global Sales": ratio_sales_france,
    }


# --- Objective 3: graphs -------------------------------------------------------


def save_bar_chart(
    series: pd.Series, title: str, path: Path, horizontal: bool = False
) -> Path:
    """Save a bar chart of `series` as a PNG file."""
    fig, ax = plt.subplots()
    if horizontal:
        series.plot.barh(ax=ax, title=title)
    else:
        series.plot.bar(ax=ax, title=title, rot=0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_graphs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create the 3 graphs of Objective 3 and return their paths."""
    return [
        save_bar_chart(
            df.groupby("Country")["Order ID"].nunique().sort_values(),
            "Number of orders per country",
            output_dir / "orders_per_country.png",
            horizontal=True,
        ),
        save_bar_chart(
            df.groupby("Category")["total_sales"].sum().sort_values(),
            "Total sales per category",
            output_dir / "sales_per_category.png",
        ),
        save_bar_chart(
            df.groupby(df["Order Date"].dt.year)["total_sales"].sum(),
            "Total sales per year",
            output_dir / "sales_per_year.png",
        ),
    ]


# --- Objective 4: export -------------------------------------------------------


def export_analysis(
    path: Path,
    df_final: pd.DataFrame,
    df_shapes: pd.DataFrame,
    df_describe: pd.DataFrame,
    df_analysis: pd.DataFrame,
    graphs: list[Path],
) -> None:
    """Export the tables and the graphs into one Excel file, one sheet each."""
    with pd.ExcelWriter(path) as writer:
        df_final.to_excel(writer, sheet_name="final_dataset", index=False)
        df_shapes.to_excel(writer, sheet_name="technical_q1_shapes", index=False)
        df_describe.to_excel(writer, sheet_name="technical_q5_describe")
        df_analysis.to_excel(writer, sheet_name="business_questions", index=False)

        sheet = writer.book.create_sheet("graphs")
        for i, graph in enumerate(graphs):
            sheet.add_image(Image(graph), f"A{1 + 25 * i}")


# --- Pipeline ------------------------------------------------------------------


def main() -> None:
    """Run the whole pipeline: load, consolidate, analyse, export."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Objective 1
    sales_part1, sales_part2, mapping = load_data(INPUT_DIR)
    df_final = consolidate(sales_part1, sales_part2, mapping)
    df_final.to_excel(OUTPUT_DIR / "medicstore_final.xlsx", index=False)

    # Objective 2
    df_shapes = compute_shapes(
        {
            "Sales_MedicStore_part1.csv": sales_part1,
            "Sales_MedicStore_part2.csv": sales_part2,
            "Mapping.xlsx": mapping,
            "final dataset": df_final,
        }
    )
    df_describe = df_final.describe(include="all")
    dict_analysis = {
        "How many lines are in Sales_MedicStore_part1.csv": len(sales_part1),
        "How many lines are in Sales_MedicStore_part2.csv": len(sales_part2),
        "How many lines are in Mapping.xlsx": len(mapping),
        **answer_business_questions(df_final),
    }
    df_analysis = pd.DataFrame(
        list(dict_analysis.items()), columns=["Question", "Answer"]
    )

    # Objectives 3 and 4
    graphs = make_graphs(df_final, OUTPUT_DIR)
    export_analysis(
        OUTPUT_DIR / "medicstore_analysis.xlsx",
        df_final,
        df_shapes,
        df_describe,
        df_analysis,
        graphs,
    )

    print(df_analysis.to_string(index=False))
    print(f"\nAnalysis exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
