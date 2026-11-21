"""MedicStore - Sales Reporting: Streamlit app.

Reproduces the whole pipeline (ingestion, transformation, macro analysis)
and lets the user export the insights.

Run it from the project root:

    uv run streamlit run app/medicstore_app.py
"""

import io
import re
import zipfile
from pathlib import Path
from typing import IO

import altair as alt
import matplotlib

matplotlib.use("Agg")  # no GUI backend: the figures are rendered by Streamlit

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from openpyxl.drawing.image import Image

PART_COLUMNS = {
    "City",
    "Country",
    "Customer Name",
    "Manufacturer",
    "Order Date",
    "Order ID",
    "Product Name",
    "Region",
    "Segment",
    "Ship Date",
    "Ship Mode",
    "State/Province",
    "Sub-Category",
    "Quantity",
    "Unit Price",
}
MAPPING_COLUMNS = {"Sub-Category", "Category"}
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

Source = IO[bytes]

# Colour of the graphs (same teal as the theme in .streamlit/config.toml)
BAR_COLOR = "#0f766e"


# --- Ingestion and transformation ----------------------------------------------


def check_columns(df: pd.DataFrame, required: set[str], name: str) -> pd.DataFrame:
    """Check that a table contains the expected columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing the columns: {sorted(missing)}")
    return df


def load_data(
    part_files: list[Source], mapping_file: Source
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Ingestion: load any number of sales parts and the mapping table.

    The parts are returned in a dictionary {file name: DataFrame}.
    """
    parts = {}
    for file in part_files:
        name = getattr(file, "name", "part")
        if name in parts:
            raise ValueError(f"The file {name} was loaded twice")
        # The sales CSV files use ";" as separator
        parts[name] = check_columns(pd.read_csv(file, sep=";"), PART_COLUMNS, name)
    mapping = check_columns(pd.read_excel(mapping_file), MAPPING_COLUMNS, "mapping")
    return parts, mapping


def consolidate(parts: dict[str, pd.DataFrame], mapping: pd.DataFrame) -> pd.DataFrame:
    """Transformation: concatenate the parts, merge the mapping, compute total_sales."""
    sales = pd.concat(parts.values(), ignore_index=True)
    df_final = sales.merge(mapping, on="Sub-Category", how="left")

    if len(df_final) != len(sales):
        raise ValueError(
            "The merge changed the number of lines: "
            "Sub-Category must be unique in the mapping table"
        )
    unmapped = df_final.loc[df_final["Category"].isna(), "Sub-Category"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Sub-categories missing from the mapping: {list(unmapped)}")

    # total_sales = Quantity x Unit Price
    df_final["total_sales"] = df_final["Quantity"] * df_final["Unit Price"]
    # Dates are written dd/mm/yyyy
    for column in ["Order Date", "Ship Date"]:
        df_final[column] = pd.to_datetime(df_final[column], format="%d/%m/%Y")
    df_final["Year"] = df_final["Order Date"].dt.year
    return df_final


def quality_checks(
    parts: dict[str, pd.DataFrame], df_final: pd.DataFrame
) -> pd.DataFrame:
    """Return a table of checks proving that the consolidation is correct."""
    same_length = len(df_final) == sum(len(df) for df in parts.values())
    checks = {
        "Check": [
            "Lines after concatenation = sum of the parts",
            "Lines unchanged by the mapping merge",
            "Every line has a category",
            "Missing values in the final dataset",
        ],
        "Result": [
            same_length,
            same_length,
            bool(df_final["Category"].notna().all()),
            int(df_final.isna().sum().sum()),
        ],
    }
    return pd.DataFrame(checks).astype({"Result": str})


# --- Macro analysis ------------------------------------------------------------


def compute_shapes(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Number of lines and columns of each dataset."""
    shapes = {
        "dataset": list(datasets),
        "nb_lines": [df.shape[0] for df in datasets.values()],
        "nb_columns": [df.shape[1] for df in datasets.values()],
    }
    return pd.DataFrame(shapes)


def headline_kpis(df: pd.DataFrame) -> dict[str, float]:
    """The key figures displayed at the top of the analysis page."""
    total_orders = df["Order ID"].nunique()
    total_sales = df["total_sales"].sum()
    return {
        "Orders": total_orders,
        "Total sales": total_sales,
        "Quantity sold": int(df["Quantity"].sum()),
        "Average sales per order": total_sales / total_orders,
    }


def answer_business_questions(df: pd.DataFrame) -> dict[str, object]:
    """Return a dictionary {question: answer} for the business questions."""
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
    ratio_sales_france = (
        total_sales_france / total_sales_amount if total_sales_amount else None
    )

    return {
        "How many orders do we have": total_orders,
        "What is the total sales amount": total_sales_amount,
        "Who are the top 3 customers (order count)": top3_text,
        "What is the total Sales for France": total_sales_france,
        "How much does the total Sales for France represent compare to the total global Sales": ratio_sales_france,
    }


def build_insights_table(
    parts: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    df_final: pd.DataFrame,
    df_filtered: pd.DataFrame,
) -> pd.DataFrame:
    """Question / Answer table: dataset sizes, then business questions on the filtered data."""
    answers = {
        **{f"How many lines are in {name}": len(df) for name, df in parts.items()},
        "How many lines are in the mapping": len(mapping),
        "Lines selected by the filters": f"{len(df_filtered)} of {len(df_final)}",
        **answer_business_questions(df_filtered),
    }
    table = pd.DataFrame(list(answers.items()), columns=["Question", "Answer"])
    # "n/a" when a value cannot be computed (e.g. France filtered out)
    table["Answer"] = table["Answer"].map(lambda x: "n/a" if x is None else str(x))
    return table


def graph_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The data of each graph, keyed by title (shared by static and interactive charts)."""
    return {
        "Number of orders per country": df.groupby("Country")["Order ID"]
        .nunique()
        .sort_values(ascending=False),
        "Total sales per category": df.groupby("Category")["total_sales"]
        .sum()
        .sort_values(ascending=False),
        "Total sales per year": df.groupby("Year")["total_sales"].sum(),
        "Total sales per sub-category": df.groupby("Sub-Category")["total_sales"]
        .sum()
        .sort_values(ascending=False),
    }


# --- Charts --------------------------------------------------------------------


def bar_chart(series: pd.Series, title: str) -> Figure:
    """Static bar chart (used in the Excel and PNG exports)."""
    fig, ax = plt.subplots()
    series.plot.bar(ax=ax, title=title, color=BAR_COLOR)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def make_graphs(df: pd.DataFrame) -> dict[str, Figure]:
    """Create the static graphs of the analysis, keyed by title."""
    return {
        title: bar_chart(series, title) for title, series in graph_series(df).items()
    }


def interactive_bar_chart(series: pd.Series, title: str) -> alt.Chart:
    """Interactive bar chart (tooltip on hover), bars in the order of the series."""
    data = series.rename("value").rename_axis("category").reset_index()
    data["category"] = data["category"].astype(str)
    return (
        alt.Chart(data, title=title)
        .mark_bar(color=BAR_COLOR)
        .encode(
            x=alt.X(
                "category:N",
                title=None,
                sort=list(data["category"]),
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format=",d")),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("value:Q", title="Value", format=",d"),
            ],
        )
    )


def make_interactive_charts(df: pd.DataFrame) -> dict[str, alt.Chart]:
    """Interactive version of the graphs, displayed in the app, keyed by title."""
    return {
        title: interactive_bar_chart(series, title)
        for title, series in graph_series(df).items()
    }


# --- Export --------------------------------------------------------------------


def figure_to_png(fig: Figure) -> bytes:
    """Render a figure as PNG bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def png_file_name(title: str) -> str:
    """File name of a graph, built from its title."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".png"


def to_excel_bytes(
    df: pd.DataFrame,
    df_shapes: pd.DataFrame,
    df_describe: pd.DataFrame,
    df_insights: pd.DataFrame,
    graphs: dict[str, Figure],
) -> bytes:
    """The analysis report as Excel bytes: one sheet per table, plus a sheet of graphs."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="final_dataset", index=False)
        df_shapes.to_excel(writer, sheet_name="shapes", index=False)
        df_describe.to_excel(writer, sheet_name="describe")
        df_insights.to_excel(writer, sheet_name="business_questions", index=False)

        sheet = writer.book.create_sheet("graphs")
        for i, fig in enumerate(graphs.values()):
            sheet.add_image(Image(io.BytesIO(figure_to_png(fig))), f"A{1 + 25 * i}")
    return buffer.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """A dataset as CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def graphs_to_zip_bytes(graphs: dict[str, Figure]) -> bytes:
    """All the graphs as PNG files in a zip archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for title, fig in graphs.items():
            archive.writestr(png_file_name(title), figure_to_png(fig))
    return buffer.getvalue()


# --- User interface ------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_css(path: Path) -> str:
    """Read the CSS file (cached: no disk access on every rerun)."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_style() -> None:
    """Inject the custom CSS of the app (shared with the VitaCheck app)."""
    css = read_css(ASSETS_DIR / "style.css")
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def display_header() -> None:
    """Centered title, then subtitle, description and author aligned on the left."""
    st.title("MedicStore - Sales Reporting")
    st.markdown(
        """
        <div class="app-description">
            <p class="app-subtitle">Ingestion → Transformation → Analysis → Export</p>
            <p class="app-text">
                MedicStore turns the raw sales files into insights.
                Load the sales parts and the category mapping, and the app consolidates
                them, computes the total sales, answers the key business questions
                (orders, total sales, top customers, weight of France...), draws
                interactive graphs and lets you filter the sales and export the results.
            </p>
            <p class="app-author">
                Author: Qiongyi WU ·
                <a href="mailto:qiongyi.wu@edu.em-lyon.com">qiongyi.wu@edu.em-lyon.com</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def sidebar_header() -> None:
    """Sidebar: company logo and name."""
    with st.sidebar:
        logo_column, name_column = st.columns([1, 3], vertical_alignment="center")
        logo_column.image(str(ASSETS_DIR / "medicstore_logo.svg"), width=48)
        name_column.markdown("**MedicStore**  \nSales Reporting")
        st.divider()


def choose_inputs() -> tuple[list[Source], Source]:
    """Sidebar: the user loads the files here, whatever the number of parts."""
    st.sidebar.header("📁 Data files")
    part_files = st.sidebar.file_uploader(
        "Sales parts (CSV, ';' separator): load as many as you have",
        type="csv",
        accept_multiple_files=True,
    )
    mapping_file = st.sidebar.file_uploader("Category mapping (Excel)", type="xlsx")
    if not part_files or not mapping_file:
        st.info("Load the sales parts and the mapping in the sidebar to start.")
        st.stop()
    return part_files, mapping_file


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar: filter the sales. The analysis and the export use the filtered data."""
    st.sidebar.header("🔎 Filters")
    countries = sorted(df["Country"].unique())
    categories = sorted(df["Category"].unique())
    selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
    selected_categories = st.sidebar.multiselect(
        "Category", categories, default=categories
    )

    mask = df["Country"].isin(selected_countries) & df["Category"].isin(
        selected_categories
    )
    year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
    if year_min < year_max:
        low, high = st.sidebar.slider("Year", year_min, year_max, (year_min, year_max))
        mask &= df["Year"].between(low, high)

    filtered = df[mask]
    st.sidebar.caption(f"{len(filtered):,} of {len(df):,} sales lines selected")
    return filtered


def main() -> None:
    """Run the pipeline and display the four tabs."""
    st.set_page_config(
        page_title="MedicStore - Sales Reporting", page_icon="💊", layout="wide"
    )
    load_style()
    display_header()

    sidebar_header()
    try:
        parts, mapping = load_data(*choose_inputs())
        df_final = consolidate(parts, mapping)
    except ValueError as error:
        st.error(f"Could not run the pipeline: {error}")
        st.stop()

    datasets = {**parts, "mapping": mapping}
    df_filtered = apply_filters(df_final)
    if df_filtered.empty:
        st.warning("No sale matches the filters: widen the selection.")
        st.stop()

    df_shapes = compute_shapes(
        {**datasets, "final dataset": df_final, "filtered dataset": df_filtered}
    )
    df_describe = df_filtered.describe(include="all")
    df_insights = build_insights_table(parts, mapping, df_final, df_filtered)

    tab_ingest, tab_transform, tab_analysis, tab_export = st.tabs(
        ["📥 1. Ingestion", "🔄 2. Transformation", "📊 3. Analysis", "📤 4. Export"]
    )

    with tab_ingest:
        st.subheader("🗂️ Raw files")
        st.dataframe(df_shapes.iloc[: len(datasets)], hide_index=True)
        for name, df in datasets.items():
            with st.expander(f"{name} ({df.shape[0]} lines)"):
                st.dataframe(df.head(20))

    with tab_transform:
        st.subheader("🧩 Consolidation")
        st.markdown(
            f"1. Concatenate the {len(parts)} sales parts\n"
            "2. Left-merge the mapping on `Sub-Category` (adds `Category`)\n"
            "3. Compute `total_sales = Quantity × Unit Price`"
        )
        st.dataframe(quality_checks(parts, df_final), hide_index=True)
        st.write(
            f"Final dataset: {df_final.shape[0]} lines, {df_final.shape[1]} columns"
        )
        st.dataframe(df_final.head(50))
        st.subheader("📈 Summary statistics")
        st.dataframe(df_final.describe())
        st.dataframe(df_final.describe(include=["str"]).astype(str))

    with tab_analysis:
        kpis = headline_kpis(df_filtered)
        cols = st.columns(len(kpis))
        cols[0].metric("Orders", f"{kpis['Orders']:,}", border=True)
        cols[1].metric("Total sales", f"{kpis['Total sales']:,}", border=True)
        cols[2].metric("Quantity sold", f"{kpis['Quantity sold']:,}", border=True)
        cols[3].metric(
            "Average sales per order",
            f"{kpis['Average sales per order']:,.0f}",
            border=True,
        )

        st.subheader("💼 Business questions")
        st.dataframe(df_insights, hide_index=True, width="stretch")

        st.subheader("📉 Graphs")
        left, right = st.columns(2)
        for i, chart in enumerate(make_interactive_charts(df_filtered).values()):
            (left if i % 2 == 0 else right).altair_chart(chart, width="stretch")

    with tab_export:
        st.subheader("⬇️ Download the insights")
        st.caption(
            f"The report and the graphs use the {len(df_filtered):,} filtered sales lines."
        )
        graphs = make_graphs(df_filtered)
        st.download_button(
            "Analysis report (Excel)",
            to_excel_bytes(df_filtered, df_shapes, df_describe, df_insights, graphs),
            file_name="medicstore_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Filtered dataset (CSV)",
            to_csv_bytes(df_filtered),
            file_name="medicstore_final.csv",
            mime="text/csv",
        )
        st.download_button(
            "Graphs (PNG, zip)",
            graphs_to_zip_bytes(graphs),
            file_name="medicstore_graphs.zip",
            mime="application/zip",
        )
    plt.close("all")


main()
