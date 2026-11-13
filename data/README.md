# Dataset

`B2B_Transaction_Data_.xlsx` – B2B transaction data of a book distributor (2020), provided on Ninova for ISL 467E.

Because the original workbook (40 MB) exceeds GitHub's 25 MB browser upload limit, the same data is stored here as
two lossless xz-compressed CSV parts:

| File | Rows |
|---|---|
| `B2B_Transaction_Data_part1.csv.xz` | 332,077 |
| `B2B_Transaction_Data_part2.csv.xz` | 332,077 |

Columns: InvoiceNo, CustomerID, InvoiceDate, StockCode, Description, Category, City, Quantity, NetPrice, UnitPrice.
`data_pipeline.py` concatenates both parts on load; if the original Excel file is placed in this folder it is used instead.
