"""Upload a dummy CSV to the RAG volume for testing execute_python tool."""
import tempfile
import modal
modal.enable_output()

from agents.infra.shared import app, rag_vol

CSV_CONTENT = """product,category,region,units_sold,revenue,cost
Widget A,Electronics,North,142,7100.00,4260.00
Widget B,Electronics,South,89,4450.00,2670.00
Gadget X,Accessories,North,311,9330.00,4665.00
Gadget Y,Accessories,East,204,6120.00,3060.00
Gizmo Pro,Electronics,West,67,6700.00,4020.00
Gizmo Lite,Electronics,East,198,9900.00,5940.00
Doohickey,Accessories,South,423,8460.00,3384.00
Thingamajig,Tools,North,56,2800.00,1680.00
Whatchamacallit,Tools,West,134,5360.00,2680.00
Contraption,Tools,East,78,3900.00,2340.00
Widget A,Electronics,East,99,4950.00,2970.00
Widget B,Electronics,West,113,5650.00,3390.00
Gadget X,Accessories,South,267,8010.00,4005.00
Gadget Y,Accessories,West,189,5670.00,2835.00
Gizmo Pro,Electronics,North,44,4400.00,2640.00
"""


@app.local_entrypoint()
def main():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write(CSV_CONTENT)
        tmp_path = f.name

    with rag_vol.batch_upload(force=True) as batch:
        batch.put_file(tmp_path, "/rag/docs/sales_data.csv")

    print("Uploaded sales_data.csv to sandbox-rag volume at /rag/docs/sales_data.csv")
    print("(maps to /data/rag/docs/sales_data.csv inside the sandbox)")
