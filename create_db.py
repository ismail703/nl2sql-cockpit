import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv
import os


# docker run -d 
# --name cockpit-postgres 
# -e POSTGRES_USER=root 
# -e POSTGRES_PASSWORD=1234 
# -e POSTGRES_DB=cockpit_db 
# -p 5432:5432 
# postgres:latest

load_dotenv()
COCKPIT_DB_URI = os.getenv("COCKPIT_DB_URI")

engine = create_engine(COCKPIT_DB_URI)

create_table_sql = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id_date   TIMESTAMP,
    id_day    INTEGER,
    id_week   INTEGER,
    id_month  INTEGER,
    id_year   INTEGER,
    valeur_d1 REAL,
    split     TEXT,
    segment   TEXT,
    kpi       TEXT
);
"""

table_names = [
    "bda_activations", "bda_encaissement", "bda_ftth_delai_raccordement",
    "bda_ftth_reclamations", "bda_parc", "bda_revenu_billing", "bda_recharge"
]

with engine.begin() as conn:
    for table_name in table_names:
        try:
            conn.execute(text(create_table_sql.format(table_name=table_name)))
            
            csv_file = Path(f"Cockpit/{table_name}_mock.csv")
            
            if not csv_file.exists():
                print(f"Skipping {table_name}: File {csv_file} not found.")
                continue

            df = pd.read_csv(csv_file, sep=';')

            df["id_date"] = pd.to_datetime(df["id_date"], errors="coerce")

            df.to_sql(table_name, conn, if_exists="append", index=False)
            print(f"{table_name} created and seeded with {len(df)} rows.")

        except Exception as e:
            print(f"Error processing {table_name}: {e}")

print("\nProcess finished. Check the logs above for any skipped tables.")