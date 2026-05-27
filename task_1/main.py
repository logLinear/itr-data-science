import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ============================================================
# ****** STEP 1: DATA INGESTION ******
# ============================================================
# Loaded first 10,000 rows during initial development.
# Removed nrows= for full run.

df = pd.read_csv("task_1/customers-100000.csv")

# --- Sanity checks grouped into one reusable function ---
def sanity_check(df, label=""):
    if label:
        print(f"\n{'='*50}")
        print(f"  SANITY CHECK — {label}")
        print(f"{'='*50}")
    print(f"\n[Shape] {df.shape}")
    print(f"\n[First 5 rows]\n{df.head()}")
    print(f"\n[Column dtypes]\n{df.dtypes}")
    print(f"\n[Null counts]\n{df.isnull().sum()}")

sanity_check(df, label="After Ingestion")


# ============================================================
# ****** STEP 2: DEDUPLICATION ******
# ============================================================

#   - Exact rows:            every field is identical (true copy)
#   - Same Email:            strongest real-world unique identifier
#   - Same Customer Id:      should never repeat
#   - Same Name + Company:   catches person-level duplicates across
#                            different contact entries

print("\n\n" + "="*50)
print("  STEP 2: DEDUPLICATION")
print("="*50)

# --- 2a. Exact duplicate rows ---
exact_dupes = df.duplicated().sum()
print(f"\n[Exact duplicate rows]: {exact_dupes}")
df = df.drop_duplicates()

# --- 2b. Partial duplicate: same Email ---
email_dupes = df.duplicated(subset=["Email"]).sum()
print(f"[Duplicate Emails]: {email_dupes}")
df = df.drop_duplicates(subset=["Email"], keep="first")

# --- 2c. Partial duplicate: same Customer Id ---
id_dupes = df.duplicated(subset=["Customer Id"]).sum()
print(f"[Duplicate Customer Ids]: {id_dupes}")
df = df.drop_duplicates(subset=["Customer Id"], keep="first")

# --- 2d. Cross-column: same First Name + Last Name + Company ---
name_co_dupes = df.duplicated(subset=["First Name", "Last Name", "Company"]).sum()
print(f"[Duplicate Name + Company combos]: {name_co_dupes}")
df = df.drop_duplicates(subset=["First Name", "Last Name", "Company"], keep="first")

# --- Validation ---
assert df.duplicated().sum() == 0, "Exact duplicates still present!"
assert df.duplicated(subset=["Email"]).sum() == 0, "Email duplicates still present!"
assert df.duplicated(subset=["Customer Id"]).sum() == 0, "Customer Id duplicates still present!"
print("\n[OK] All deduplication assertions passed.")

# --- Unique value spot-check ---
print(f"\n[Unique Emails]: {df['Email'].nunique()}")
print(f"[Unique Customer Ids]: {df['Customer Id'].nunique()}")
print(f"[Unique Countries]: {df['Country'].unique()[:10]} ...")  # first 10 as sample

sanity_check(df, label="After Deduplication")


# ============================================================
# ****** STEP 3: COLUMN MANAGEMENT ******
# ============================================================
#   DROP   'Index'       — pure row number from the CSV; after
#                          deduplication the original index no
#                          longer maps to anything meaningful.
#                          
#   KEEP   'Customer Id' — a real identifier useful for joins or
#                          lookups downstream.

#   RENAME — snake_case is the Python convention; consistent
#             naming avoids df["First Name"] (space = friction).

#   REORDER — group identity > contact > location > metadata

print("\n\n" + "="*50)
print("  STEP 3: COLUMN MANAGEMENT")
print("="*50)

# --- 3a. Drop irrelevant columns ---
df = df.drop(columns=["Index"])
print("\n[Dropped] 'Index' column.")

# --- 3b. Rename columns for clarity (snake_case) ---
df = df.rename(columns={
    "Customer Id"       : "customer_id",
    "First Name"        : "first_name",
    "Last Name"         : "last_name",
    "Company"           : "company",
    "City"              : "city",
    "Country"           : "country",
    "Phone 1"           : "phone_primary",
    "Phone 2"           : "phone_secondary",
    "Email"             : "email",
    "Subscription Date" : "subscription_date",
    "Website"           : "website",
})
print("[Renamed] All columns to snake_case.")

# --- 3c. Reorder columns logically ---
# Identity → Name → Company → Location → Contact → Metadata
df = df.reindex(columns=[
    "customer_id",
    "first_name", "last_name",
    "company",
    "city", "country",
    "phone_primary", "phone_secondary",
    "email",
    "subscription_date",
    "website",
])
print("[Reordered] Columns: identity → name → company → location → contact → metadata.")

sanity_check(df, label="After Column Management")


# ============================================================
# ****** STEP 4: MISSING VALUE HANDLING ******
# ============================================================
# Strategy:
#   1. Quantify missingness per column.
#   2. If a column is >70% missing → drop it entirely 
#   3. For remaining missing values:
#        - Numerical → impute with median (robust to outliers).
#        - Categorical/text → impute with mode (most frequent).
#   4. If no missing values are found → print and skip, as
#      imputing a clean dataset would corrupt it.

print("\n\n" + "="*50)
print("  STEP 4: MISSING VALUE HANDLING")
print("="*50)

null_counts = df.isna().sum()
null_pct    = (null_counts / len(df)) * 100
missing_summary = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
print(f"\n[Missing value summary]\n{missing_summary}")

# --- 4a. Drop columns with >70% missing ---
cols_to_drop = missing_summary[missing_summary["null_pct"] > 70].index.tolist()
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"\n[Dropped columns >70% missing]: {cols_to_drop}")
else:
    print("\n[OK] No columns exceed 70% missing threshold.")

# --- 4b. Impute remaining missing values ---
cols_with_nulls = df.columns[df.isna().any()].tolist()

if not cols_with_nulls:
    print("[OK] No missing values found — skipping imputation.")
else:
    print(f"\n[Columns requiring imputation]: {cols_with_nulls}")

    numerical_cols   = df[cols_with_nulls].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[cols_with_nulls].select_dtypes(exclude=[np.number]).columns.tolist()

    # Median imputation for numerical columns
    if numerical_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        print(f"[Imputed median] Numerical columns: {numerical_cols}")

    # Mode imputation for categorical columns
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        print(f"[Imputed mode] Categorical columns: {categorical_cols}")

# --- Validation ---
assert df.isna().sum().sum() == 0, "Missing values still present after imputation!"
print("\n[OK] Missing value assertion passed — zero nulls in dataframe.")

sanity_check(df, label="After Missing Value Handling")


# ============================================================
# ****** STEP 5: DATA TYPE CORRECTION ******
# ============================================================
#   - subscription_date: stored as a plain string from CSV;
#   - phone columns: may contain formatting characters like
#     parentheses, dashes, dot.
#   - All other columns are already correct object (string)
#     types for this dataset.

print("\n\n" + "="*50)
print("  STEP 5: DATA TYPE CORRECTION")
print("="*50)

# --- 5a. Convert subscription_date to datetime ---
df["subscription_date"] = pd.to_datetime(df["subscription_date"], errors="coerce")
print(f"\n[Converted] 'subscription_date' → {df['subscription_date'].dtype}")

# --- 5b. Check for unparseable dates (set to NaT by errors="coerce") ---
invalid_dates = df["subscription_date"].isna().sum()
if invalid_dates > 0:
    print(f"[Warning] {invalid_dates} subscription_date values could not be parsed → set to NaT.")
else:
    print("[OK] All subscription_date values parsed successfully.")

# --- 5c. Normalise phone columns to digits-only strings ---
for phone_col in ["phone_primary", "phone_secondary"]:
    df[phone_col] = df[phone_col].astype(str).str.replace(r"\D", "", regex=True)
    print(f"[Normalised] '{phone_col}' → digits-only string")

# --- Validation ---
# FIX: pandas 2.0+ uses datetime64[us] instead of datetime64[ns], so
# checking the exact dtype string had broken on my current version.
assert df["subscription_date"].dtype.kind == "M", \
    "subscription_date is not a datetime type!"
assert df["phone_primary"].str.isnumeric().all(), \
    "phone_primary contains non-numeric characters!"
print("\n[OK] Data type assertions passed.")

sanity_check(df, label="After Data Type Correction")


# ============================================================
# ****** STEP 6: FORMAT STANDARDISATION ******
# ============================================================
# Goals:
#   - Text columns: lowercase + strip whitespace so that
#     "  New York " and "new york" are the same value.
#     Applied to name/company/city/country/email/website.
#   - email: fully lowercase as emails are case-insensitive.
#   - country / city: title-case AFTER lowercasing so that
#     "eritrea" → "Eritrea" for readability in reports.

print("\n\n" + "="*50)
print("  STEP 6: FORMAT STANDARDISATION")
print("="*50)

# --- 6a. Strip whitespace from all string columns ---
str_cols = df.select_dtypes(include="object").columns.tolist()
for col in str_cols:
    df[col] = df[col].str.strip()
print(f"[Stripped whitespace] Columns: {str_cols}")

# --- 6b. Lowercase text identity/contact fields ---
for col in ["first_name", "last_name", "company", "email", "website"]:
    df[col] = df[col].str.lower()
print("[Lowercased] first_name, last_name, company, email, website")

# --- 6c. Title-case geographic fields for readability ---
for col in ["city", "country"]:
    df[col] = df[col].str.title()
print("[Title-cased] city, country")

# --- Validation ---
assert not df["email"].str.contains(r"[A-Z]").any(), \
    "email column still contains uppercase characters!"
assert not df["city"].str.islower().any(), \
    "city column has fully lowercase entries — title-case failed!"
print("\n[OK] Format standardisation assertions passed.")

sanity_check(df, label="After Format Standardisation — FINAL")


# ============================================================
# ****** PIPELINE COMPLETE ******
# ============================================================
print("\n\n" + "="*50)
print("  PIPELINE COMPLETE")
print("="*50)
print(f"\nFinal dataframe shape : {df.shape}")
print(f"Columns               : {df.columns.tolist()}")
print(f"Dtypes summary:\n{df.dtypes}")
print("\nReady to export or pass to next analysis stage.")

