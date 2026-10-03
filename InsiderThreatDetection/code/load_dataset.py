import pandas as pd

file_path = r"C:\Users\useer\OneDrive\Documents\InsiderThreatDetection\dataset\r1\r1\logon.csv"

# Load dataset
data = pd.read_csv(file_path)

# Convert date to datetime
data["date"] = pd.to_datetime(data["date"])

print("Dataset loaded successfully!")
print("Total records:", len(data))

# Keep only Logon events
logons = data[data["activity"] == "Logon"].copy()

# Extract hour
logons["hour"] = logons["date"].dt.hour

# ---------------------------------------
# 1. Unusual login time
# ---------------------------------------

logons["unusual_time"] = (
    (logons["hour"] < 9) |
    (logons["hour"] >= 18)
)

# ---------------------------------------
# 2. Login frequency
# ---------------------------------------

user_login_count = logons.groupby("user").size()

logons["user_login_count"] = logons["user"].map(user_login_count)

# ---------------------------------------
# 3. First risk check
# ---------------------------------------

logons["risk_score"] = 0

# Unusual login
logons.loc[
    logons["unusual_time"],
    "risk_score"
] += 1

# High login frequency
logons.loc[
    logons["user_login_count"] > 1000,
    "risk_score"
] += 1

# ---------------------------------------
# 4. Warning generation
# ---------------------------------------

logons["warning"] = "No Warning"

logons.loc[
    logons["risk_score"] >= 1,
    "warning"
] = "Warning: Suspicious Activity"

# ---------------------------------------
# 5. Second-check mechanism
# ---------------------------------------

logons["second_check"] = "Not Required"

# Activities with risk score 1 are checked again
logons.loc[
    logons["risk_score"] == 1,
    "second_check"
] = "Second Check Required"

# Activities with score 2 or more
# receive high-risk status
logons.loc[
    logons["risk_score"] >= 2,
    "second_check"
] = "Confirmed High Risk"

# ---------------------------------------
# 6. Final status
# ---------------------------------------

logons["final_status"] = "Normal"

logons.loc[
    logons["risk_score"] == 1,
    "final_status"
] = "Needs Verification"

logons.loc[
    logons["risk_score"] >= 2,
    "final_status"
] = "High Risk"

# ---------------------------------------
# Display results
# ---------------------------------------

print("\nWarning and second-check system completed!")

print("\nFinal status distribution:")
print(logons["final_status"].value_counts())

print("\nSample results:")

print(
    logons[
        [
            "date",
            "user",
            "pc",
            "risk_score",
            "warning",
            "second_check",
            "final_status"
        ]
    ].head(15)
)