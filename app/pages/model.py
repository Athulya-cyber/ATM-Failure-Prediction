import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

st.title("🤖 ATM Failure Prediction")

st.write(
    "Enter the ATM details below and click **Predict** to determine "
    "whether the ATM will fail in the next 48 hours or not."
)

st.markdown("---")


# -------------------------------------------------
# LOAD SAVED MODEL, ENCODERS AND SCALER
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.joblib"
ENCODER_PATH = BASE_DIR / "models" / "encoders.joblib"

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


# -------------------------------------------------
# ATM INFORMATION
# -------------------------------------------------

st.header("ATM Information")

col1, col2, col3 = st.columns(3)

with col1:
    district = st.selectbox(
        "District",
        encoders["District"].classes_
    )

with col2:
    location_type = st.selectbox(
        "Location Type",
        encoders["Location_Type"].classes_
    )

with col3:
    manufacturer = st.selectbox(
        "Manufacturer",
        encoders["Manufacturer"].classes_
    )


# -------------------------------------------------
# ATM OPERATIONAL DETAILS
# -------------------------------------------------

st.header("ATM Operational Details")

col1, col2, col3 = st.columns(3)

with col1:

    atm_age = st.number_input(
        "ATM Age (Years)",
        min_value=0.0,
        value=5.0
    )

    daily_transactions = st.number_input(
        "Daily Transactions",
        min_value=0,
        value=100
    )

    failed_transactions = st.number_input(
        "Failed Transactions",
        min_value=0,
        value=0
    )

    cash_remaining = st.number_input(
        "Cash Remaining (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    cash_loaded = st.number_input(
        "Cash Loaded Today",
        min_value=0,
        value=100000
    )


    cash_jams = st.number_input(
        "Cash Jams",
        min_value=0,
        value=0
    )


with col2:

    card_reader_errors = st.number_input(
        "Card Reader Errors",
        min_value=0,
        value=0
    )

    cash_dispenser_errors = st.number_input(
        "Cash Dispenser Errors",
        min_value=0,
        value=0
    )

    cpu_temp = st.number_input(
        "CPU Temperature (°C)",
        min_value=0.0,
        value=50.0
    )

    cpu_usage = st.number_input(
        "CPU Usage (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    memory_usage = st.number_input(
        "Memory Usage (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


    paper_remaining = st.number_input(
        "Paper Remaining (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


with col3:

    network_latency = st.number_input(
        "Network Latency (ms)",
        min_value=0.0,
        value=50.0
    )

    network_drops = st.number_input(
        "Network Drops",
        min_value=0,
        value=0
    )

    ups_battery = st.number_input(
        "UPS Battery (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    power_outages = st.number_input(
        "Power Outages (7 Days)",
        min_value=0,
        value=0
    )

    days_since_maintenance = st.number_input(
        "Days Since Maintenance",
        min_value=0,
        value=30
    )

   
    printer_errors = st.number_input(
        "Printer Errors",
        min_value=0,
        value=0
    )


# -------------------------------------------------
# FAILURE AND COMPLAINT HISTORY
# -------------------------------------------------

st.header("Failure & Complaint History")

col1, col2 = st.columns(2)

with col1:

    previous_failures = st.number_input(
        "Previous Failures (90 Days)",
        min_value=0,
        value=0
    )

with col2:

    customer_complaints = st.number_input(
        "Customer Complaints (7 Days)",
        min_value=0,
        value=0
    )


if st.button("🔮 Predict ATM Failure", use_container_width=True):

    # Encode categorical variables
    district_encoded = encoders["District"].transform([district])[0]
    location_type_encoded = encoders["Location_Type"].transform([location_type])[0]
    manufacturer_encoded = encoders["Manufacturer"].transform([manufacturer])[0]

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "District": district_encoded,
        "Location_Type": location_type_encoded,
        "Manufacturer": manufacturer_encoded,

        "ATM_Age_Yrs": atm_age,
        "Daily_Transactions": daily_transactions,
        "Failed_Transactions": failed_transactions,
        "Cash_Remaining_Pct": cash_remaining,
        "Cash_Loaded_Today": cash_loaded,

        "CardReader_Errors": card_reader_errors,
        "CashDispenser_Errors": cash_dispenser_errors,
        "Cash_Jams": cash_jams,
        "Printer_Errors": printer_errors,
        "Paper_Remaining_Pct": paper_remaining,

        "CPU_Temp_C": cpu_temp,
        "CPU_Usage_Pct": cpu_usage,
        "Memory_Usage_Pct": memory_usage,

        "Network_Latency_ms": network_latency,
        "Network_Drops": network_drops,

        "UPS_Battery_Pct": ups_battery,
        "Power_Outages_7D": power_outages,

        "Days_Since_Maintenance": days_since_maintenance,
        "Previous_Failures_90D": previous_failures,
        "Customer_Complaints_7D": customer_complaints
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ ATM Failure Predicted")
        st.write("The ATM is likely to fail within the next 48 hours.")
    else:
        st.success("✅ No ATM Failure Predicted")
        st.write("The ATM is unlikely to fail within the next 48 hours.")
