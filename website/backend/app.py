from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import pandas as pd

# ============================================================
# XAI-NIDS FLASK BACKEND
# ============================================================

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "xgboost.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "preprocessor.pkl"
)

TRAINING_DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "UNSW_NB15_training-set.csv"
)

# ------------------------------------------------------------
# RAW FEATURES EXPECTED BY PREPROCESSOR
# ------------------------------------------------------------

FEATURES = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports"
]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state"
]

NUMERICAL_FEATURES = [
    feature for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]

# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

print("=" * 70)
print("             XAI-NIDS FLASK BACKEND")
print("=" * 70)

print("\nLoading XGBoost model...")

try:
    model = joblib.load(MODEL_PATH)
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"ERROR loading XGBoost model: {e}")
    model = None

print("\nLoading preprocessing pipeline...")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Preprocessor loaded successfully.")
except Exception as e:
    print(f"ERROR loading preprocessor: {e}")
    preprocessor = None


# ------------------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------------------

def prepare_input(data):
    """
    Prepare raw network traffic data for the saved
    preprocessing pipeline.
    """

    missing_features = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            "Missing features: " + ", ".join(missing_features)
        )

    # Keep only the exact feature order expected by preprocessing
    input_data = {
        feature: data[feature]
        for feature in FEATURES
    }

    df = pd.DataFrame([input_data])

    # Ensure numerical values are numeric
    for feature in NUMERICAL_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # Convert categorical fields to strings
    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].astype(str)

    return df


# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "project": "Explainable Artificial Intelligence for Network Intrusion Detection",
        "model": "XGBoost",
        "features": 194,
        "raw_features": 42
    })


# ------------------------------------------------------------
# MODEL INFO
# ------------------------------------------------------------

@app.route("/api/model-info", methods=["GET"])
def model_info():

    return jsonify({
        "model": "XGBoost",
        "accuracy": 87.32,
        "precision": 81.92,
        "recall": 98.76,
        "f1_score": 89.56,
        "roc_auc": 98.46,
        "raw_features": 42,
        "processed_features": 194,
        "dataset": "UNSW-NB15"
    })


# ------------------------------------------------------------
# SCHEMA
# ------------------------------------------------------------

@app.route("/api/schema", methods=["GET"])
def schema():

    return jsonify({
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categories": {
            "proto": [
                '3pc', 'a/n', 'aes-sp3-d', 'any', 'argus',
                'aris', 'arp', 'ax.25', 'bbn-rcc', 'bna',
                'br-sat-mon', 'cbt', 'cftp', 'chaos',
                'compaq-peer', 'cphb', 'cpnx', 'crtp',
                'crudp', 'dcn', 'ddp', 'ddx', 'dgp', 'egp',
                'eigrp', 'emcon', 'encap', 'etherip', 'fc',
                'fire', 'ggp', 'gmtp', 'gre', 'hmp', 'i-nlsp',
                'iatp', 'ib', 'icmp', 'idpr', 'idpr-cmtp',
                'idrp', 'ifmp', 'igmp', 'igp', 'il', 'ip',
                'ipcomp', 'ipcv', 'ipip', 'iplt', 'ipnip',
                'ippc', 'ipv6', 'ipv6-frag', 'ipv6-no',
                'ipv6-opts', 'ipv6-route', 'ipx-n-ip', 'irtp',
                'isis', 'iso-ip', 'iso-tp4', 'kryptolan',
                'l2tp', 'larp', 'leaf-1', 'leaf-2', 'merit-inp',
                'mfe-nsp', 'mhrp', 'micp', 'mobile', 'mtp',
                'mux', 'narp', 'netblt', 'nsfnet-igp', 'nvp',
                'ospf', 'pgm', 'pim', 'pipe', 'pnni', 'pri-enc',
                'prm', 'ptp', 'pup', 'pvp', 'qnx', 'rdp',
                'rsvp', 'rtp', 'rvd', 'sat-expak', 'sat-mon',
                'sccopmce', 'scps', 'sctp', 'sdrp', 'secure-vmtp',
                'sep', 'skip', 'sm', 'smp', 'snp', 'sprite-rpc',
                'sps', 'srp', 'st2', 'stp', 'sun-nd', 'swipe',
                'tcf', 'tcp', 'tlsp', 'tp++', 'trunk-1',
                'trunk-2', 'ttp', 'udp', 'unas', 'uti', 'vines',
                'visa', 'vmtp', 'vrrp', 'wb-expak', 'wb-mon',
                'wsn', 'xnet', 'xns-idp', 'xtp', 'zero'
            ],
            "service": [
                '-', 'dhcp', 'dns', 'ftp', 'ftp-data',
                'http', 'irc', 'pop3', 'radius', 'smtp',
                'snmp', 'ssh', 'ssl'
            ],
            "state": [
                'CON', 'ECO', 'FIN', 'INT', 'PAR',
                'REQ', 'RST', 'URN', 'no'
            ]
        }
    })


# ------------------------------------------------------------
# SAMPLE NETWORK TRAFFIC
# ------------------------------------------------------------

@app.route("/api/sample", methods=["GET"])
def sample():

    try:

        df = pd.read_csv(
            TRAINING_DATA_PATH,
            nrows=1
        )

        sample = {}

        for feature in FEATURES:
            value = df.iloc[0][feature]

            # Convert numpy values into normal Python values
            if pd.isna(value):
                sample[feature] = 0
            else:
                if feature in CATEGORICAL_FEATURES:
                    sample[feature] = str(value)
                else:
                    sample[feature] = float(value)

        return jsonify({
            "success": True,
            "data": sample
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():

    if model is None or preprocessor is None:

        return jsonify({
            "success": False,
            "error": "Model or preprocessor is not loaded."
        }), 500

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400

        # Prepare exact raw feature DataFrame
        input_df = prepare_input(data)

        # Apply SAME preprocessing pipeline used during research
        processed_input = preprocessor.transform(input_df)

        # XGBoost prediction
        prediction = int(
            model.predict(processed_input)[0]
        )

        probabilities = model.predict_proba(
            processed_input
        )[0]

        normal_probability = float(probabilities[0])
        attack_probability = float(probabilities[1])

        result = (
            "Attack"
            if prediction == 1
            else "Normal"
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "result": result,
            "attack_probability": round(
                attack_probability,
                6
            ),
            "normal_probability": round(
                normal_probability,
                6
            ),
            "model": "XGBoost",
            "processed_features": int(
                processed_input.shape[1]
            )
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ------------------------------------------------------------
# SERVER
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("              SERVER STARTING")
    print("=" * 70)

    print("\nBackend URL:")
    print("http://127.0.0.1:5000")

    print("\nAvailable endpoints:")
    print("GET  /")
    print("GET  /api/model-info")
    print("GET  /api/schema")
    print("GET  /api/sample")
    print("POST /api/predict")

    print("\n" + "=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )