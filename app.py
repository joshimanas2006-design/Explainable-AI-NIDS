
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
import numpy as np
import joblib
import os
import shap


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "UNSW_NB15_testing-set.csv"
)


# ============================================================
# FRONTEND PATH
# ============================================================

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "website",
    "frontend"
)


# ============================================================
# SHAP RESULTS PATH
# ============================================================

XAI_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "xai",
    "results"
)


FEATURE_IMPORTANCE_PATH = os.path.join(
    XAI_RESULTS_DIR,
    "feature_importance.csv"
)


INDIVIDUAL_EXPLANATION_PATH = os.path.join(
    XAI_RESULTS_DIR,
    "individual_explanation.txt"
)


SHAP_BAR_PATH = os.path.join(
    XAI_RESULTS_DIR,
    "shap_feature_importance_bar.png"
)


SHAP_SUMMARY_PATH = os.path.join(
    XAI_RESULTS_DIR,
    "shap_summary.png"
)


SHAP_INDIVIDUAL_PATH = os.path.join(
    XAI_RESULTS_DIR,
    "shap_individual_prediction.png"
)


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

print("=" * 70)
print("             XAI-NIDS FLASK API")
print("=" * 70)


print("\nLoading XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print("✓ XGBoost model loaded")


print("\nLoading preprocessing pipeline...")

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

print("✓ Preprocessor loaded")


# ============================================================
# RAW FEATURES EXPECTED BY PREPROCESSOR
# ============================================================

RAW_FEATURES = list(
    preprocessor.feature_names_in_
)


PROCESSED_FEATURES = list(
    preprocessor.get_feature_names_out()
)


print(
    f"\nRaw input features expected: "
    f"{len(RAW_FEATURES)}"
)


print(
    "Processed features produced: "
    f"{len(PROCESSED_FEATURES)}"
)


# ============================================================
# SHAP TREE EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

try:

    shap_explainer = shap.TreeExplainer(
        model
    )

    SHAP_AVAILABLE = True

    print("✓ SHAP TreeExplainer created")

except Exception as error:

    shap_explainer = None

    SHAP_AVAILABLE = False

    print(
        "\n⚠ SHAP TreeExplainer could not be created:"
    )

    print(error)


# ============================================================
# SAMPLE DATASET
# ============================================================

sample_df = None


if os.path.exists(TEST_DATA_PATH):

    try:

        sample_df = pd.read_csv(
            TEST_DATA_PATH
        )

        print(
            f"\n✓ Test dataset loaded: "
            f"{sample_df.shape}"
        )

    except Exception as error:

        print(
            "\n⚠ Could not load test dataset:"
        )

        print(error)

else:

    print(
        "\n⚠ Test dataset file not found:"
    )

    print(TEST_DATA_PATH)


# ============================================================
# HELPER FUNCTION
# ============================================================

def clean_value(value):
    """
    Convert NumPy/Pandas values into
    JSON-compatible Python values.
    """

    if value is None:

        return None


    try:

        if pd.isna(value):

            return None

    except Exception:

        pass


    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
            np.int16
        )
    ):

        return int(value)


    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
            np.float16
        )
    ):

        return float(value)


    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)


    return value


# ============================================================
# HELPER: CONVERT PROCESSED DATA TO DENSE
# ============================================================

def make_dense(data):

    if hasattr(
        data,
        "toarray"
    ):

        return data.toarray()

    return np.asarray(data)


# ============================================================
# HELPER: CALCULATE INDIVIDUAL SHAP
# ============================================================

def calculate_shap_explanation(processed_data):

    if not SHAP_AVAILABLE:

        return {

            "available":
                False,

            "error":
                "SHAP TreeExplainer is not available."

        }


    try:

        dense_data = make_dense(
            processed_data
        )


        # ----------------------------------------------------
        # Calculate SHAP values
        # ----------------------------------------------------

        shap_result = shap_explainer.shap_values(
            dense_data
        )


        # ----------------------------------------------------
        # Handle different SHAP output formats
        # ----------------------------------------------------

        if isinstance(
            shap_result,
            list
        ):

            # Binary classification
            # Normally class 1 = Attack

            if len(shap_result) > 1:

                shap_values = np.asarray(
                    shap_result[1]
                )[0]

            else:

                shap_values = np.asarray(
                    shap_result[0]
                )[0]

        else:

            shap_array = np.asarray(
                shap_result
            )


            # Some SHAP/XGBoost versions
            # may return an extra output dimension

            if shap_array.ndim == 3:

                # Use class 1 for binary classification

                if shap_array.shape[-1] > 1:

                    shap_values = shap_array[
                        0,
                        :,
                        1
                    ]

                else:

                    shap_values = shap_array[
                        0,
                        :,
                        0
                    ]

            elif shap_array.ndim == 2:

                shap_values = shap_array[0]

            else:

                shap_values = shap_array


        shap_values = np.asarray(
            shap_values,
            dtype=float
        ).flatten()


        # ----------------------------------------------------
        # Make sure feature count matches
        # ----------------------------------------------------

        feature_count = min(
            len(PROCESSED_FEATURES),
            len(shap_values),
            dense_data.shape[1]
        )


        shap_values = shap_values[
            :feature_count
        ]


        feature_names = PROCESSED_FEATURES[
            :feature_count
        ]


        feature_values = dense_data[
            0,
            :feature_count
        ]


        # ----------------------------------------------------
        # Create explanation records
        # ----------------------------------------------------

        explanations = []


        for index in range(
            feature_count
        ):

            shap_value = float(
                shap_values[index]
            )


            explanations.append({

                "feature":
                    feature_names[index],

                "shap_value":
                    shap_value,

                "feature_value":
                    clean_value(
                        feature_values[index]
                    ),

                "absolute_shap":
                    abs(shap_value),

                "direction":
                    "Attack"
                    if shap_value > 0
                    else "Normal"

            })


        # ----------------------------------------------------
        # Sort by absolute SHAP value
        # ----------------------------------------------------

        explanations.sort(
            key=lambda item:
                item["absolute_shap"],
            reverse=True
        )


        # ----------------------------------------------------
        # Top 15 features
        # ----------------------------------------------------

        top_features = explanations[
            :15
        ]


        # ----------------------------------------------------
        # Positive / negative contributors
        # ----------------------------------------------------

        attack_features = [

            item

            for item in explanations

            if item["shap_value"] > 0

        ][:10]


        normal_features = [

            item

            for item in explanations

            if item["shap_value"] < 0

        ][:10]


        # ----------------------------------------------------
        # Expected value
        # ----------------------------------------------------

        try:

            expected_value = (
                shap_explainer.expected_value
            )


            if isinstance(
                expected_value,
                (list, np.ndarray)
            ):

                expected_array = np.asarray(
                    expected_value
                ).flatten()


                if len(expected_array) > 1:

                    expected_value = float(
                        expected_array[1]
                    )

                else:

                    expected_value = float(
                        expected_array[0]
                    )

            else:

                expected_value = float(
                    expected_value
                )

        except Exception:

            expected_value = None


        return {

            "available":
                True,

            "feature_count":
                feature_count,

            "top_features":
                top_features,

            "attack_contributors":
                attack_features,

            "normal_contributors":
                normal_features,

            "expected_value":
                expected_value

        }


    except Exception as error:

        print(
            "\nSHAP calculation error:"
        )

        print(error)


        return {

            "available":
                False,

            "error":
                str(error)

        }


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# CSS FILES
# ============================================================

@app.route(
    "/css/<path:filename>"
)
def css_files(filename):

    return send_from_directory(
        os.path.join(
            FRONTEND_DIR,
            "css"
        ),
        filename
    )


# ============================================================
# JAVASCRIPT FILES
# ============================================================

@app.route(
    "/js/<path:filename>"
)
def js_files(filename):

    return send_from_directory(
        os.path.join(
            FRONTEND_DIR,
            "js"
        ),
        filename
    )


# ============================================================
# SHAP RESULT IMAGES
# ============================================================

@app.route(
    "/xai/results/<path:filename>"
)
def xai_result_files(filename):

    return send_from_directory(
        XAI_RESULTS_DIR,
        filename
    )


# ============================================================
# API SCHEMA
# ============================================================

@app.route(
    "/api/schema",
    methods=["GET"]
)
def schema():

    schema_data = {}


    # --------------------------------------------------------
    # Find categorical features
    # --------------------------------------------------------

    categorical_features = set()


    try:

        categorical_features = set(
            preprocessor
            .transformers_[1][2]
        )

    except Exception:

        categorical_features = set()


    # --------------------------------------------------------
    # Build schema
    # --------------------------------------------------------

    for feature in RAW_FEATURES:

        schema_data[feature] = {

            "name":
                feature,

            "type":
                "categorical"
                if feature
                in categorical_features
                else "numerical"

        }


    return jsonify({

        "success":
            True,

        "feature_count":
            len(RAW_FEATURES),

        "features":
            RAW_FEATURES,

        "schema":
            schema_data

    })


# ============================================================
# API SAMPLE
# ============================================================

@app.route(
    "/api/sample",
    methods=["GET"]
)
def sample():

    if sample_df is None:

        return jsonify({

            "success":
                False,

            "error":
                "Test dataset is not available."

        }), 500


    try:

        row = sample_df.iloc[0]


        sample = {}


        for feature in RAW_FEATURES:

            if feature in row.index:

                sample[feature] = clean_value(
                    row[feature]
                )


        return jsonify({

            "success":
                True,

            "sample":
                sample

        })


    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# API GLOBAL SHAP IMPORTANCE
# ============================================================

@app.route(
    "/api/xai/global",
    methods=["GET"]
)
def global_shap():

    try:

        if not os.path.exists(
            FEATURE_IMPORTANCE_PATH
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "SHAP feature importance file not found."

            }), 404


        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )


        # ----------------------------------------------------
        # Sort by importance
        # ----------------------------------------------------

        importance_df = importance_df.sort_values(
            "mean_abs_shap",
            ascending=False
        )


        # ----------------------------------------------------
        # Top 20
        # ----------------------------------------------------

        top_df = importance_df.head(
            20
        )


        features = []


        for _, row in top_df.iterrows():

            feature_name = str(
                row["feature"]
            )


            # Remove preprocessing prefixes
            display_name = (
                feature_name
                .replace(
                    "numerical__",
                    ""
                )
                .replace(
                    "categorical__",
                    ""
                )
            )


            features.append({

                "feature":
                    feature_name,

                "display_name":
                    display_name,

                "mean_abs_shap":
                    float(
                        row["mean_abs_shap"]
                    )

            })


        return jsonify({

            "success":
                True,

            "total_features":
                len(importance_df),

            "top_features":
                features,

            "chart":

                {

                    "bar":
                        "/xai/results/shap_feature_importance_bar.png",

                    "summary":
                        "/xai/results/shap_summary.png"

                }

        })


    except Exception as error:

        print(
            "\nGlobal SHAP error:"
        )

        print(error)


        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# API EXISTING SHAP INDIVIDUAL EXAMPLE
# ============================================================

@app.route(
    "/api/xai/example",
    methods=["GET"]
)
def shap_example():

    try:

        if not os.path.exists(
            INDIVIDUAL_EXPLANATION_PATH
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Individual SHAP explanation file not found."

            }), 404


        with open(
            INDIVIDUAL_EXPLANATION_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            explanation_text = file.read()


        return jsonify({

            "success":
                True,

            "explanation":
                explanation_text,

            "prediction":
                1,

            "classification":
                "Attack",

            "attack_probability":
                0.9997,

            "chart":
                "/xai/results/shap_individual_prediction.png"

        })


    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# API PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # GET JSON
        # ----------------------------------------------------

        data = request.get_json()


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No JSON data received."

            }), 400


        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in RAW_FEATURES

            if feature not in data

        ]


        if missing_features:

            return jsonify({

                "success":
                    False,

                "error":
                    "Missing required features.",

                "missing_features":
                    missing_features

            }), 400


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = {

            feature:
                data[feature]

            for feature in RAW_FEATURES

        }


        input_df = pd.DataFrame(
            [input_data]
        )


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_data = (
            preprocessor.transform(
                input_df
            )
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]


        probabilities = (
            model.predict_proba(
                processed_data
            )[0]
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        normal_probability = float(
            probabilities[0]
        )


        attack_probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if int(prediction) == 1:

            classification = "Attack"

            description = (
                "The XGBoost model classified "
                "this network traffic as "
                "potentially malicious."
            )

        else:

            classification = "Normal"

            description = (
                "The XGBoost model classified "
                "this network traffic as "
                "normal network activity."
            )


        # ----------------------------------------------------
        # INDIVIDUAL SHAP EXPLANATION
        # ----------------------------------------------------

        shap_explanation = (
            calculate_shap_explanation(
                processed_data
            )
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "prediction":
                int(prediction),

            "classification":
                classification,

            "description":
                description,

            "probabilities": {

                "normal":
                    normal_probability,

                "attack":
                    attack_probability

            },

            "model":
                "XGBoost",

            "raw_features":
                len(RAW_FEATURES),

            "processed_features":
                processed_data.shape[1],

            "xai": {

                "available":
                    shap_explanation[
                        "available"
                    ],

                "top_features":
                    shap_explanation.get(
                        "top_features",
                        []
                    ),

                "attack_contributors":
                    shap_explanation.get(
                        "attack_contributors",
                        []
                    ),

                "normal_contributors":
                    shap_explanation.get(
                        "normal_contributors",
                        []
                    ),

                "expected_value":
                    shap_explanation.get(
                        "expected_value"
                    )

            }

        })


    except Exception as error:

        print(
            "\nPrediction error:"
        )

        print(error)


        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# API XAI STATUS
# ============================================================

@app.route(
    "/api/xai/status",
    methods=["GET"]
)
def xai_status():

    return jsonify({

        "success":
            True,

        "shap_available":
            SHAP_AVAILABLE,

        "global_results_available":
            os.path.exists(
                FEATURE_IMPORTANCE_PATH
            ),

        "individual_results_available":
            os.path.exists(
                INDIVIDUAL_EXPLANATION_PATH
            ),

        "charts": {

            "bar":
                os.path.exists(
                    SHAP_BAR_PATH
                ),

            "summary":
                os.path.exists(
                    SHAP_SUMMARY_PATH
                ),

            "individual":
                os.path.exists(
                    SHAP_INDIVIDUAL_PATH
                )

        }

    })


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "          XAI-NIDS API SERVER"
    )

    print("=" * 70)


    print(
        "\nAPI endpoints:"
    )


    print(
        "GET  /"
    )


    print(
        "GET  /api/schema"
    )


    print(
        "GET  /api/sample"
    )


    print(
        "POST /api/predict"
    )


    print(
        "GET  /api/xai/status"
    )


    print(
        "GET  /api/xai/global"
    )


    print(
        "GET  /api/xai/example"
    )


    print(
        "\nSHAP result images:"
    )


    print(
        "/xai/results/shap_feature_importance_bar.png"
    )


    print(
        "/xai/results/shap_summary.png"
    )


    print(
        "/xai/results/shap_individual_prediction.png"
    )


    print(
        "\nWebsite:"
    )

    print(
        "http://127.0.0.1:5000"
    )


    print(
        "\nPress CTRL+C to stop."
    )


    print("=" * 70)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )