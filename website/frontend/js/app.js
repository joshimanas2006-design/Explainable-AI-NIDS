const API_BASE_URL = "http://127.0.0.1:5000";

let latestPrediction = null;
let globalXAI = null;

document.addEventListener("DOMContentLoaded", function () {
    initializeApp();
});

async function initializeApp() {
    setupNavigation();
    setupButtons();

    await checkBackendStatus();
    await loadGlobalXAI();
}

/* =========================================================
   NAVIGATION
========================================================= */

function setupNavigation() {
    document.querySelectorAll(".nav-item").forEach(function (item) {
        item.addEventListener("click", function () {
            const page = item.getAttribute("data-page");

            if (page) {
                showPage(page);
            }
        });
    });

    document.querySelectorAll("[data-go]").forEach(function (button) {
        button.addEventListener("click", function () {
            const page = button.getAttribute("data-go");

            if (page) {
                showPage(page);
            }
        });
    });
}

function showPage(pageId) {
    document.querySelectorAll(".page").forEach(function (page) {
        page.classList.remove("active");
    });

    const targetPage = document.getElementById("page-" + pageId);

    if (targetPage) {
        targetPage.classList.add("active");
    }

    document.querySelectorAll(".nav-item").forEach(function (item) {
        item.classList.remove("active");

        if (item.getAttribute("data-page") === pageId) {
            item.classList.add("active");
        }
    });

    const pageTitle = document.getElementById("page-title");

    if (pageTitle) {
        const titles = {
            dashboard: "Dashboard",
            detection: "Intrusion Detection",
            explain: "Explain Prediction",
            comparison: "Model Comparison",
            shap: "SHAP Analysis",
            research: "Research"
        };

        pageTitle.textContent = titles[pageId] || "Dashboard";
    }

    if (pageId === "explain") {
        renderExplanationPage();
    }

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}

/* =========================================================
   BUTTONS
========================================================= */

function setupButtons() {
    const sampleButtons = [
        document.getElementById("load-sample-btn"),
        document.getElementById("load-sample-btn-bottom")
    ];

    sampleButtons.forEach(function (button) {
        if (button) {
            button.addEventListener("click", loadSampleTraffic);
        }
    });

    const clearButton = document.getElementById("clear-form-btn");

    if (clearButton) {
        clearButton.addEventListener("click", clearForm);
    }

    const trafficForm = document.getElementById("traffic-form");

    if (trafficForm) {
        trafficForm.addEventListener("submit", function (event) {
            event.preventDefault();
            analyzeTraffic();
        });
    }

    const explainResultButton =
        document.getElementById("explain-result-btn");

    if (explainResultButton) {
        explainResultButton.addEventListener("click", function () {
            showPage("explain");
        });
    }

    const fullExplanationButton =
        document.getElementById("view-full-explanation");

    if (fullExplanationButton) {
        fullExplanationButton.addEventListener("click", function () {
            showPage("explain");
        });
    }
}

/* =========================================================
   BACKEND STATUS
========================================================= */

async function checkBackendStatus() {
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.getElementById("backend-status-text");

    try {
        const response =
            await fetch(API_BASE_URL + "/api/xai/status");

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (statusDot) {
            statusDot.classList.remove("offline");
            statusDot.classList.add("online");
        }

        if (statusText) {
            statusText.textContent =
                data.shap_available
                    ? "Backend Online • SHAP Ready"
                    : "Backend Online";
        }

    } catch (error) {
        if (statusDot) {
            statusDot.classList.remove("online");
            statusDot.classList.add("offline");
        }

        if (statusText) {
            statusText.textContent = "Backend Offline";
        }

        console.error("Backend status error:", error);
    }
}

/* =========================================================
   SAMPLE TRAFFIC
========================================================= */

async function loadSampleTraffic() {
    hideError();

    try {
        const response =
            await fetch(API_BASE_URL + "/api/sample");

        if (!response.ok) {
            throw new Error("Could not load sample traffic.");
        }

        const data = await response.json();

        if (!data.sample) {
            throw new Error("Sample traffic data not found.");
        }

        fillFormWithSample(data.sample);

        showDetectionStatus(
            "Sample network traffic loaded successfully."
        );

    } catch (error) {
        console.error("Sample error:", error);

        showError(
            error.message ||
            "Unable to load sample traffic."
        );
    }
}

function fillFormWithSample(sample) {
    Object.keys(sample).forEach(function (key) {
        const input = document.getElementById(key);

        if (!input) {
            return;
        }

        const value = sample[key];

        if (value === null || value === undefined) {
            input.value = "";
        } else {
            input.value = value;
        }
    });

    console.log("Sample traffic filled into form.");
}

/* =========================================================
   FORM DATA
========================================================= */

function collectFormData() {
    const data = {};

    const inputs =
        document.querySelectorAll(
            "#traffic-form input, #traffic-form select"
        );

    inputs.forEach(function (input) {
        const key =
            input.name ||
            input.id;

        if (!key) {
            return;
        }

        let value = input.value;

        if (value === "") {
            value = 0;
        }

        data[key] = value;
    });

    console.log("Collected features:", data);

    return data;
}

/* =========================================================
   ANALYZE TRAFFIC
========================================================= */

async function analyzeTraffic() {
    hideError();
    showLoading(true);

    showDetectionStatus(
        "Analyzing network traffic and generating SHAP explanation..."
    );

    const data = collectFormData();

    try {
        const response =
            await fetch(
                API_BASE_URL + "/api/predict",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                }
            );

        const result = await response.json();

        console.log("Prediction response:", result);

        if (!response.ok || !result.success) {
            throw new Error(
                result.error ||
                "Prediction failed."
            );
        }

        latestPrediction = result;

        displayPrediction(result);
        renderCurrentXAI();
        renderExplanationPage();

        const explainCTA =
            document.getElementById("explain-cta");

        if (explainCTA) {
            explainCTA.classList.remove("hidden");
            explainCTA.classList.add("show");
        }

        showDetectionStatus(
            "Traffic analyzed successfully. SHAP explanation generated."
        );

    } catch (error) {
        console.error("Prediction error:", error);

        displayPredictionError(
            error.message ||
            "Unable to analyze traffic."
        );

    } finally {
        showLoading(false);
    }
}

/* =========================================================
   DISPLAY PREDICTION
========================================================= */

function displayPrediction(result) {
    const predictionResult =
        document.getElementById("prediction-result");

    if (!predictionResult) {
        return;
    }

    predictionResult.classList.remove("hidden");
    predictionResult.classList.add("show");

    const prediction =
        Number(result.prediction);

    const isAttack =
        prediction === 1 ||
        String(result.classification || "")
            .toLowerCase() === "attack";

    const predictionMain =
        predictionResult.querySelector(".prediction-main");

    const badge =
        document.getElementById("prediction-badge");

    const icon =
        document.getElementById("prediction-icon");

    const label =
        document.getElementById("prediction-label");

    const description =
        document.getElementById("prediction-description");

    const attackProbability =
        document.getElementById("attack-probability");

    const normalProbability =
        document.getElementById("normal-probability");

    const attackProgress =
        document.getElementById("attack-progress");

    const normalProgress =
        document.getElementById("normal-progress");

    if (predictionMain) {
        predictionMain.classList.remove(
            "attack",
            "normal"
        );

        predictionMain.classList.add(
            isAttack ? "attack" : "normal"
        );
    }

    if (badge) {
        badge.classList.remove(
            "attack",
            "normal"
        );

        badge.classList.add(
            isAttack ? "attack" : "normal"
        );
    }

    if (label) {
        label.classList.remove(
            "attack",
            "normal"
        );

        label.classList.add(
            isAttack ? "attack" : "normal"
        );
    }

    if (icon) {
        icon.textContent =
            isAttack ? "!" : "✓";
    }

    if (label) {
        label.textContent =
            result.classification ||
            (isAttack ? "Attack" : "Normal");
    }

    if (description) {
        description.textContent =
            result.description ||
            "The XGBoost model analyzed the submitted network traffic.";
    }

    const probabilities =
        result.probabilities || {};

    const attack =
        Number(probabilities.attack || 0);

    const normal =
        Number(probabilities.normal || 0);

    if (attackProbability) {
        attackProbability.textContent =
            formatPercent(attack);
    }

    if (normalProbability) {
        normalProbability.textContent =
            formatPercent(normal);
    }

    if (attackProgress) {
        attackProgress.style.width =
            Math.max(0, Math.min(100, attack * 100)) + "%";
    }

    if (normalProgress) {
        normalProgress.style.width =
            Math.max(0, Math.min(100, normal * 100)) + "%";
    }
}

/* =========================================================
   PREDICTION ERROR
========================================================= */

function displayPredictionError(message) {
    const predictionResult =
        document.getElementById("prediction-result");

    if (predictionResult) {
        predictionResult.classList.remove("show");
        predictionResult.classList.add("hidden");
    }

    showError(message);
}

/* =========================================================
   CURRENT SHAP
========================================================= */

function renderCurrentXAI() {
    if (
        !latestPrediction ||
        !latestPrediction.xai ||
        !latestPrediction.xai.available
    ) {
        return;
    }

    const xai =
        latestPrediction.xai;

    const currentSHAP =
        document.getElementById("current-shap");

    if (currentSHAP) {
        currentSHAP.classList.remove("hidden");
        currentSHAP.classList.add("show");
    }

    renderTopFeatures(
        xai.top_features || []
    );

    renderDirectionList(
        xai.top_features || []
    );
}

/* =========================================================
   TOP SHAP FEATURES
========================================================= */

function renderTopFeatures(features) {
    const container =
        document.getElementById("top-features-list");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!features.length) {
        container.innerHTML =
            '<div class="chart-placeholder">No SHAP features available.</div>';

        return;
    }

    const maxValue =
        Math.max.apply(
            null,
            features.map(function (item) {
                return Math.abs(
                    Number(item.shap_value || 0)
                );
            })
        );

    features
        .slice(0, 10)
        .forEach(function (item) {
            const row =
                document.createElement("div");

            row.className =
                "feature-row";

            const name =
                document.createElement("div");

            name.className =
                "feature-name";

            name.textContent =
                cleanFeatureName(item.feature);

            const bar =
                document.createElement("div");

            bar.className =
                "feature-bar";

            const fill =
                document.createElement("div");

            fill.className =
                "feature-bar-fill";

            const shapValue =
                Number(item.shap_value || 0);

            if (shapValue >= 0) {
                fill.classList.add("positive");
            } else {
                fill.classList.add("negative");
            }

            const width =
                maxValue > 0
                    ? Math.abs(shapValue) / maxValue * 100
                    : 0;

            fill.style.width =
                Math.max(width, 3) + "%";

            bar.appendChild(fill);

            const value =
                document.createElement("div");

            value.className =
                "feature-value";

            value.textContent =
                (shapValue >= 0 ? "+" : "") +
                formatNumber(shapValue);

            row.appendChild(name);
            row.appendChild(bar);
            row.appendChild(value);

            container.appendChild(row);
        });
}

/* =========================================================
   SHAP DIRECTION
========================================================= */

function renderDirectionList(features) {
    const container =
        document.getElementById("direction-list");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!features.length) {
        return;
    }

    features
        .slice(0, 10)
        .forEach(function (item) {
            const row =
                document.createElement("div");

            row.className =
                "feature-row";

            const name =
                document.createElement("div");

            name.className =
                "feature-name";

            name.textContent =
                cleanFeatureName(item.feature);

            const direction =
                document.createElement("div");

            direction.className =
                "feature-value";

            const shapValue =
                Number(item.shap_value || 0);

            direction.textContent =
                shapValue >= 0
                    ? "→ Attack"
                    : "→ Normal";

            direction.classList.add(
                shapValue >= 0
                    ? "shap-positive"
                    : "shap-negative"
            );

            row.appendChild(name);
            row.appendChild(direction);

            container.appendChild(row);
        });
}

/* =========================================================
   EXPLANATION PAGE
========================================================= */

function renderExplanationPage() {
    const empty =
        document.getElementById("explain-empty");

    const content =
        document.getElementById("explain-content");

    if (!latestPrediction) {
        if (empty) {
            empty.classList.remove("hidden");
            empty.classList.add("show");
        }

        if (content) {
            content.classList.remove("show");
            content.classList.add("hidden");
        }

        return;
    }

    if (empty) {
        empty.classList.remove("show");
        empty.classList.add("hidden");
    }

    if (content) {
        content.classList.remove("hidden");
        content.classList.add("show");
    }

    const classification =
        document.getElementById("explain-classification");

    const probability =
        document.getElementById("explain-probability");

    if (classification) {
        classification.textContent =
            latestPrediction.classification ||
            "Unknown";
    }

    if (probability) {
        const probabilities =
            latestPrediction.probabilities || {};

        const prediction =
            Number(latestPrediction.prediction);

        const selectedProbability =
            prediction === 1
                ? Number(probabilities.attack || 0)
                : Number(probabilities.normal || 0);

        probability.textContent =
            formatPercent(selectedProbability);
    }

    const xai =
        latestPrediction.xai || {};

    renderExplanationFeatures(
        xai.top_features || []
    );

    renderCurrentGlobalFeatures();
}

function renderExplanationFeatures(features) {
    const container =
        document.getElementById("explain-feature-list");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!features.length) {
        container.innerHTML =
            '<div class="chart-placeholder">No SHAP explanation available.</div>';

        return;
    }

    features
        .slice(0, 15)
        .forEach(function (item) {
            const row =
                document.createElement("div");

            row.className =
                "feature-row";

            const name =
                document.createElement("div");

            name.className =
                "feature-name";

            name.textContent =
                cleanFeatureName(item.feature);

            const value =
                document.createElement("div");

            value.className =
                "feature-value";

            const shap =
                Number(item.shap_value || 0);

            value.textContent =
                (shap >= 0 ? "+" : "") +
                formatNumber(shap);

            value.classList.add(
                shap >= 0
                    ? "shap-positive"
                    : "shap-negative"
            );

            row.appendChild(name);
            row.appendChild(value);

            container.appendChild(row);
        });
}

/* =========================================================
   GLOBAL SHAP
========================================================= */

async function loadGlobalXAI() {
    try {
        const response =
            await fetch(
                API_BASE_URL + "/api/xai/global"
            );

        if (!response.ok) {
            throw new Error(
                "Global SHAP data unavailable."
            );
        }

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                "Global SHAP request failed."
            );
        }

        globalXAI = data;

        renderGlobalFeatures(
            data.top_features || []
        );

        renderGlobalCharts(data);

    } catch (error) {
        console.error(
            "Global XAI loading failed:",
            error
        );
    }
}

/* =========================================================
   GLOBAL FEATURES
========================================================= */

function renderGlobalFeatures(features) {
    const container =
        document.getElementById("global-feature-list");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!features.length) {
        container.innerHTML =
            '<div class="chart-placeholder">Global SHAP data unavailable.</div>';

        return;
    }

    const maxValue =
        Math.max.apply(
            null,
            features.map(function (item) {
                return Number(
                    item.mean_abs_shap || 0
                );
            })
        );

    features
        .slice(0, 20)
        .forEach(function (item) {
            const row =
                document.createElement("div");

            row.className =
                "global-feature";

            const name =
                document.createElement("div");

            name.className =
                "global-feature-name";

            name.textContent =
                cleanFeatureName(item.feature);

            const bar =
                document.createElement("div");

            bar.className =
                "global-feature-bar";

            const span =
                document.createElement("span");

            const value =
                Number(item.mean_abs_shap || 0);

            const width =
                maxValue > 0
                    ? value / maxValue * 100
                    : 0;

            span.style.width =
                width + "%";

            bar.appendChild(span);

            const valueElement =
                document.createElement("div");

            valueElement.className =
                "global-feature-value";

            valueElement.textContent =
                formatNumber(value);

            row.appendChild(name);
            row.appendChild(bar);
            row.appendChild(valueElement);

            container.appendChild(row);
        });
}

/* =========================================================
   GLOBAL CHARTS
========================================================= */

function renderGlobalCharts(data) {
    if (!data || !data.charts) {
        return;
    }

    const chartImages = document.querySelectorAll(
        'img[src*="/xai/results/"]'
    );

    chartImages.forEach(function (image) {
        const source =
            image.getAttribute("src");

        if (!source) {
            return;
        }

        if (source.indexOf("/xai/results/") === 0) {
            image.src =
                API_BASE_URL +
                source +
                "?t=" +
                Date.now();
        }
    });
}

/* =========================================================
   CURRENT GLOBAL FEATURES
========================================================= */

function renderCurrentGlobalFeatures() {
    const container =
        document.getElementById(
            "current-global-features"
        );

    if (!container) {
        return;
    }

    if (
        !globalXAI ||
        !globalXAI.top_features
    ) {
        container.innerHTML =
            '<div class="chart-placeholder">Global SHAP data unavailable.</div>';

        return;
    }

    container.innerHTML = "";

    globalXAI.top_features
        .slice(0, 10)
        .forEach(function (item) {
            const row =
                document.createElement("div");

            row.className =
                "feature-row";

            const name =
                document.createElement("div");

            name.className =
                "feature-name";

            name.textContent =
                cleanFeatureName(item.feature);

            const value =
                document.createElement("div");

            value.className =
                "feature-value";

            value.textContent =
                formatNumber(
                    Number(
                        item.mean_abs_shap || 0
                    )
                );

            row.appendChild(name);
            row.appendChild(value);

            container.appendChild(row);
        });
}

/* =========================================================
   CLEAR FORM
========================================================= */

function clearForm() {
    const form =
        document.getElementById(
            "traffic-form"
        );

    if (form) {
        form.reset();
    }

    const predictionResult =
        document.getElementById(
            "prediction-result"
        );

    if (predictionResult) {
        predictionResult.classList.remove("show");
        predictionResult.classList.add("hidden");
    }

    const currentSHAP =
        document.getElementById(
            "current-shap"
        );

    if (currentSHAP) {
        currentSHAP.classList.remove("show");
        currentSHAP.classList.add("hidden");
    }

    const explainCTA =
        document.getElementById(
            "explain-cta"
        );

    if (explainCTA) {
        explainCTA.classList.remove("show");
        explainCTA.classList.add("hidden");
    }

    latestPrediction = null;

    hideError();

    renderExplanationPage();

    showDetectionStatus(
        "Form cleared."
    );
}

/* =========================================================
   STATUS MESSAGE
========================================================= */

function showDetectionStatus(message) {
    const status =
        document.getElementById(
            "detection-status"
        );

    const statusText =
        document.getElementById(
            "detection-status-text"
        );

    if (!status) {
        return;
    }

    if (statusText) {
        statusText.textContent =
            message;
    }

    status.classList.remove("hidden");
    status.classList.add("show");

    setTimeout(function () {
        status.classList.remove("show");
        status.classList.add("hidden");
    }, 4000);
}

/* =========================================================
   LOADING
========================================================= */

function showLoading(show) {
    const button =
        document.querySelector(
            ".analyze-btn"
        );

    if (!button) {
        return;
    }

    if (show) {
        button.disabled = true;
        button.textContent =
            "Analyzing...";
    } else {
        button.disabled = false;
        button.textContent =
            "Analyze Traffic →";
    }
}

/* =========================================================
   ERROR
========================================================= */

function showError(message) {
    const errorBox =
        document.getElementById(
            "prediction-error"
        );

    if (!errorBox) {
        return;
    }

    errorBox.textContent =
        message;

    errorBox.classList.remove("hidden");
    errorBox.classList.add("show");
}

function hideError() {
    const errorBox =
        document.getElementById(
            "prediction-error"
        );

    if (!errorBox) {
        return;
    }

    errorBox.classList.remove("show");
    errorBox.classList.add("hidden");
}

/* =========================================================
   HELPERS
========================================================= */

function cleanFeatureName(name) {
    if (!name) {
        return "";
    }

    return String(name)
        .replace(/^numerical__/, "")
        .replace(/^categorical__/, "")
        .replace(/_/g, " ")
        .replace(/\b\w/g, function (letter) {
            return letter.toUpperCase();
        });
}

function formatNumber(value) {
    const number =
        Number(value);

    if (!Number.isFinite(number)) {
        return "0.0000";
    }

    return number.toFixed(4);
}

function formatPercent(value) {
    const number =
        Number(value);

    if (!Number.isFinite(number)) {
        return "0.00%";
    }

    return (
        number * 100
    ).toFixed(2) + "%";
}