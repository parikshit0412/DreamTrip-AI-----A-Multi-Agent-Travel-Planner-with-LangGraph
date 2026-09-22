/**
 * ==============================================================================
 * DreamTrip AI — Interactive Frontend Orchestration Engine
 * ==============================================================================
 * This script powers the client-side single page application:
 * 1. Manages conversation session threads with browser localStorage persistence.
 * 2. Drives the live LangGraph multi-agent pipeline traversal visualization.
 * 3. Handles asynchronous API communication with latency and token tracking.
 * 4. Renders dynamic Markdown into executive luxury travel briefs.
 * 5. Controls tab switching across Master Plan, Flight Intel, Hotels, and Telemetry.
 * 6. Generates high-contrast executive PDF reports via html2pdf & html2canvas.
 * 7. Interfaces with Redis cache telemetry and live cache clearing endpoints.
 *
 * External Client Libraries (Loaded via CDN in index.html):
 * - marked.js: Fast markdown parser converting LLM output into semantic HTML.
 * - html2pdf.js / html2canvas / jsPDF: Client-side PDF generation engine.
 * ==============================================================================
 */

// Global session & document state
let currentThreadId = localStorage.getItem("dreamtrip_thread_id") || null; // Persistent thread UUID for PostgreSQL checkpoints
let currentAnswerMarkdown = "";  // Cached markdown string used for copy, PDF export, and printing
let pipelineInterval = null;      // Timer interval driving the live multi-agent visualizer
let agentStep = 0;                // Current active agent index in the pipeline animation

// Initialize application on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    initSessionBadge();       // Load or display existing session ID
    initKeyboardShortcuts();  // Bind Ctrl+Enter shortcut on the prompt textarea
    fetchLiveCacheStats();    // Fetch current Redis connection & token economy metrics
});

/* ==========================================================================
   Session & Thread Management
   ========================================================================== */
function initSessionBadge() {
    const badge = document.getElementById("activeThreadBadge");
    if (!badge) return;

    if (currentThreadId) {
        badge.textContent = currentThreadId;
    } else {
        badge.textContent = "session_new";
    }
}

function initKeyboardShortcuts() {
    const textarea = document.getElementById("userInput");
    if (!textarea) return;

    textarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            sendMessage();
        }
    });
}

function setPrompt(text) {
    const input = document.getElementById("userInput");
    if (!input) return;

    input.value = text;
    input.focus();

    // Subtle highlight animation
    input.style.borderColor = "var(--yellow-neon)";
    setTimeout(() => {
        input.style.borderColor = "";
    }, 800);
}

/* ==========================================================================
   Toast Notification System
   ========================================================================== */
function showToast(message, icon = "✨") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    // Trigger enter animation
    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/* ==========================================================================
   Error Banner Handling
   ========================================================================== */
function showError(msg) {
    const banner = document.getElementById("errorBanner");
    const msgEl = document.getElementById("errorMessage");
    if (!banner || !msgEl) return;

    msgEl.textContent = msg;
    banner.classList.remove("hidden");
    banner.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
    const banner = document.getElementById("errorBanner");
    if (banner) banner.classList.add("hidden");
}

/* ==========================================================================
   LangGraph Live Agent Pipeline Animation
   ========================================================================== */
function startPipelineAnimation() {
    const grid = document.getElementById("agentNodesGrid");
    const statusBadge = document.getElementById("pipelineStatusBadge");
    const nodes = [
        { id: "nodeFlight", statusId: "statusFlight", label: "Querying AviationStack..." },
        { id: "nodeHotel", statusId: "statusHotel", label: "Tavily AI scanning stays..." },
        { id: "nodeWeather", statusId: "statusWeather", label: "OpenWeather MCP checking climate..." },
        { id: "nodeItinerary", statusId: "statusItinerary", label: "Generating day schedule..." },
        { id: "nodeFinal", statusId: "statusFinal", label: "Synthesizing budget..." }
    ];

    if (grid) grid.classList.add("animated");
    if (statusBadge) {
        statusBadge.textContent = "RUNNING AGENTS";
        statusBadge.className = "pipeline-status-badge running";
    }

    // Reset all nodes
    nodes.forEach(n => {
        const el = document.getElementById(n.id);
        const st = document.getElementById(n.statusId);
        if (el) el.className = "agent-node";
        if (st) st.textContent = "Queued";
    });

    agentStep = 0;
    
    // Simulate pipeline traversal
    const stepDuration = 2200; // ms per node highlight
    pipelineInterval = setInterval(() => {
        if (agentStep < nodes.length) {
            // Mark previous as completed
            if (agentStep > 0) {
                const prev = document.getElementById(nodes[agentStep - 1].id);
                const prevSt = document.getElementById(nodes[agentStep - 1].statusId);
                if (prev) prev.className = "agent-node completed";
                if (prevSt) prevSt.textContent = "✓ Complete";
            }

            // Mark current as active
            const curr = document.getElementById(nodes[agentStep].id);
            const currSt = document.getElementById(nodes[agentStep].statusId);
            if (curr) curr.className = "agent-node active";
            if (currSt) currSt.textContent = nodes[agentStep].label;

            agentStep++;
        }
    }, stepDuration);
}

function completePipelineAnimation(isCached = false, latencyMs = 0) {
    clearInterval(pipelineInterval);

    const grid = document.getElementById("agentNodesGrid");
    const statusBadge = document.getElementById("pipelineStatusBadge");
    if (grid) grid.classList.remove("animated");
    if (statusBadge) {
        if (isCached) {
            statusBadge.textContent = `⚡ REDIS CACHE HIT (${latencyMs}ms)`;
            statusBadge.className = "pipeline-status-badge running";
        } else {
            statusBadge.textContent = `EXECUTION COMPLETE (${latencyMs}ms)`;
            statusBadge.className = "pipeline-status-badge";
        }
    }

    const nodeIds = [
        { id: "nodeFlight", statusId: "statusFlight", cachedText: "⚡ From Redis Flight Cache" },
        { id: "nodeHotel", statusId: "statusHotel", cachedText: "⚡ From Redis Hotel Cache" },
        { id: "nodeWeather", statusId: "statusWeather", cachedText: "⚡ From Redis Weather Cache" },
        { id: "nodeItinerary", statusId: "statusItinerary", cachedText: "⚡ 0 LLM Tokens Consumed" },
        { id: "nodeFinal", statusId: "statusFinal", cachedText: "⚡ Instant Cached Synthesis" }
    ];

    nodeIds.forEach(n => {
        const el = document.getElementById(n.id);
        const st = document.getElementById(n.statusId);
        if (el) el.className = "agent-node completed";
        if (st) st.textContent = isCached ? n.cachedText : "✓ Executed & Verified";
    });
}

function resetPipelineAnimation() {
    clearInterval(pipelineInterval);
    const grid = document.getElementById("agentNodesGrid");
    const statusBadge = document.getElementById("pipelineStatusBadge");
    if (grid) grid.classList.remove("animated");
    if (statusBadge) {
        statusBadge.textContent = "STANDBY";
        statusBadge.className = "pipeline-status-badge";
    }

    ["Flight", "Hotel", "Weather", "Itinerary", "Final"].forEach(name => {
        const el = document.getElementById(`node${name}`);
        const st = document.getElementById(`status${name}`);
        if (el) el.className = "agent-node";
        if (st) st.textContent = "Awaiting trigger";
    });
}

/* ==========================================================================
   Core Agent Invocation
   ========================================================================== */
async function sendMessage() {
    hideError();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel details or select one of the curated prompts.");
        return;
    }

    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    sendBtn.disabled = true;
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");

    startPipelineAnimation();

    // Smooth scroll to pipeline visualizer
    document.getElementById("agentPipelineCard")?.scrollIntoView({ behavior: "smooth", block: "start" });

    const startTime = performance.now();

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: currentThreadId
            })
        });

        const data = await response.json();
        const latencyMs = Math.round(performance.now() - startTime);

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Agent orchestration failed.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("dreamtrip_thread_id", currentThreadId);
        initSessionBadge();

        completePipelineAnimation(data.is_cached, latencyMs);
        displayResults(data, latencyMs);

        if (data.is_cached) {
            const savedTokens = (data.tokens_saved || 4500).toLocaleString();
            showToast(`⚡ Redis Cache HIT in ${latencyMs}ms! Saved ~${savedTokens} tokens.`, "⚡");
        } else {
            showToast(`Autonomous Plan Synthesized in ${latencyMs}ms! Saved to Redis.`, "🚀");
        }

        fetchLiveCacheStats();

    } catch (error) {
        resetPipelineAnimation();
        showError(error.message || "An unexpected error occurred during execution.");
        showToast("Execution error occurred.", "❌");
    } finally {
        sendBtn.disabled = false;
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

/* ==========================================================================
   Result Presentation & Dashboard Tabs
   ========================================================================== */
function displayResults(data, latencyMs = 0) {
    currentAnswerMarkdown = data.answer || "";

    const resultsWrapper = document.getElementById("resultsCardWrapper");
    const markdownRender = document.getElementById("markdownResultRender");
    const threadChip = document.getElementById("telemetryThreadId");
    const callsChip = document.getElementById("telemetryCalls");
    const cacheBadge = document.getElementById("telemetryCacheBadge");
    const latencyChip = document.getElementById("telemetryLatency");

    const telemetryLlmCalls = document.getElementById("telemetryLlmCalls");
    const telemetryCacheStatus = document.getElementById("telemetryCacheStatus");
    const telemetryCacheSub = document.getElementById("telemetryCacheSub");
    const telemetryLatencyVal = document.getElementById("telemetryLatencyVal");
    const telemetryLatencySub = document.getElementById("telemetryLatencySub");

    const rawFlight = document.getElementById("rawFlightBox");
    const rawHotel = document.getElementById("rawHotelBox");
    const rawWeather = document.getElementById("rawWeatherBox");
    const rawItinerary = document.getElementById("rawItineraryBox");
    const rawTelemetry = document.getElementById("rawTelemetryJson");

    // Markdown rendering
    if (typeof marked !== "undefined") {
        markdownRender.innerHTML = marked.parse(currentAnswerMarkdown);
    } else {
        markdownRender.innerText = currentAnswerMarkdown;
    }

    // Telemetry & chips
    const isCached = !!data.is_cached;
    const tokensSaved = (data.tokens_saved || (isCached ? 4500 : 0)).toLocaleString();

    if (threadChip) threadChip.textContent = `Thread: ${data.thread_id || "N/A"}`;
    if (callsChip) callsChip.textContent = `LLM Invocations: ${isCached ? 0 : (data.llm_calls ?? 2)}`;
    if (latencyChip) latencyChip.textContent = `Latency: ${latencyMs}ms`;

    if (cacheBadge) {
        if (isCached) {
            cacheBadge.className = "meta-chip highlight-cache";
            cacheBadge.innerHTML = `⚡ <strong>Redis HIT</strong> (~${tokensSaved} tokens saved)`;
        } else {
            cacheBadge.className = "meta-chip";
            cacheBadge.innerHTML = `⚡ Cache: Fresh (Saved to Redis)`;
        }
    }

    if (telemetryLlmCalls) telemetryLlmCalls.textContent = isCached ? 0 : (data.llm_calls ?? 2);
    if (telemetryCacheStatus) {
        telemetryCacheStatus.textContent = isCached ? "CACHE HIT" : "CACHE MISS";
        telemetryCacheStatus.style.color = isCached ? "var(--yellow-neon)" : "#ffffff";
    }
    if (telemetryCacheSub) {
        telemetryCacheSub.textContent = isCached ? `~${tokensSaved} Tokens Conserved` : "Saved for repeat queries";
    }
    if (telemetryLatencyVal) telemetryLatencyVal.textContent = `${latencyMs}ms`;
    if (telemetryLatencySub) {
        telemetryLatencySub.textContent = isCached ? "Sub-10ms Direct from Redis" : "Full Multi-Agent Graph";
    }

    if (rawFlight) rawFlight.textContent = data.flight_results || "// No dedicated flight stream payload available.";
    if (rawHotel) rawHotel.textContent = data.hotel_results || "// No dedicated hotel stream payload available.";
    if (rawWeather) rawWeather.textContent = data.weather_results || "// No dedicated weather stream payload available.";
    if (rawItinerary) rawItinerary.textContent = data.itinerary || "// No raw itinerary draft payload available.";

    // Render interactive weather visualizer cards
    renderWeatherDashboard(data.weather_results);

    if (rawTelemetry) {
        rawTelemetry.textContent = JSON.stringify({
            thread_id: data.thread_id,
            is_cached: isCached,
            cache_backend: data.cache_source || (isCached ? "Redis" : "Fresh Invocations"),
            tokens_saved: isCached ? data.tokens_saved : 0,
            latency_ms: latencyMs,
            llm_calls: isCached ? 0 : (data.llm_calls ?? 2),
            graph_nodes: ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent", "final_agent"],
            checkpoint_engine: "PostgreSQL (dict_row)",
            status: "SUCCESS_COMPILED",
            timestamp: new Date().toISOString()
        }, null, 2);
    }

    resultsWrapper.classList.remove("hidden");

    // Smooth scroll down to results
    setTimeout(() => {
        resultsWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
}

/* ==========================================================================
   Live Cache Stats & Controls
   ========================================================================== */
async function fetchLiveCacheStats() {
    try {
        const res = await fetch("/api/cache/stats");
        const stats = await res.json();
        const summaryEl = document.getElementById("cacheStatsSummary");
        const chip = document.getElementById("headerRedisChip");
        if (chip) {
            chip.textContent = stats.connected ? "⚡ Redis 8 Active" : "⚠️ In-Memory Fallback";
        }
        if (summaryEl) {
            const hits = stats.hits || 0;
            const misses = stats.misses || 0;
            const hitRate = stats.hit_ratio_percent || 0;
            const tokens = (stats.tokens_saved || 0).toLocaleString();
            summaryEl.innerHTML = `Backend: <strong style="color:var(--yellow-neon);">${stats.backend}</strong> • Cache Hits: <strong>${hits}</strong> • Misses: <strong>${misses}</strong> • Hit Rate: <strong>${hitRate}%</strong> • Cumulative Tokens Saved: <strong style="color:var(--yellow-neon);">~${tokens}</strong>`;
        }
    } catch (e) {
        console.warn("Failed to fetch cache stats:", e);
    }
}

async function flushRedisCache() {
    try {
        const res = await fetch("/api/cache/clear", { method: "POST" });
        const data = await res.json();
        showToast(`Redis Cache Cleared (${data.cleared_keys || 0} keys flushed)`, "🗑️");
        fetchLiveCacheStats();
        const badge = document.getElementById("telemetryCacheBadge");
        if (badge) {
            badge.className = "meta-chip";
            badge.textContent = "⚡ Cache Flushed";
        }
    } catch (e) {
        showToast("Failed to clear cache.", "❌");
    }
}

/* ==========================================================================
   Tab Switching Engine
   ========================================================================== */
function switchTab(tabKey) {
    const tabMapping = {
        masterPlan: "tabPanelMasterPlan",
        flightData: "tabPanelFlightData",
        hotelData: "tabPanelHotelData",
        weatherData: "tabPanelWeatherData",
        rawItinerary: "tabPanelRawItinerary",
        telemetry: "tabPanelTelemetry"
    };

    const targetPanelId = tabMapping[tabKey];
    if (!targetPanelId) return;

    // Update buttons
    const buttons = document.querySelectorAll(".studio-tab-btn");
    buttons.forEach(btn => {
        if (btn.getAttribute("onclick")?.includes(tabKey)) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    // Update panels
    const panels = document.querySelectorAll(".tab-panel");
    panels.forEach(p => p.classList.remove("active"));

    const targetPanel = document.getElementById(targetPanelId);
    if (targetPanel) targetPanel.classList.add("active");
}

/* ==========================================================================
   Weather Intelligence Dashboard Renderer
   ========================================================================== */
function renderWeatherDashboard(weatherStr) {
    const container = document.getElementById("weatherDashboardContainer");
    if (!container) return;

    if (!weatherStr || weatherStr.includes("unavailable") || weatherStr.trim() === "") {
        container.innerHTML = `
            <div class="weather-empty-state">
                <span style="font-size: 36px;">⛅</span>
                <p>${weatherStr || "Weather intelligence awaiting execution."}</p>
            </div>
        `;
        return;
    }

    try {
        let city = "Destination";
        let temp = "--";
        let feelsLike = "--";
        let humidity = "--";
        let condition = "Pleasant";
        let windSpeed = "--";
        let forecastList = [];

        // Helper for condition icon
        const getWeatherIcon = (cond) => {
            const c = (cond || "").toLowerCase();
            if (c.includes("rain") || c.includes("drizzle")) return "🌧️";
            if (c.includes("thunder") || c.includes("storm")) return "⛈️";
            if (c.includes("snow")) return "❄️";
            if (c.includes("cloud")) return "⛅";
            if (c.includes("clear") || c.includes("sun")) return "☀️";
            if (c.includes("mist") || c.includes("fog") || c.includes("haze")) return "🌫️";
            return "🌤️";
        };

        // Current weather parsing
        const cityMatch = weatherStr.match(/['"]city['"]\s*:\s*['"]([^'"]+)['"]/i);
        if (cityMatch) city = cityMatch[1];

        const tempMatch = weatherStr.match(/['"]temperature_c['"]\s*:\s*([0-9.-]+)/i);
        if (tempMatch) temp = Math.round(parseFloat(tempMatch[1]));

        const feelsMatch = weatherStr.match(/['"]feels_like_c['"]\s*:\s*([0-9.-]+)/i);
        if (feelsMatch) feelsLike = Math.round(parseFloat(feelsMatch[1]));

        const humMatch = weatherStr.match(/['"]humidity['"]\s*:\s*([0-9]+)/i);
        if (humMatch) humidity = humMatch[1];

        const condMatch = weatherStr.match(/['"]condition['"]\s*:\s*['"]([^'"]+)['"]/i);
        if (condMatch) condition = condMatch[1];

        const windMatch = weatherStr.match(/['"]wind_speed['"]\s*:\s*([0-9.]+)/i);
        if (windMatch) windSpeed = windMatch[1];

        // Parse Forecast items
        const itemRegex = /\{['"]datetime['"]\s*:\s*['"]([^'"]+)['"]\s*,\s*['"]temperature['"]\s*:\s*([0-9.-]+)\s*,\s*['"]weather['"]\s*:\s*['"]([^'"]+)['"]\}/g;
        let match;
        while ((match = itemRegex.exec(weatherStr)) !== null) {
            forecastList.push({
                datetime: match[1],
                temperature: Math.round(parseFloat(match[2])),
                weather: match[3]
            });
        }

        const mainIcon = getWeatherIcon(condition);

        let forecastCardsHtml = "";
        if (forecastList.length > 0) {
            forecastCardsHtml = forecastList.slice(0, 5).map(item => {
                const icon = getWeatherIcon(item.weather);
                const parts = item.datetime.split(" ");
                const datePart = parts[0] ? parts[0].substring(5) : "Day"; // MM-DD
                const timePart = parts[1] ? parts[1].substring(0, 5) : "";
                return `
                    <div class="forecast-card">
                        <div class="forecast-date">${datePart}</div>
                        ${timePart ? `<div class="forecast-time">${timePart}</div>` : ""}
                        <div class="forecast-icon">${icon}</div>
                        <div class="forecast-temp">${item.temperature}°C</div>
                        <div class="forecast-condition">${item.weather}</div>
                    </div>
                `;
            }).join("");
        }

        container.innerHTML = `
            <div class="weather-hero-card">
                <div class="weather-hero-main">
                    <div class="weather-location-pill">
                        <span>📍</span>
                        <strong>${city}</strong>
                        <span class="live-tag">LIVE OPENWEATHER MCP</span>
                    </div>
                    <div class="weather-temp-row">
                        <span class="weather-huge-icon">${mainIcon}</span>
                        <div class="weather-temp-digits">
                            <span class="temp-val">${temp}</span>
                            <span class="temp-unit">°C</span>
                        </div>
                        <div class="weather-cond-badge">
                            <span class="cond-title">${condition.toUpperCase()}</span>
                            <span class="cond-sub">Feels like ${feelsLike}°C</span>
                        </div>
                    </div>
                </div>

                <div class="weather-metrics-deck">
                    <div class="weather-metric-pill">
                        <div class="metric-icon">🌡️</div>
                        <div class="metric-info">
                            <span class="metric-lbl">Feels Like</span>
                            <span class="metric-val">${feelsLike}°C</span>
                        </div>
                    </div>
                    <div class="weather-metric-pill">
                        <div class="metric-icon">💧</div>
                        <div class="metric-info">
                            <span class="metric-lbl">Humidity</span>
                            <span class="metric-val">${humidity}%</span>
                        </div>
                    </div>
                    <div class="weather-metric-pill">
                        <div class="metric-icon">💨</div>
                        <div class="metric-info">
                            <span class="metric-lbl">Wind Speed</span>
                            <span class="metric-val">${windSpeed} m/s</span>
                        </div>
                    </div>
                </div>
            </div>

            ${forecastList.length > 0 ? `
                <div class="weather-forecast-section">
                    <div class="forecast-section-title">
                        <span>📅</span>
                        <h4>5-Day Multi-Horizon Weather Forecast</h4>
                    </div>
                    <div class="forecast-cards-grid">
                        ${forecastCardsHtml}
                    </div>
                </div>
            ` : ""}
        `;

    } catch (err) {
        console.error("Error rendering weather dashboard:", err);
        container.innerHTML = `
            <div class="weather-empty-state">
                <span style="font-size: 32px;">⛅</span>
                <p>Live Weather Stream Available Below</p>
            </div>
        `;
    }
}

/* ==========================================================================
   Export & Utility Actions
   ========================================================================== */
function copyMasterResult() {
    if (!currentAnswerMarkdown) {
        showToast("No generated plan to copy.", "⚠️");
        return;
    }

    navigator.clipboard.writeText(currentAnswerMarkdown)
        .then(() => {
            const btnText = document.getElementById("copyBtnText");
            if (btnText) {
                const old = btnText.textContent;
                btnText.textContent = "Copied!";
                setTimeout(() => btnText.textContent = old, 1500);
            }
            showToast("Plan Markdown copied to clipboard!", "📋");
        })
        .catch(() => {
            showToast("Clipboard access failed.", "❌");
        });
}

function buildPrintableHTML(markdownText, threadId) {
    const dateStr = new Date().toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric"
    });

    const parsedContent = typeof marked !== "undefined"
        ? marked.parse(markdownText)
        : markdownText.replace(/\n/g, "<br>");

    return `
        <div id="pdfPrintArea" style="width: 760px; background: #ffffff !important; color: #0f172a !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 32px 36px; box-sizing: border-box;">
            <style>
                #pdfPrintArea * {
                    box-sizing: border-box !important;
                    color: #0f172a !important;
                    -webkit-text-fill-color: initial !important;
                    text-shadow: none !important;
                    box-shadow: none !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
                }
                #pdfPrintArea .pdf-header {
                    border-bottom: 2.5px solid #881337 !important;
                    padding-bottom: 12px !important;
                    margin-bottom: 20px !important;
                    display: flex !important;
                    justify-content: space-between !important;
                    align-items: flex-end !important;
                }
                #pdfPrintArea .pdf-title {
                    font-size: 20px !important;
                    font-weight: 900 !important;
                    color: #881337 !important;
                    margin: 0 !important;
                    line-height: 1.2 !important;
                    -webkit-text-fill-color: #881337 !important;
                    letter-spacing: -0.01em !important;
                }
                #pdfPrintArea .pdf-subtitle {
                    font-size: 10px !important;
                    color: #64748b !important;
                    margin-top: 4px !important;
                    font-weight: 700 !important;
                    letter-spacing: 0.08em !important;
                    text-transform: uppercase !important;
                    -webkit-text-fill-color: #64748b !important;
                }
                #pdfPrintArea .pdf-meta {
                    font-size: 10px !important;
                    color: #334155 !important;
                    text-align: right !important;
                    line-height: 1.5 !important;
                    -webkit-text-fill-color: #334155 !important;
                }
                #pdfPrintArea h1 {
                    color: #881337 !important;
                    font-size: 18px !important;
                    font-weight: 800 !important;
                    border-bottom: 2px solid #d97706 !important;
                    padding-bottom: 5px !important;
                    margin-top: 18px !important;
                    margin-bottom: 10px !important;
                    -webkit-text-fill-color: #881337 !important;
                    page-break-after: avoid !important;
                    break-after: avoid !important;
                }
                #pdfPrintArea h2 {
                    color: #881337 !important;
                    font-size: 14px !important;
                    font-weight: 800 !important;
                    background: #fff1f2 !important;
                    border-left: 4px solid #be123c !important;
                    padding: 6px 12px !important;
                    border-radius: 4px !important;
                    margin-top: 16px !important;
                    margin-bottom: 8px !important;
                    -webkit-text-fill-color: #881337 !important;
                    page-break-after: avoid !important;
                    break-after: avoid !important;
                }
                #pdfPrintArea h3 {
                    color: #b45309 !important;
                    font-size: 12.5px !important;
                    font-weight: 800 !important;
                    margin-top: 12px !important;
                    margin-bottom: 6px !important;
                    -webkit-text-fill-color: #b45309 !important;
                    page-break-after: avoid !important;
                    break-after: avoid !important;
                }
                #pdfPrintArea h4 {
                    color: #0f172a !important;
                    font-size: 11.5px !important;
                    font-weight: 700 !important;
                    margin-top: 10px !important;
                    margin-bottom: 4px !important;
                }
                #pdfPrintArea p {
                    color: #1e293b !important;
                    font-size: 11px !important;
                    line-height: 1.55 !important;
                    margin-bottom: 8px !important;
                    -webkit-text-fill-color: #1e293b !important;
                }
                #pdfPrintArea strong, #pdfPrintArea b {
                    color: #000000 !important;
                    font-weight: 800 !important;
                    -webkit-text-fill-color: #000000 !important;
                }
                #pdfPrintArea ul, #pdfPrintArea ol {
                    color: #1e293b !important;
                    font-size: 11px !important;
                    line-height: 1.55 !important;
                    margin: 4px 0 8px 18px !important;
                    padding: 0 !important;
                }
                #pdfPrintArea li {
                    color: #1e293b !important;
                    margin-bottom: 4px !important;
                    -webkit-text-fill-color: #1e293b !important;
                    page-break-inside: avoid !important;
                    break-inside: avoid !important;
                }
                #pdfPrintArea table {
                    width: 100% !important;
                    border-collapse: collapse !important;
                    margin: 10px 0 !important;
                    font-size: 10.5px !important;
                    page-break-inside: avoid !important;
                    break-inside: avoid !important;
                }
                #pdfPrintArea th {
                    background: #881337 !important;
                    color: #ffffff !important;
                    font-weight: 800 !important;
                    border: 1px solid #881337 !important;
                    padding: 6px 8px !important;
                    text-align: left !important;
                    -webkit-text-fill-color: #ffffff !important;
                }
                #pdfPrintArea td {
                    color: #1e293b !important;
                    border: 1px solid #cbd5e1 !important;
                    padding: 5px 8px !important;
                    background: #ffffff !important;
                    -webkit-text-fill-color: #1e293b !important;
                }
                #pdfPrintArea tr:nth-child(even) td {
                    background: #f8fafc !important;
                }
                #pdfPrintArea hr {
                    border: none !important;
                    border-top: 1px solid #e2e8f0 !important;
                    margin: 14px 0 !important;
                }
                #pdfPrintArea blockquote {
                    background: #f8fafc !important;
                    border-left: 3px solid #be123c !important;
                    padding: 6px 10px !important;
                    margin: 8px 0 !important;
                    color: #334155 !important;
                    font-size: 10.5px !important;
                    page-break-inside: avoid !important;
                }
                #pdfPrintArea .pdf-footer {
                    border-top: 1px solid #cbd5e1 !important;
                    margin-top: 20px !important;
                    padding-top: 6px !important;
                    font-size: 9.5px !important;
                    color: #64748b !important;
                    text-align: center !important;
                    -webkit-text-fill-color: #64748b !important;
                }
            </style>
            <div class="pdf-header">
                <div>
                    <h1 class="pdf-title">DreamTrip AI — Travel Master Plan</h1>
                    <div class="pdf-subtitle">LANGGRAPH MULTI-AGENT EXPEDITION REPORT • REDIS CACHED</div>
                </div>
                <div class="pdf-meta">
                    <div><strong>Date:</strong> ${dateStr}</div>
                    <div><strong>Session:</strong> ${threadId || "session_export"}</div>
                    <div><strong>Engine:</strong> LangGraph + Gemini 2.5 Flash</div>
                </div>
            </div>
            <div class="pdf-body">
                ${parsedContent}
            </div>
            <div class="pdf-footer">
                Generated by DreamTrip AI • Multi-Agent Travel Orchestration Engine • High-Performance Token Optimization
            </div>
        </div>
    `;
}

function exportPDF() {
    if (!currentAnswerMarkdown) {
        showToast("No plan available for PDF export.", "⚠️");
        return;
    }

    const btnText = document.getElementById("pdfBtnText");
    const oldText = btnText ? btnText.textContent : "Download PDF";
    if (btnText) btnText.textContent = "Rendering PDF...";

    // 1. Create a foreground rendering modal so html2canvas renders with 100% full opacity and contrast
    const overlay = document.createElement("div");
    overlay.id = "pdfStagingOverlay";
    overlay.style.position = "fixed";
    overlay.style.top = "0";
    overlay.style.left = "0";
    overlay.style.width = "100vw";
    overlay.style.height = "100vh";
    overlay.style.background = "rgba(10, 3, 7, 0.88)";
    overlay.style.backdropFilter = "blur(6px)";
    overlay.style.zIndex = "999999";
    overlay.style.display = "flex";
    overlay.style.flexDirection = "column";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "flex-start";
    overlay.style.overflowY = "auto";
    overlay.style.padding = "24px 12px";

    const banner = document.createElement("div");
    banner.style.cssText = "color: #ffe600; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px; font-weight: 800; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; background: rgba(24, 10, 16, 0.95); padding: 8px 18px; border-radius: 9999px; border: 1px solid rgba(255, 183, 3, 0.4);";
    banner.innerHTML = `<span>📥</span><span>Generating Executive PDF Master Plan...</span>`;
    overlay.appendChild(banner);

    const printContainer = document.createElement("div");
    printContainer.style.boxShadow = "0 20px 60px rgba(0,0,0,0.85)";
    printContainer.style.borderRadius = "4px";
    printContainer.style.overflow = "hidden";
    printContainer.innerHTML = buildPrintableHTML(currentAnswerMarkdown, currentThreadId);
    overlay.appendChild(printContainer);

    document.body.appendChild(overlay);

    const targetElement = printContainer.querySelector("#pdfPrintArea");

    const opt = {
        margin: [10, 10, 10, 10], // 10mm margins
        filename: `DreamTrip-Plan-${currentThreadId || "export"}.pdf`,
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff",
            logging: false,
            windowWidth: 800,
            scrollY: 0
        },
        jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
        pagebreak: { mode: ["css", "legacy"] }
    };

    if (typeof html2pdf !== "undefined") {
        html2pdf().set(opt).from(targetElement).save()
            .then(() => {
                overlay.remove();
                if (btnText) btnText.textContent = oldText;
                showToast("Executive PDF downloaded successfully!", "📄");
            })
            .catch(err => {
                console.error("html2pdf error:", err);
                overlay.remove();
                if (btnText) btnText.textContent = oldText;
                showToast("Opening Print dialogue fallback...", "🖨️");
                printPlan();
            });
    } else {
        overlay.remove();
        if (btnText) btnText.textContent = oldText;
        showToast("Opening print window...", "🖨️");
        printPlan();
    }
}

function printPlan() {
    if (!currentAnswerMarkdown) {
        showToast("No plan available to print.", "⚠️");
        return;
    }

    const htmlContent = buildPrintableHTML(currentAnswerMarkdown, currentThreadId);
    const printWindow = window.open("", "_blank");
    if (!printWindow) {
        window.print();
        return;
    }

    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>DreamTrip AI — Travel Master Plan</title>
            <meta charset="utf-8">
            <style>
                body { margin: 0; padding: 20px; background: #ffffff; display: flex; justify-content: center; }
                @page { size: A4 portrait; margin: 12mm; }
            </style>
        </head>
        <body>
            ${htmlContent}
            <script>
                window.onload = function() {
                    window.focus();
                    window.print();
                    setTimeout(function() { window.close(); }, 1000);
                };
            </script>
        </body>
        </html>
    `);
    printWindow.document.close();
}