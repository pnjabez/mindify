// ============================================================
// script.js  –  Mindify Frontend Logic
// Handles user input, async fetch to /api/generate & /api/history,
// tab switching, NLP Developer Inspector panel, and DOM updates.
// ============================================================

"use strict";

document.addEventListener("DOMContentLoaded", () => {
  // ── DOM references ───────────────────────────────────────────
  const generateBtn        = document.getElementById("generate-btn");
  const manifestationInput  = document.getElementById("manifestation-input");
  const outputSection      = document.getElementById("output-section");
  const errorBanner        = document.getElementById("error-banner");
  const errorMessage       = document.getElementById("error-message");

  // Output block content elements
  const outOriginalGoal    = document.getElementById("out-original-goal");
  const outAffirmation     = document.getElementById("out-affirmation");

  // NLP Developer Mode Inspector elements
  const devModeToggle      = document.getElementById("devModeToggle");
  const nlpInspector       = document.getElementById("nlpInspector");
  const nlpDebugOutput     = document.getElementById("nlpDebugOutput");

  // Tabs & Views
  const tabGenerate        = document.getElementById("tabGenerate");
  const tabHistory         = document.getElementById("tabHistory");
  const generatorView      = document.getElementById("generatorView");
  const historyView        = document.getElementById("historyView");
  const historyContainer  = document.getElementById("historyContainer");


  // ── Helpers ──────────────────────────────────────────────────

  /**
   * Put the button in its loading state (disabled + spinner visible).
   */
  function setButtonLoading(isLoading) {
    if (!generateBtn) return;
    if (isLoading) {
      generateBtn.disabled = true;
      generateBtn.classList.add("loading");
      generateBtn.setAttribute("aria-busy", "true");
    } else {
      generateBtn.disabled = false;
      generateBtn.classList.remove("loading");
      generateBtn.setAttribute("aria-busy", "false");
    }
  }

  /**
   * Show or hide the error banner with a given message.
   * @param {string|null} message – Pass null to hide the banner.
   */
  function showError(message) {
    if (!errorBanner || !errorMessage) return;
    if (message) {
      errorMessage.textContent = message;
      errorBanner.classList.add("visible");
      errorBanner.setAttribute("role", "alert");
    } else {
      errorBanner.classList.remove("visible");
      errorMessage.textContent = "";
    }
  }

  /**
   * Populate output blocks with API response data, inject NLP debug telemetry,
   * and reveal the section.
   * @param {Object} data – JSON response from /api/generate
   */
  function populateOutput(data) {
    if (outOriginalGoal) outOriginalGoal.textContent = data.original_goal ?? "—";
    if (outAffirmation)  outAffirmation.textContent  = data.generated_affirmation ?? "—";

    // Format and inject intermediate NLP debug diagnostic payload into terminal pre tag
    if (data && data.nlp_debug && nlpDebugOutput) {
      nlpDebugOutput.textContent = JSON.stringify(data.nlp_debug, null, 2);
    } else if (nlpDebugOutput) {
      nlpDebugOutput.textContent = "No debug data received.";
    }

    if (outputSection) {
      outputSection.classList.remove("visible");
      void outputSection.offsetWidth; // force reflow for animation replay
      outputSection.classList.add("visible");
    }
  }

  /**
   * Format timestamp string into a clean readable date/time display.
   */
  function formatDate(timeStr) {
    if (!timeStr) return "Just now";
    try {
      const d = new Date(timeStr);
      if (isNaN(d.getTime())) return timeStr;
      return d.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
      });
    } catch {
      return timeStr;
    }
  }


  // ── Developer Mode Inspector Toggle ──────────────────────────

  if (devModeToggle && nlpInspector) {
    devModeToggle.addEventListener("click", () => {
      nlpInspector.classList.toggle("hidden");
    });
  }


  // ── History & Trends Loader ──────────────────────────────────

  async function loadHistory() {
    if (!historyContainer) return;
    historyContainer.innerHTML = '<div class="empty-history"><span>⏳</span>Loading history...</div>';

    try {
      const response = await fetch("/api/history");
      if (!response.ok) {
        throw new Error(`Failed to load history (${response.status})`);
      }

      const historyItems = await response.json();

      if (!Array.isArray(historyItems) || historyItems.length === 0) {
        historyContainer.innerHTML = `
          <div class="empty-history">
            <span>🌱</span>
            <p>No saved manifestations yet.</p>
            <p style="font-size:0.85rem; color:var(--color-muted); margin-top:0.25rem;">
              Generate your first affirmation to see your history &amp; trends here!
            </p>
          </div>
        `;
        return;
      }

      // Build history cards
      historyContainer.innerHTML = "";
      historyItems.forEach((item) => {
        const card = document.createElement("div");
        card.className = "history-card";

        let badgesHtml = "";
        if (Array.isArray(item.distortions) && item.distortions.length > 0) {
          badgesHtml = item.distortions
            .map((d) => `<span class="badge badge-distortion">${escapeHtml(String(d))}</span>`)
            .join("");
        }

        card.innerHTML = `
          <div class="history-header">
            <div class="history-time">
              <span>📅</span> ${escapeHtml(formatDate(item.timestamp))}
            </div>
            <div class="history-badges">
              ${badgesHtml}
            </div>
          </div>

          <div class="history-goal">
            <strong>Goal:</strong> "${escapeHtml(item.goal || "—")}"
          </div>

          <div class="history-affirmation">
            💬 ${escapeHtml(item.affirmation || "—")}
          </div>
        `;

        historyContainer.appendChild(card);
      });

    } catch (err) {
      historyContainer.innerHTML = `
        <div class="empty-history">
          <span>⚠️</span>
          <p>${escapeHtml(err.message || "Failed to load history.")}</p>
        </div>
      `;
    }
  }

  /**
   * XSS escape utility helper.
   */
  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }


  // ── Tab Switching Logic ──────────────────────────────────────

  function switchTab(target) {
    if (target === "generator") {
      if (tabGenerate) tabGenerate.classList.add("active");
      if (tabHistory)  tabHistory.classList.remove("active");
      if (generatorView) generatorView.classList.remove("hidden");
      if (historyView)   historyView.classList.add("hidden");
    } else if (target === "history") {
      if (tabHistory)  tabHistory.classList.add("active");
      if (tabGenerate) tabGenerate.classList.remove("active");
      if (historyView)   historyView.classList.remove("hidden");
      if (generatorView) generatorView.classList.add("hidden");
      loadHistory();
    }
  }

  if (tabGenerate && tabHistory) {
    tabGenerate.addEventListener("click", () => switchTab("generator"));
    tabHistory.addEventListener("click", () => switchTab("history"));
  }


  // ── Core: Generate handler ───────────────────────────────────

  async function handleGenerate() {
    if (!manifestationInput) return;
    const userText = manifestationInput.value.trim();
    if (!userText) {
      showError("Please describe your goal or manifestation before generating.");
      manifestationInput.focus();
      return;
    }

    showError(null);
    setButtonLoading(true);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manifestation: userText }),
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error("The server returned an unexpected response. Please try again.");
      }

      if (!response.ok) {
        const serverMsg = data?.error ?? `Server error (${response.status}).`;
        throw new Error(serverMsg);
      }

      populateOutput(data);

    } catch (err) {
      showError(err.message || "An unexpected error occurred. Please try again.");
      if (outputSection) outputSection.classList.remove("visible");
    } finally {
      setButtonLoading(false);
    }
  }


  // ── Event Listeners ──────────────────────────────────────────

  if (generateBtn) {
    generateBtn.addEventListener("click", handleGenerate);
  }

  if (manifestationInput) {
    manifestationInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        handleGenerate();
      }
    });

    manifestationInput.addEventListener("input", () => {
      if (errorBanner && errorBanner.classList.contains("visible")) {
        showError(null);
      }
    });
  }
});
