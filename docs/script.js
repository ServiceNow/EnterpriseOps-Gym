/* ============================================
   EnterpriseOps-Gym — Client-side interactivity
   ============================================ */

// ── Leaderboard Data ──
const LEADERBOARD_DATA = [
  // Closed Source
  { model: "Claude Opus 4.5", type: "closed", teams: 50.0, csm: 34.2, email: 51.9, itsm: 23.8, calendar: 43.2, hr: 32.1, drive: 49.5, hybrid: 30.7, avg: 37.0 },
  { model: "Gemini-3-Flash", type: "closed", teams: 47.3, csm: 35.0, email: 44.3, itsm: 28.5, calendar: 30.5, hr: 12.6, drive: 49.7, hybrid: 24.2, avg: 31.7 },
  { model: "Claude Sonnet 4.5", type: "closed", teams: 51.0, csm: 16.7, email: 51.3, itsm: 17.6, calendar: 34.6, hr: 21.6, drive: 52.1, hybrid: 28.1, avg: 30.5 },
  { model: "GPT-5.2 (High)", type: "closed", teams: 31.0, csm: 34.8, email: 51.0, itsm: 21.7, calendar: 38.5, hr: 25.0, drive: 40.0, hybrid: 22.2, avg: 31.3 },
  { model: "GPT-5", type: "closed", teams: 26.3, csm: 36.4, email: 49.0, itsm: 18.9, calendar: 41.3, hr: 17.9, drive: 34.0, hybrid: 23.5, avg: 29.2 },
  { model: "Gemini-3-Pro", type: "closed", teams: 43.0, csm: 27.7, email: 33.6, itsm: 22.2, calendar: 28.8, hr: 12.5, drive: 46.7, hybrid: 22.9, avg: 27.4 },
  { model: "GPT-5.2 (Low)", type: "closed", teams: 25.0, csm: 21.2, email: 43.3, itsm: 6.7, calendar: 28.9, hr: 13.0, drive: 26.7, hybrid: 20.9, avg: 21.1 },
  { model: "GPT-5-Mini", type: "closed", teams: 25.7, csm: 15.8, email: 47.4, itsm: 8.9, calendar: 28.8, hr: 10.7, drive: 23.8, hybrid: 22.5, avg: 20.6 },
  { model: "Gemini-2.5-Pro", type: "closed", teams: 39.3, csm: 11.6, email: 31.1, itsm: 13.9, calendar: 12.5, hr: 4.9, drive: 27.0, hybrid: 19.6, avg: 17.8 },
  // Open Source
  { model: "DeepSeek-V3.2 (High)", type: "open", teams: 37.0, csm: 14.1, email: 47.1, itsm: 16.1, calendar: 21.2, hr: 16.3, drive: 35.2, hybrid: 22.9, avg: 23.8 },
  { model: "GPT-OSS-120B (High)", type: "open", teams: 32.0, csm: 16.3, email: 42.3, itsm: 6.1, calendar: 35.6, hr: 16.3, drive: 41.0, hybrid: 19.6, avg: 23.0 },
  { model: "DeepSeek-V3.2 (Medium)", type: "open", teams: 35.7, csm: 15.4, email: 45.8, itsm: 9.6, calendar: 21.5, hr: 15.0, drive: 27.6, hybrid: 22.9, avg: 21.8 },
  { model: "Kimi-K2-Thinking", type: "open", teams: 30.0, csm: 7.1, email: 51.0, itsm: 12.2, calendar: 15.4, hr: 8.2, drive: 39.6, hybrid: 15.7, avg: 19.2 },
  { model: "Qwen3-30B (Think)", type: "open", teams: 22.0, csm: 5.4, email: 51.9, itsm: 6.7, calendar: 18.3, hr: 7.6, drive: 25.7, hybrid: 15.7, avg: 16.3 },
  { model: "Qwen3-235B (Inst.)", type: "open", teams: 28.0, csm: 4.7, email: 38.1, itsm: 9.3, calendar: 15.7, hr: 7.8, drive: 23.8, hybrid: 17.7, avg: 15.8 },
  { model: "Qwen3-4B (Think)", type: "open", teams: 24.0, csm: 3.8, email: 38.4, itsm: 5.6, calendar: 5.8, hr: 7.1, drive: 21.9, hybrid: 15.8, avg: 13.2 },
];

const DOMAIN_COLS = ["teams", "csm", "email", "itsm", "calendar", "hr", "drive", "hybrid", "avg"];

// ── State ──
let currentFilter = "all";
let sortCol = "avg";
let sortDir = "desc";

// ── DOM Ready ──
document.addEventListener("DOMContentLoaded", () => {
  renderLeaderboard();
  initSortHeaders();
  initFilterTabs();
  initScrollReveal();
  initStatCounters();
  initCopyBibtex();
  initNavToggle();
  initThemeToggle();
});

// ── Leaderboard Rendering ──
function renderLeaderboard() {
  const tbody = document.getElementById("leaderboardBody");
  let data = LEADERBOARD_DATA.slice();

  // Filter
  if (currentFilter !== "all") {
    data = data.filter(d => d.type === currentFilter);
  }

  // Sort
  data.sort((a, b) => {
    const va = typeof a[sortCol] === "string" ? a[sortCol].toLowerCase() : a[sortCol];
    const vb = typeof b[sortCol] === "string" ? b[sortCol].toLowerCase() : b[sortCol];
    if (va < vb) return sortDir === "asc" ? -1 : 1;
    if (va > vb) return sortDir === "asc" ? 1 : -1;
    return 0;
  });

  // Find column bests (for highlighting)
  const bests = {};
  DOMAIN_COLS.forEach(col => {
    bests[col] = Math.max(...data.map(d => d[col]));
  });

  // Build rows
  let html = "";
  data.forEach((row, idx) => {
    const rank = idx + 1;
    const rankBadge = rank <= 3
      ? `<span class="rank-badge rank-${rank}">${rank}</span>`
      : `<span style="color:var(--text-muted)">${rank}</span>`;

    const tag = row.type === "closed"
      ? `<span class="model-tag closed">Closed</span>`
      : `<span class="model-tag open">Open</span>`;

    let cells = `<td>${rankBadge}</td><td>${row.model}${tag}</td>`;
    DOMAIN_COLS.forEach(col => {
      const isBest = row[col] === bests[col] ? "cell-best" : "";
      const isAvg = col === "avg" ? "col-avg" : "";
      const classes = [isBest, isAvg].filter(Boolean).join(" ");
      const val = row[col].toFixed(1);
      cells += `<td class="${classes}">${val}</td>`;
    });

    html += `<tr data-type="${row.type}">${cells}</tr>`;
  });

  tbody.innerHTML = html;

  // Update header sort indicators
  document.querySelectorAll(".leaderboard-table th").forEach(th => {
    th.classList.remove("sorted-asc", "sorted-desc");
    if (th.dataset.col === sortCol) {
      th.classList.add(sortDir === "asc" ? "sorted-asc" : "sorted-desc");
      const arrow = th.querySelector(".sort-arrow");
      if (arrow) arrow.textContent = sortDir === "asc" ? "↑" : "↓";
    } else {
      const arrow = th.querySelector(".sort-arrow");
      if (arrow) arrow.textContent = "↕";
    }
  });
}

// ── Sorting ──
function initSortHeaders() {
  document.querySelectorAll(".leaderboard-table th[data-col]").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (col === "rank") return; // Don't sort by rank directly
      if (col === sortCol) {
        sortDir = sortDir === "desc" ? "asc" : "desc";
      } else {
        sortCol = col;
        sortDir = col === "model" ? "asc" : "desc";
      }
      renderLeaderboard();
    });
  });
}

// ── Filter Tabs ──
function initFilterTabs() {
  document.querySelectorAll(".leaderboard-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".leaderboard-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentFilter = tab.dataset.filter;
      renderLeaderboard();
    });
  });
}

// ── Scroll Reveal ──
function initScrollReveal() {
  const reveals = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("revealed");
        // Trigger stat counter if inside stats-strip
        if (entry.target.classList.contains("stats-strip") ||
          entry.target.closest(".stats-strip")) {
          animateCounters();
        }
      }
    });
  }, { threshold: 0.15 });

  reveals.forEach(el => observer.observe(el));
}

// ── Stat Counter Animation ──
let countersAnimated = false;
function animateCounters() {
  if (countersAnimated) return;
  countersAnimated = true;

  document.querySelectorAll(".stat-value[data-target]").forEach(el => {
    const target = parseFloat(el.dataset.target);
    const suffix = el.dataset.suffix || "";
    const isFloat = target % 1 !== 0;
    const duration = 1500;
    const startTime = performance.now();

    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;
      el.textContent = (isFloat ? current.toFixed(1) : Math.round(current)) + suffix;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

// ── BibTeX Copy ──
function initCopyBibtex() {
  const btn = document.getElementById("copyBibtex");
  const content = document.getElementById("bibtexContent");
  if (!btn || !content) return;

  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(content.textContent).then(() => {
      btn.textContent = "Copied!";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.classList.remove("copied");
      }, 2000);
    });
  });
}

// ── Mobile Nav Toggle ──
function initNavToggle() {
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (!toggle || !links) return;

  toggle.addEventListener("click", () => {
    links.classList.toggle("open");
  });

  // Close on link click
  links.querySelectorAll("a").forEach(a => {
    a.addEventListener("click", () => links.classList.remove("open"));
  });
}

// ── Theme Toggle ──
function initThemeToggle() {
  const toggle = document.getElementById("themeToggle");
  if (!toggle) return;

  // Check for saved preference
  const savedTheme = localStorage.getItem("eog-theme");
  if (savedTheme) {
    document.documentElement.setAttribute("data-theme", savedTheme);
  }

  toggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("eog-theme", next);
  });
}
