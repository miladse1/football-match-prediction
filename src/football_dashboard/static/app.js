/* MatchLab dashboard.
   Read-only client for the FastAPI endpoints. Renders stored probabilities
   exactly as the API returns them and never recomputes a number. */

const view = document.getElementById("view");
const seasonLabel = document.getElementById("season-label");
const footMeta = document.getElementById("foot-meta");
const PAGE_SIZE = 16;

/* ---------------------------------------------------------------- utils -- */

function pathOf() {
  const path = window.location.pathname.replace(/\/$/, "");
  return path || "/";
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function qs(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value) search.set(key, value);
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}

function html(strings, ...values) {
  return strings.reduce((out, chunk, i) => out + chunk + (values[i] ?? ""), "");
}

function parseDate(dateStr, timeStr) {
  return new Date(`${dateStr}T${timeStr || "00:00"}:00`);
}

function formatKickoff(dateStr, timeStr) {
  const day = parseDate(dateStr, timeStr).toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  return timeStr ? `${day} · ${timeStr}` : day;
}

function formatDay(dateStr) {
  return parseDate(dateStr, "12:00").toLocaleDateString("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

function formatShortDate(dateStr) {
  return parseDate(dateStr, "12:00").toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "2-digit",
  });
}

function formatStamp(iso) {
  if (!iso) return "Unknown";
  return new Date(iso).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function pct(value) {
  if (value == null) return "—";
  return `${Math.round(value * 100)}%`;
}

function fmtPct(value) {
  if (value == null) return "—";
  const p = value * 100;
  if (p >= 1) return `${p.toFixed(1)}%`;
  if (p > 0) return `${p.toFixed(2)}%`;
  return "0%";
}

function fmtNum(value, digits = 1) {
  if (value == null) return "—";
  return Number(value).toFixed(digits);
}

function setSeason(label) {
  seasonLabel.textContent = label;
}

function setFootMeta(model, extra) {
  if (!model) {
    footMeta.textContent = "";
    return;
  }
  const bits = [`${model.model_name} · ${model.feature_version}`];
  if (extra) bits.push(extra);
  footMeta.textContent = bits.join(" · ");
}

function setNav() {
  const current = pathOf();
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const target = link.dataset.nav;
    const active =
      current === target ||
      (target === "/upcoming" && current.startsWith("/upcoming/")) ||
      (target === "/results" && current.startsWith("/results/"));
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  const active = document.querySelector(".nav a.active");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function matchDetailRoute() {
  const upcoming = pathOf().match(/^\/upcoming\/(\d+)$/);
  if (upcoming) return { kind: "upcoming", id: Number(upcoming[1]) };
  const result = pathOf().match(/^\/results\/(\d+)$/);
  if (result) return { kind: "result", id: Number(result[1]) };
  return null;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = "Could not load this page from the database.";
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (error) {
      /* keep the default message */
    }
    throw new Error(detail);
  }
  return response.json();
}

/* --------------------------------------------------------------- crests -- */

function crestHtml(team, crest, sizeClass = "crest-sm") {
  const badge = crest || {};
  const initials = escapeHtml(badge.initials || String(team || "?").slice(0, 3).toUpperCase());
  const color = escapeHtml(badge.color || "#3d4338");
  const ink = escapeHtml(badge.ink || "#f4f1e8");
  const fallback = `<span class="crest-fallback" style="background:${color};color:${ink}">${initials}</span>`;
  if (!badge.crest_url) {
    return `<span class="crest-wrap ${sizeClass}">${fallback}</span>`;
  }
  return html`<span class="crest-wrap ${sizeClass}"
    >${fallback}<img class="crest" src="${escapeHtml(badge.crest_url)}" alt="" loading="lazy" /></span
  >`;
}

function bindCrestFallbacks(root = document) {
  root.querySelectorAll("img.crest").forEach((img) => {
    const hideFallback = () => {
      const fallback = img.parentElement?.querySelector(".crest-fallback");
      if (fallback) fallback.hidden = true;
    };
    img.addEventListener("error", () => img.remove(), { once: true });
    img.addEventListener("load", hideFallback);
    if (img.complete && img.naturalWidth) hideFallback();
    else if (img.complete) img.remove();
  });
}

/* ---------------------------------------------- prediction strip (core) -- */

/* The signature MatchLab component: three equal columns of team + percentage
   above one segmented bar. Same markup at every size so the meaning of each
   colour is identical across the whole product. */
function probStrip(match, { large = false, showFlags = true } = {}) {
  const predicted = match.predicted_outcome;
  const actual = match.actual_outcome ?? null;

  const cols = [
    { key: "home", label: match.home_team, value: match.p_home_pct, name: match.home_team },
    { key: "draw", label: "Draw", value: match.p_draw_pct, name: "Draw" },
    { key: "away", label: match.away_team, value: match.p_away_pct, name: match.away_team },
  ];

  const head = cols
    .map((col) => {
      const isPick = predicted === col.name;
      const isActual = actual != null && actual === col.name;
      const classes = [
        "prob-col",
        `is-${col.key}`,
        isPick ? "is-pick" : "",
        isActual ? "is-actual" : "",
      ]
        .filter(Boolean)
        .join(" ");
      let flag = "";
      if (showFlags && isPick && isActual) flag = `<span class="prob-flag">Pick · Actual</span>`;
      else if (showFlags && isActual) flag = `<span class="prob-flag">Actual</span>`;
      else if (showFlags && isPick) flag = `<span class="prob-flag">Pick</span>`;
      return html`
        <div class="${classes}">
          <span class="prob-team">${escapeHtml(col.label)}</span>
          <span class="prob-pct">${col.value}%</span>
          ${flag}
        </div>
      `;
    })
    .join("");

  const seg = (key, value, name) =>
    `<span class="seg ${key} ${predicted === name ? "is-pick" : ""}" style="width:${value}%"></span>`;

  const aria = `Home ${match.p_home_pct} percent, draw ${match.p_draw_pct} percent, away ${match.p_away_pct} percent`;

  return html`
    <div class="probstrip ${large ? "probstrip-lg" : ""}">
      <div class="probstrip-head">${head}</div>
      <div class="probbar" role="img" aria-label="${escapeHtml(aria)}">
        ${seg("home", match.p_home_pct, match.home_team)}${seg("draw", match.p_draw_pct, "Draw")}${seg(
          "away",
          match.p_away_pct,
          match.away_team,
        )}
      </div>
    </div>
  `;
}

function pickDotClass(match) {
  if (match.predicted_outcome === "Draw") return "draw";
  if (match.predicted_outcome === match.home_team) return "home";
  return "away";
}

/* ---------------------------------------------------------- match cards -- */

function noteTag(note) {
  if (!note) return "";
  const cls = note === "Draw watch" ? "tag-watch" : "tag-close";
  return `<span class="tag ${cls}">${escapeHtml(note)}</span>`;
}

function verdictBadge(correct) {
  if (correct == null) return "";
  return correct
    ? `<span class="verdict verdict-ok"><span class="mark" aria-hidden="true">✓</span>Correct</span>`
    : `<span class="verdict verdict-miss"><span class="mark" aria-hidden="true">✕</span>Missed</span>`;
}

function matchCard(item, { settled = false, showDate = false } = {}) {
  const href = item.match_id ? `${settled ? "/results" : "/upcoming"}/${item.match_id}` : "";
  const link = href
    ? `<a class="stretch-link" data-link href="${href}" aria-label="${escapeHtml(
        `${item.home_team} versus ${item.away_team}, match detail`,
      )}"></a>`
    : "";

  const timeText = showDate
    ? formatKickoff(item.kickoff_date, item.kickoff_time)
    : item.kickoff_time || formatShortDate(item.kickoff_date);

  const center = settled
    ? `<span class="match-score">${item.home_goals}<span class="muted">–</span>${item.away_goals}</span>`
    : `<span class="match-vs">vs</span>`;

  const topRight = settled ? verdictBadge(item.correct) : noteTag(item.note);

  const body = item.has_prediction
    ? probStrip(item, { showFlags: settled })
    : `<p class="no-pred">No stored pre-match prediction for this fixture.</p>`;

  const foot = item.has_prediction
    ? html`
        <div class="match-card-foot">
          <span class="pick-line">
            <i class="pick-dot ${pickDotClass(item)}" aria-hidden="true"></i>
            Pick <b>${escapeHtml(item.predicted_outcome)}</b>
          </span>
          ${settled && item.actual_outcome
            ? `<span>Result <b style="color:var(--ink)">${escapeHtml(item.actual_outcome)}</b></span>`
            : ""}
        </div>
      `
    : "";

  return html`
    <article class="match-card ${href ? "is-link" : ""}">
      ${link}
      <div class="match-card-top">
        <time class="match-time" datetime="${escapeHtml(item.kickoff_date)}">${escapeHtml(timeText)}</time>
        ${topRight}
      </div>
      <div class="match-teams">
        <div class="team-line home">
          ${crestHtml(item.home_team, item.home_crest, "crest-sm")}
          <span class="team-name">${escapeHtml(item.home_team)}</span>
        </div>
        <div class="match-center">${center}</div>
        <div class="team-line away">
          ${crestHtml(item.away_team, item.away_crest, "crest-sm")}
          <span class="team-name">${escapeHtml(item.away_team)}</span>
        </div>
      </div>
      ${body} ${foot}
    </article>
  `;
}

function renderGroups(groups, { settled = false } = {}) {
  if (!groups.length) {
    return `<p class="empty">No matches match these filters. Try clearing them.</p>`;
  }
  return groups
    .map(
      (group) => html`
        <section class="daygroup">
          <div class="day-head"><h2>${formatDay(group.date)}</h2></div>
          <div class="match-grid">
            ${group.fixtures.map((item) => matchCard(item, { settled })).join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

/* ------------------------------------------------------ filters + pager -- */

function filters(values) {
  return html`
    <form class="filters" id="filter-form">
      <label>
        Team
        <select name="team" id="team-filter">
          <option value="">All teams</option>
        </select>
      </label>
      <label>
        From
        <input name="date_from" id="date-from" type="date" value="${escapeHtml(values.date_from || "")}" />
      </label>
      <label>
        To
        <input name="date_to" id="date-to" type="date" value="${escapeHtml(values.date_to || "")}" />
      </label>
      <div class="filter-actions">
        <button type="submit" class="btn btn-primary">Apply</button>
        <button type="button" class="btn btn-ghost" id="clear-filters">Clear</button>
      </div>
    </form>
  `;
}

function fillTeams(teams, selected) {
  const select = document.getElementById("team-filter");
  if (!select) return;
  for (const team of teams) {
    const option = document.createElement("option");
    option.value = team;
    option.textContent = team;
    if (team === selected) option.selected = true;
    select.append(option);
  }
}

function bindFilters(basePath) {
  const form = document.getElementById("filter-form");
  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);
    go(
      `${basePath}${qs({
        team: data.get("team"),
        date_from: data.get("date_from"),
        date_to: data.get("date_to"),
        page: "1",
      })}`,
    );
  });
  document.getElementById("clear-filters")?.addEventListener("click", () => go(basePath));
}

function pager(data, buildHref) {
  if (data.pages <= 1) {
    return `<div class="pager"><span>${data.total} ${data.total === 1 ? "match" : "matches"}</span></div>`;
  }
  const prev =
    data.page > 1 ? `<a class="btn" data-link href="${buildHref(data.page - 1)}">← Previous</a>` : "";
  const next =
    data.page < data.pages ? `<a class="btn" data-link href="${buildHref(data.page + 1)}">Next →</a>` : "";
  return html`
    <div class="pager">
      <span>Page ${data.page} of ${data.pages} · ${data.total} matches</span>
      <div class="pager-actions">${prev}${next}</div>
    </div>
  `;
}

/* -------------------------------------------------------------- overview -- */

function accuracyMeter(live) {
  const value = live.n ? Math.round(live.accuracy * 100) : 0;
  return html`
    <div class="meter">
      <div class="meter-ring" style="--pct:${value}" data-label="${live.n ? `${value}%` : "—"}"></div>
      <div class="meter-body">
        <p class="kpi-label">Live accuracy</p>
        <p class="help">
          ${live.n
            ? `${live.correct} of ${live.n} settled predictions correct · log loss ${live.log_loss.toFixed(3)}`
            : "No settled live predictions yet this season."}
        </p>
      </div>
    </div>
  `;
}

function titleRacePanel(forecast) {
  if (!forecast) return "";
  const top = (forecast.teams || []).slice(0, 5);
  if (!top.length) return "";
  const max = Math.max(...top.map((row) => row.title_prob), 0.0001);
  const rows = top
    .map(
      (row, index) => html`
        <div class="race-row">
          <span class="race-fill" style="--w:${Math.round((row.title_prob / max) * 100)}%"></span>
          <span class="race-pos">${index + 1}</span>
          ${crestHtml(row.team, row.crest, "crest-xs")}
          <span class="race-name">${escapeHtml(row.team)}</span>
          <span class="race-pct">${fmtPct(row.title_prob)}</span>
        </div>
      `,
    )
    .join("");
  return html`
    <article class="panel">
      <div class="section-head" style="margin-bottom:var(--sp-3)">
        <h2>Title race</h2>
        <a class="pill" data-link href="/forecast">Full forecast</a>
      </div>
      <p class="help" style="margin-bottom:var(--sp-3)">
        Chance of finishing first across ${forecast.n_sims.toLocaleString("en-GB")} simulated seasons.
      </p>
      <div class="race-list">${rows}</div>
    </article>
  `;
}

async function renderOverview() {
  const data = await fetchJson("/api/overview");
  /* The forecast is a separate read-only endpoint. It can legitimately be
     unavailable (503) before the first pipeline run, so never block on it. */
  const forecast = await fetchJson("/api/forecast").catch(() => null);

  setSeason(`Premier League · ${data.live_season}`);
  setFootMeta(data.model, `Predictions updated ${formatStamp(data.predictions_updated_at)}`);

  const live = data.live_scorecard;
  const latest = data.latest_completed_match;

  view.innerHTML = html`
    <div class="page-head">
      <div class="page-head-main">
        <p class="eyebrow">Season ${escapeHtml(data.live_season)}</p>
        <h1>Premier League predictions</h1>
        <p class="lede">
          Home, Draw and Away probabilities for every remaining fixture, frozen the moment a match
          kicks off so results are always judged against what was predicted beforehand.
        </p>
      </div>
    </div>

    <section class="kpi-grid">
      <article class="kpi kpi-accent">
        <span class="kpi-label">Live accuracy</span>
        <span class="kpi-value">${live.n ? pct(live.accuracy) : "—"}</span>
        <span class="kpi-foot">${live.n ? `${live.correct} of ${live.n} correct` : "Awaiting results"}</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Log loss</span>
        <span class="kpi-value">${live.n ? live.log_loss.toFixed(3) : "—"}</span>
        <span class="kpi-foot">Lower is better</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Fixtures ahead</span>
        <span class="kpi-value">${data.n_upcoming}</span>
        <span class="kpi-foot">All carry a stored prediction</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Matches settled</span>
        <span class="kpi-value">${data.n_settled}</span>
        <span class="kpi-foot">Played so far in ${escapeHtml(data.live_season)}</span>
      </article>
    </section>

    <div class="overview-cols">
      <div>
        <div class="section-head">
          <h2>Next fixtures</h2>
          <a class="pill" data-link href="/upcoming">All fixtures →</a>
        </div>
        <div class="match-grid">
          ${data.next_upcoming.length
            ? data.next_upcoming.slice(0, 6).map((item) => matchCard(item, { showDate: true })).join("")
            : `<p class="empty">No upcoming fixtures with a stored prediction.</p>`}
        </div>
      </div>

      <div class="side-stack">
        <article class="panel">${accuracyMeter(live)}</article>

        <article class="panel">
          <div class="section-head" style="margin-bottom:var(--sp-3)">
            <h2>Latest result</h2>
            <a class="pill" data-link href="/results">All results</a>
          </div>
          ${latest
            ? html`
                <div class="latest-result">
                  <div class="latest-teams">
                    <div class="latest-club">
                      ${crestHtml(latest.home_team, latest.home_crest, "crest-md")}
                      <span>${escapeHtml(latest.home_team)}</span>
                    </div>
                    <span class="latest-score">${latest.home_goals}–${latest.away_goals}</span>
                    <div class="latest-club">
                      ${crestHtml(latest.away_team, latest.away_crest, "crest-md")}
                      <span>${escapeHtml(latest.away_team)}</span>
                    </div>
                  </div>
                  <p class="help" style="text-align:center">
                    ${formatKickoff(latest.kickoff_date, latest.kickoff_time)}
                  </p>
                </div>
              `
            : `<p class="help">No completed matches yet this season.</p>`}
        </article>

        ${titleRacePanel(forecast)}
      </div>
    </div>
  `;
  bindCrestFallbacks(view);
}

/* ------------------------------------------------------------- fixtures -- */

function listQuery() {
  const params = new URLSearchParams(window.location.search);
  return {
    team: params.get("team") || "",
    date_from: params.get("date_from") || "",
    date_to: params.get("date_to") || "",
    page: params.get("page") || "1",
    page_size: String(PAGE_SIZE),
  };
}

async function renderUpcoming() {
  const query = listQuery();
  const data = await fetchJson(`/api/upcoming${qs(query)}`);
  setSeason(`Premier League · ${data.live_season}`);
  setFootMeta(data.model);

  const hrefFor = (page) =>
    `/upcoming${qs({ team: query.team, date_from: query.date_from, date_to: query.date_to, page })}`;

  view.innerHTML = html`
    <div class="page-head">
      <div class="page-head-main">
        <p class="eyebrow">Season ${escapeHtml(data.live_season)}</p>
        <h1>Fixtures</h1>
        <p class="lede">
          Every unplayed match with its pre-match probabilities. ${data.total} fixtures remaining.
        </p>
      </div>
    </div>
    ${filters(query)} ${renderGroups(data.groups)} ${pager(data, hrefFor)}
  `;
  fillTeams(data.teams, query.team);
  bindFilters("/upcoming");
  bindCrestFallbacks(view);
}

async function renderResults() {
  const query = listQuery();
  const data = await fetchJson(`/api/results${qs(query)}`);
  setSeason(`Premier League · ${data.live_season}`);
  setFootMeta(data.model);

  const hrefFor = (page) =>
    `/results${qs({ team: query.team, date_from: query.date_from, date_to: query.date_to, page })}`;

  view.innerHTML = html`
    <div class="page-head">
      <div class="page-head-main">
        <p class="eyebrow">Season ${escapeHtml(data.live_season)}</p>
        <h1>Results</h1>
        <p class="lede">
          Completed matches shown against the probabilities stored before kickoff. Those numbers are
          never rewritten once a result is known.
        </p>
      </div>
    </div>
    ${filters(query)} ${renderGroups(data.groups, { settled: true })} ${pager(data, hrefFor)}
  `;
  fillTeams(data.teams, query.team);
  bindFilters("/results");
  bindCrestFallbacks(view);
}

/* ---------------------------------------------------------- performance -- */

/* The bar encodes the success RATE, and the sample size is shown as text.
   The previous version sized the bar by sample size, which made a 0% draw
   record render as an almost full bar. */
function rateRows(slice, emptyLabel) {
  const names = [
    ["home", "Home"],
    ["draw", "Draw"],
    ["away", "Away"],
  ];
  return names
    .map(([key, label]) => {
      const row = slice[key] || { n: 0, correct: 0, rate: null };
      const width = row.n ? Math.round((row.rate || 0) * 100) : 0;
      return html`
        <div class="rate-row">
          <span class="rate-name">${label}</span>
          <div class="rate-track"><span class="${key}" style="width:${width}%"></span></div>
          <span class="rate-value">
            ${row.n ? pct(row.rate) : "—"}<small>${row.n ? `${row.correct}/${row.n}` : emptyLabel}</small>
          </span>
        </div>
      `;
    })
    .join("");
}

async function renderPerformance() {
  const data = await fetchJson("/api/performance");
  const recent = await fetchJson(`/api/results${qs({ page: "1", page_size: "20" })}`).catch(() => null);
  const live = data.live_scorecard;
  setSeason(`Premier League · ${data.live_season}`);
  setFootMeta(data.model);

  const settled = recent
    ? recent.groups.flatMap((group) => group.fixtures).filter((item) => item.has_prediction)
    : [];

  const streak = settled
    .slice(0, 20)
    .map(
      (item) => html`
        <a
          class="streak-chip ${item.correct ? "ok" : "miss"}"
          data-link
          href="/results/${item.match_id}"
          title="${escapeHtml(
            `${item.home_team} ${item.home_goals}–${item.away_goals} ${item.away_team} · picked ${item.predicted_outcome}`,
          )}"
          >${item.correct ? "✓" : "✕"}</a
        >
      `,
    )
    .join("");

  view.innerHTML = html`
    <div class="page-head">
      <div class="page-head-main">
        <p class="eyebrow">Season ${escapeHtml(data.live_season)}</p>
        <h1>Performance</h1>
        <p class="lede">
          How the frozen pre-match predictions have actually performed this season. Walk-forward
          selection and the untouched holdout test are on the model page.
        </p>
      </div>
      <a class="pill" data-link href="/about">How the model was chosen →</a>
    </div>

    <section class="kpi-grid">
      <article class="kpi kpi-accent">
        <span class="kpi-label">Accuracy</span>
        <span class="kpi-value">${live.n ? pct(live.accuracy) : "—"}</span>
        <span class="kpi-foot">Top pick matched the result</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Log loss</span>
        <span class="kpi-value">${live.n ? live.log_loss.toFixed(3) : "—"}</span>
        <span class="kpi-foot">Scores the probability, not the pick</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Matches judged</span>
        <span class="kpi-value">${live.n}</span>
        <span class="kpi-foot">Settled with a stored prediction</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Correct picks</span>
        <span class="kpi-value">${live.correct || 0}</span>
        <span class="kpi-foot">Out of ${live.n || 0}</span>
      </article>
    </section>

    ${settled.length
      ? html`
          <article class="panel section">
            <h2>Recent form</h2>
            <p class="help">Most recent settled matches, newest first. Select one to open it.</p>
            <div class="streak" style="margin-top:var(--sp-4)">${streak}</div>
          </article>
        `
      : ""}

    <div class="detail-cols">
      <article class="panel">
        <h2>Accuracy by actual result</h2>
        <p class="help">
          When a match really ended Home, Draw or Away, how often the top pick had said so.
        </p>
        ${live.n
          ? `<div class="rate-list" style="margin-top:var(--sp-4)">${rateRows(live.by_actual, "none yet")}</div>`
          : `<p class="empty">No settled live predictions yet.</p>`}
      </article>
      <article class="panel">
        <h2>Accuracy by predicted class</h2>
        <p class="help">
          When the model's most likely outcome was this class, how often it turned out right.
        </p>
        ${live.n
          ? `<div class="rate-list" style="margin-top:var(--sp-4)">${rateRows(
              live.by_predicted,
              "never picked",
            )}</div>`
          : `<p class="empty">No settled live predictions yet.</p>`}
      </article>
    </div>
  `;
  bindCrestFallbacks(view);
}

/* ------------------------------------------------------------- forecast -- */

let forecastState = { rows: [], sort: null, dir: "desc" };

const FORECAST_COLUMNS = [
  { key: "pos", label: "#", sortable: false },
  { key: "team", label: "Club", sortable: true, type: "text" },
  { key: "current_played", label: "Pld", sortable: true },
  { key: "current_points", label: "Pts", sortable: true },
  { key: "title_prob", label: "Title", sortable: true },
  { key: "top4_prob", label: "Top 4", sortable: true },
  { key: "relegation_prob", label: "Relegation", sortable: true },
  { key: "expected_points", label: "Exp. pts", sortable: true },
  { key: "expected_position", label: "Exp. pos", sortable: true },
];

function forecastRow(row, index) {
  const zone =
    row.title_prob >= 0.05
      ? "zone-title"
      : row.top4_prob >= 0.5
        ? "zone-ucl"
        : row.relegation_prob >= 0.5
          ? "zone-rel"
          : "";
  const bar = (kind, value) => html`
    <span class="databar ${kind} ${value <= 0 ? "is-zero" : ""}">
      <span class="fill" style="--w:${Math.min(100, Math.round(value * 100))}%"></span>
      <span class="v">${fmtPct(value)}</span>
    </span>
  `;
  return html`
    <tr class="${zone}">
      <td class="col-pos">${index + 1}</td>
      <td class="col-club">
        <span class="club-cell">
          ${crestHtml(row.team, row.crest, "crest-xs")}
          <span class="name">${escapeHtml(row.team)}</span>
        </span>
      </td>
      <td class="num-soft">${row.current_played}</td>
      <td class="num">${row.current_points}</td>
      <td>${bar("title", row.title_prob)}</td>
      <td>${bar("top4", row.top4_prob)}</td>
      <td>${bar("rel", row.relegation_prob)}</td>
      <td class="num">${fmtNum(row.expected_points)}</td>
      <td class="num-soft">${fmtNum(row.expected_position)}</td>
    </tr>
  `;
}

function forecastTableBody() {
  const rows = [...forecastState.rows];
  const { sort, dir } = forecastState;
  if (sort) {
    const column = FORECAST_COLUMNS.find((col) => col.key === sort);
    rows.sort((a, b) => {
      if (column?.type === "text") {
        return dir === "asc" ? a[sort].localeCompare(b[sort]) : b[sort].localeCompare(a[sort]);
      }
      return dir === "asc" ? a[sort] - b[sort] : b[sort] - a[sort];
    });
  }
  return rows.map(forecastRow).join("");
}

function paintForecastTable() {
  const body = document.getElementById("forecast-body");
  if (body) {
    body.innerHTML = forecastTableBody();
    bindCrestFallbacks(body);
  }
  document.querySelectorAll("#forecast-table th[data-sort]").forEach((th) => {
    const active = th.dataset.sort === forecastState.sort;
    th.setAttribute("aria-sort", active ? (forecastState.dir === "asc" ? "ascending" : "descending") : "none");
    const arrow = th.querySelector(".arrow");
    if (arrow) arrow.textContent = active ? (forecastState.dir === "asc" ? "▲" : "▼") : "";
  });
}

function bindForecastSort() {
  document.querySelectorAll("#forecast-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (forecastState.sort === key) {
        forecastState.dir = forecastState.dir === "desc" ? "asc" : "desc";
      } else {
        forecastState.sort = key;
        forecastState.dir = key === "team" ? "asc" : "desc";
      }
      paintForecastTable();
    });
    th.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        th.click();
      }
    });
  });
}

async function renderForecast() {
  const data = await fetchJson("/api/forecast");
  setSeason(`Premier League · ${data.live_season}`);
  setFootMeta(data.model, `Forecast generated ${formatStamp(data.generated_at)}`);

  forecastState = { rows: data.teams || [], sort: null, dir: "desc" };

  const podium = (data.teams || [])
    .slice(0, 3)
    .map(
      (row, index) => html`
        <article class="podium-card ${index === 0 ? "is-first" : ""}">
          ${crestHtml(row.team, row.crest, "crest-md")}
          <div class="podium-body">
            <span class="podium-label">${index === 0 ? "Favourite" : `${index + 1}${index === 1 ? "nd" : "rd"} most likely`}</span>
            <span class="podium-name">${escapeHtml(row.team)}</span>
            <span class="podium-pct">${fmtPct(row.title_prob)}</span>
          </div>
        </article>
      `,
    )
    .join("");

  const headCells = FORECAST_COLUMNS.map((col) => {
    if (!col.sortable) return `<th scope="col">${col.label}</th>`;
    return `<th scope="col" data-sort="${col.key}" tabindex="0" role="columnheader" aria-sort="none">${col.label} <span class="arrow"></span></th>`;
  }).join("");

  view.innerHTML = html`
    <div class="page-head">
      <div class="page-head-main">
        <p class="eyebrow">Season ${escapeHtml(data.live_season)}</p>
        <h1>Season forecast</h1>
        <p class="lede">
          Every remaining fixture is replayed ${data.n_sims.toLocaleString("en-GB")} times using the
          stored Home / Draw / Away probabilities, then the final table is counted up.
        </p>
      </div>
    </div>

    <section class="race-podium">${podium}</section>

    <section class="kpi-grid">
      <article class="kpi">
        <span class="kpi-label">Simulations</span>
        <span class="kpi-value">${data.n_sims.toLocaleString("en-GB")}</span>
        <span class="kpi-foot">Seed ${data.seed} · reproducible</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Matches played</span>
        <span class="kpi-value">${data.n_completed}</span>
        <span class="kpi-foot">Points already on the board</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Still to play</span>
        <span class="kpi-value">${data.n_remaining}</span>
        <span class="kpi-foot">Sampled in every simulation</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Forecast generated</span>
        <span class="kpi-value sm">${formatStamp(data.generated_at)}</span>
        <span class="kpi-foot">Rebuilt by the pipeline</span>
      </article>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Projected table</h2>
        <span class="help">Select a column heading to sort</span>
      </div>
      <div class="table-scroll">
        <table class="standings" id="forecast-table">
          <thead>
            <tr>${headCells}</tr>
          </thead>
          <tbody id="forecast-body"></tbody>
        </table>
      </div>
      <div class="legend">
        <span><i style="background:var(--accent)"></i> Title contender</span>
        <span><i style="background:var(--away)"></i> Likely top four</span>
        <span><i style="background:var(--miss)"></i> Likely relegation</span>
        <span>Rows are ordered by title probability until you sort.</span>
      </div>
    </section>

    <div class="detail-cols">
      <article class="panel">
        <h2>How ties are broken</h2>
        <p class="help">${escapeHtml(data.tiebreak)}</p>
      </article>
      <article class="panel">
        <h2>Early-season caution</h2>
        <p class="help">${escapeHtml(data.early_season_note || "")}</p>
      </article>
    </div>
  `;

  paintForecastTable();
  bindForecastSort();
  bindCrestFallbacks(view);
}

/* --------------------------------------------------------- match detail -- */

function heroSide(name, crest, positionLabel) {
  return html`
    <div class="hero-side">
      ${crestHtml(name, crest, "crest-lg")}
      <p class="hero-team">${escapeHtml(name)}</p>
      ${positionLabel ? `<span class="rank-pill">${escapeHtml(positionLabel)}</span>` : ""}
    </div>
  `;
}

function matchHero(match, { settled }) {
  const center = settled
    ? html`
        <span class="hero-score">${match.home_goals}–${match.away_goals}</span>
        <span class="hero-sub">${escapeHtml(match.actual_outcome || "Full time")}</span>
      `
    : html`
        <span class="hero-kick">${escapeHtml(match.kickoff_time || "TBC")}</span>
        <span class="hero-sub">${escapeHtml(formatShortDate(match.kickoff_date))}</span>
      `;

  const statusTag = settled
    ? `<span class="tag">${escapeHtml(match.status || "Full time")}</span>`
    : `<span class="tag tag-live">${escapeHtml(match.status || "Upcoming")}</span>`;

  const prediction = match.has_prediction
    ? html`
        <div class="hero-prediction">
          <div class="hero-pred-head">
            <h2>${settled ? "Pre-match prediction" : "Win probability"}</h2>
            <div style="display:flex;gap:var(--sp-2);align-items:center;flex-wrap:wrap">
              ${settled ? verdictBadge(match.correct) : noteTag(match.note)}
              <span class="tag">${escapeHtml(match.model_name || "Model")}</span>
            </div>
          </div>
          ${probStrip(match, { large: true, showFlags: true })}
        </div>
      `
    : html`
        <div class="hero-prediction">
          <p class="no-pred">
            No stored pre-match prediction for this match. MatchLab only shows probabilities that
            were recorded before kickoff.
          </p>
        </div>
      `;

  return html`
    <article class="match-hero">
      <div class="hero-meta">
        <span>${escapeHtml(match.competition || "Premier League")} · ${formatKickoff(match.kickoff_date, match.kickoff_time)}</span>
        ${statusTag}
      </div>
      <div class="hero-board">
        ${heroSide(match.home_team, match.home_crest, match.home_position_label)}
        <div class="hero-center">${center}</div>
        ${heroSide(match.away_team, match.away_crest, match.away_position_label)}
      </div>
      ${prediction}
    </article>
  `;
}

function h2hRow(meeting) {
  const href = meeting.detail_path || `/results/${meeting.match_id}`;
  return html`
    <a class="h2h-row" data-link href="${href}">
      <span class="h2h-date">${formatShortDate(meeting.kickoff_date)}</span>
      <span class="h2h-teams">
        ${crestHtml(meeting.home_team, meeting.home_crest, "crest-xs")}
        <span class="nm">${escapeHtml(meeting.home_team)}</span>
        <span class="sc">${meeting.home_goals}–${meeting.away_goals}</span>
        <span class="nm">${escapeHtml(meeting.away_team)}</span>
        ${crestHtml(meeting.away_team, meeting.away_crest, "crest-xs")}
      </span>
      <span class="h2h-outcome">${escapeHtml(meeting.result_label || "")}</span>
    </a>
  `;
}

function h2hPanel(data, match) {
  const meetings = data.head_to_head || [];
  return html`
    <article class="panel">
      <h2>Previous meetings</h2>
      <p class="help">
        ${meetings.length
          ? `Last ${meetings.length} completed meeting${meetings.length === 1 ? "" : "s"} between these clubs before this fixture.`
          : "No previous meetings in the ingested history."}
      </p>
      ${meetings.length
        ? `<div class="h2h-list" style="margin-top:var(--sp-4)">${meetings.map(h2hRow).join("")}</div>`
        : ""}
    </article>
  `;
}

function statsPanel(match, stats, available) {
  if (!available || !stats.length) {
    return html`
      <article class="panel">
        <h2>Match stats</h2>
        <p class="help">Detailed stats are not available for this match in the ingested source.</p>
      </article>
    `;
  }
  const rows = stats
    .map((stat) => {
      const total = (Number(stat.home) || 0) + (Number(stat.away) || 0);
      const homeShare = total > 0 ? (Number(stat.home) / total) * 100 : 50;
      const awayShare = total > 0 ? 100 - homeShare : 50;
      const flat = total === 0;
      return html`
        <div class="statrow">
          <div class="statrow-top">
            <span class="n ${stat.leader === "home" ? "lead" : ""}">${stat.home}</span>
            <span class="label">${escapeHtml(stat.label)}</span>
            <span class="n away ${stat.leader === "away" ? "lead" : ""}">${stat.away}</span>
          </div>
          <div class="stat-track">
            <i class="${flat ? "flat" : "home"}" style="width:${flat ? 50 : homeShare}%"></i>
            <i class="${flat ? "flat" : "away"}" style="width:${flat ? 50 : awayShare}%"></i>
          </div>
        </div>
      `;
    })
    .join("");

  return html`
    <article class="panel">
      <h2>Match stats</h2>
      <div class="stat-legend">
        <span class="side">${crestHtml(match.home_team, match.home_crest, "crest-xs")}<b>${escapeHtml(match.home_team)}</b></span>
        <span class="side"><b>${escapeHtml(match.away_team)}</b>${crestHtml(match.away_team, match.away_crest, "crest-xs")}</span>
      </div>
      <div class="statlist">${rows}</div>
    </article>
  `;
}

async function renderMatchDetail(matchId, requestedKind) {
  const data = await fetchJson(`/api/matches/${matchId}`);
  const canonical = data.kind === "result" ? `/results/${matchId}` : `/upcoming/${matchId}`;
  if (requestedKind && requestedKind !== data.kind) {
    window.history.replaceState({}, "", canonical);
    setNav();
  }

  const match = data.match;
  const settled = data.kind === "result";
  setSeason(`Premier League · ${data.live_season}`);
  setFootMeta(data.model);

  const secondary = settled
    ? html`
        <div class="detail-cols">
          ${statsPanel(match, data.stats || [], data.stats_available)}
          <article class="panel">
            <h2>Prediction vs result</h2>
            ${match.has_prediction
              ? html`
                  <p class="help">How the stored pre-match call compared with what happened.</p>
                  <div class="metric-grid" style="margin-top:var(--sp-4)">
                    <div class="metric"><span>Picked</span><strong style="font-family:var(--font-display);font-size:1rem">${escapeHtml(match.predicted_outcome)}</strong></div>
                    <div class="metric"><span>Actual</span><strong style="font-family:var(--font-display);font-size:1rem">${escapeHtml(match.actual_outcome || "—")}</strong></div>
                    <div class="metric"><span>Chance given</span><strong>${
                      match.actual_outcome === match.home_team
                        ? match.p_home_pct
                        : match.actual_outcome === "Draw"
                          ? match.p_draw_pct
                          : match.p_away_pct
                    }%</strong></div>
                  </div>
                  <p class="help" style="margin-top:var(--sp-4)">
                    Probabilities were frozen at ${formatStamp(match.predicted_at)} and have not been
                    changed since the result arrived.
                  </p>
                `
              : `<p class="help">This match predates the live season, so no pre-match prediction was stored for it. MatchLab never back-fills a prediction after the fact.</p>`}
          </article>
        </div>
      `
    : html`<div class="detail-cols">${h2hPanel(data, match)}</div>`;

  view.innerHTML = html`
    <p class="back-row">
      <a class="pill" data-link href="${settled ? "/results" : "/upcoming"}">
        ← Back to ${settled ? "results" : "fixtures"}
      </a>
    </p>
    ${matchHero(match, { settled })} ${secondary}
  `;
  bindCrestFallbacks(view);
}

/* ---------------------------------------------------------------- about -- */

function metric(label, value) {
  return html`<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

async function renderAbout() {
  const data = await fetchJson("/api/about");
  const test = data.test || {};
  setSeason("Premier League");
  setFootMeta(data.model);

  const folds = (data.walkforward_folds || [])
    .map(
      (fold) => html`
        <div class="fold-row">
          <span class="fold-name">${escapeHtml(fold.valid_season)}</span>
          <span class="help">trained on everything through ${escapeHtml(fold.train_through)}</span>
        </div>
      `,
    )
    .join("");

  view.innerHTML = html`
    <div class="page-head">
      <div class="page-head-main">
        <p class="eyebrow">${escapeHtml(data.model.model_name)} · ${escapeHtml(data.feature_version)}</p>
        <h1>The model</h1>
        <p class="lede">
          Every probability on this site comes from one logistic regression, chosen by out-of-time
          walk-forward validation and never trained on a match it is asked to predict.
        </p>
      </div>
    </div>

    <section class="kpi-grid">
      <article class="kpi kpi-accent">
        <span class="kpi-label">Walk-forward log loss</span>
        <span class="kpi-value">${data.walkforward_mean_log_loss != null ? Number(data.walkforward_mean_log_loss).toFixed(3) : "—"}</span>
        <span class="kpi-foot">Mean across ${(data.walkforward_folds || []).length} out-of-time seasons</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Walk-forward accuracy</span>
        <span class="kpi-value">${data.walkforward_mean_accuracy != null ? pct(data.walkforward_mean_accuracy) : "—"}</span>
        <span class="kpi-foot">Diagnostic only, not the selection metric</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Holdout accuracy</span>
        <span class="kpi-value">${test.accuracy != null ? pct(test.accuracy) : "—"}</span>
        <span class="kpi-foot">${escapeHtml(data.test_season)} · scored once</span>
      </article>
      <article class="kpi">
        <span class="kpi-label">Holdout matches</span>
        <span class="kpi-value">${test.n ?? "—"}</span>
        <span class="kpi-foot">Never used to choose the model</span>
      </article>
    </section>

    <div class="about-grid section">
      <article class="panel">
        <h2>Why this model</h2>
        <p class="help">${escapeHtml(data.selection_reason || "")}</p>
        <div class="fold-list" style="margin-top:var(--sp-4)">${folds}</div>
      </article>

      <article class="panel">
        <h2>How the seasons are used</h2>
        <ul class="plain-list">
          <li>${escapeHtml(data.history.ingest)}</li>
          <li>${escapeHtml(data.history.walkforward)}</li>
          <li>${escapeHtml(data.history.production_train)}</li>
          <li>${escapeHtml(data.history.test)}</li>
          <li>${escapeHtml(data.history.live)}</li>
        </ul>
      </article>

      <article class="panel">
        <h2>Untouched ${escapeHtml(data.test_season)} holdout</h2>
        <p class="help">Scored once, after the model was chosen.</p>
        <div class="metric-grid" style="margin-top:var(--sp-4)">
          ${metric("Accuracy", test.accuracy != null ? pct(test.accuracy) : "—")}
          ${metric("Log loss", test.log_loss != null ? Number(test.log_loss).toFixed(3) : "—")}
          ${metric("Matches", test.n ?? "—")}
          ${metric("Home F1", test.f1_home != null ? Number(test.f1_home).toFixed(3) : "—")}
          ${metric("Draw F1", test.f1_draw != null ? Number(test.f1_draw).toFixed(3) : "—")}
          ${metric("Away F1", test.f1_away != null ? Number(test.f1_away).toFixed(3) : "—")}
        </div>
      </article>

      <article class="panel">
        <h2>What the model looks at</h2>
        <ul class="feature-list">
          ${data.features
            .map((item) => `<li><b>${escapeHtml(item.title)}.</b> ${escapeHtml(item.detail)}</li>`)
            .join("")}
        </ul>
        <p class="help" style="margin-top:var(--sp-4)">${escapeHtml(data.leakage_note)}</p>
      </article>

      <article class="panel span-all">
        <h2>A known limitation: draws</h2>
        <p class="help">${escapeHtml(data.draw_limitation)}</p>
      </article>
    </div>
  `;
}

/* --------------------------------------------------------------- router -- */

const routes = {
  "/": renderOverview,
  "/upcoming": renderUpcoming,
  "/results": renderResults,
  "/forecast": renderForecast,
  "/performance": renderPerformance,
  "/about": renderAbout,
};

async function render() {
  setNav();
  const detail = matchDetailRoute();
  try {
    if (detail) {
      await renderMatchDetail(detail.id, detail.kind);
      return;
    }
    const route = routes[pathOf()] || renderOverview;
    await route();
  } catch (error) {
    view.innerHTML = html`
      <div class="page-head">
        <div class="page-head-main">
          <h1>Something went wrong</h1>
          <p class="lede">${escapeHtml(error.message || "Could not load this page from the database.")}</p>
        </div>
      </div>
      <p class="back-row"><a class="btn btn-primary" data-link href="/">Back to overview</a></p>
    `;
  }
}

function go(href) {
  const url = new URL(href, window.location.origin);
  window.history.pushState({}, "", `${url.pathname}${url.search}`);
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  render();
}

document.addEventListener("click", (event) => {
  const link = event.target.closest("[data-link]");
  if (!link) return;
  const url = new URL(link.href, window.location.origin);
  if (url.origin !== window.location.origin) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
  event.preventDefault();
  go(`${url.pathname}${url.search}`);
});

window.addEventListener("popstate", render);
render();
