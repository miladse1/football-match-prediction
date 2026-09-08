const view = document.getElementById("view");
const PAGE_SIZE = 16;

function pathOf() {
  const path = window.location.pathname.replace(/\/$/, "");
  return path || "/";
}

function escapeHtml(text) {
  return String(text)
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

function formatKickoff(dateStr, timeStr) {
  const date = new Date(`${dateStr}T${timeStr || "00:00"}:00`);
  const weekday = date.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  return timeStr ? `${weekday} · ${timeStr}` : weekday;
}

function formatDay(dateStr) {
  return new Date(`${dateStr}T12:00:00`).toLocaleDateString("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
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
  const pctValue = value * 100;
  if (pctValue >= 1) return `${pctValue.toFixed(1)}%`;
  return `${pctValue.toFixed(2)}%`;
}

function fmtNum(value, digits = 1) {
  if (value == null) return "—";
  return Number(value).toFixed(digits);
}

function html(strings, ...values) {
  return strings.reduce((out, chunk, i) => out + chunk + (values[i] ?? ""), "");
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
  });
}

function matchDetailRoute() {
  const upcoming = pathOf().match(/^\/upcoming\/(\d+)$/);
  if (upcoming) return { kind: "upcoming", id: Number(upcoming[1]) };
  const result = pathOf().match(/^\/results\/(\d+)$/);
  if (result) return { kind: "result", id: Number(result[1]) };
  return null;
}

function crestHtml(team, crest, extraClass = "") {
  const badge = crest || {};
  const initials = escapeHtml(badge.initials || team.slice(0, 3).toUpperCase());
  const color = escapeHtml(badge.color || "#3d4338");
  const ink = escapeHtml(badge.ink || "#f4f1e8");
  const fallback = `<span class="crest-fallback" style="background:${color};color:${ink}">${initials}</span>`;
  if (!badge.crest_url) {
    return `<span class="crest-wrap ${extraClass}">${fallback}</span>`;
  }
  return html`
    <span class="crest-wrap ${extraClass}">
      ${fallback}
      <img class="crest" src="${escapeHtml(badge.crest_url)}" alt="" data-initials="${initials}" />
    </span>
  `;
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

function probabilityRow(label, pctValue, kind, highlight) {
  const cls = highlight ? "predicted" : "";
  return html`
    <div class="prob-row">
      <span class="${cls}">${escapeHtml(label)}</span>
      <strong class="${cls}">${pctValue}%</strong>
      <div class="bar ${kind}"><span style="width:${pctValue}%"></span></div>
    </div>
  `;
}

function fixtureCard(item, { settled = false, compact = false } = {}) {
  const predicted = item.predicted_outcome;
  const note = item.note
    ? `<span class="note ${item.note === "Draw watch" ? "watch" : ""}">${escapeHtml(item.note)}</span>`
    : "";
  const result = settled
    ? `<span class="pill ${item.correct ? "ok" : "wrong"}">${item.correct ? "Correct" : "Missed"}</span>
       <span class="scoreline">${item.home_goals}–${item.away_goals} · ${escapeHtml(item.actual_outcome)}</span>`
    : `<span class="pill">${escapeHtml(item.model_name)} · ${escapeHtml(item.feature_version)}</span>`;
  const href = item.match_id
    ? `${settled ? "/results" : "/upcoming"}/${item.match_id}`
    : "";
  const link = href
    ? `<a class="fixture-card-link" data-link href="${href}" aria-label="${escapeHtml(item.home_team)} vs ${escapeHtml(item.away_team)} match detail"></a>`
    : "";
  return html`
    <article class="fixture-card ${compact ? "compact-list" : ""} ${href ? "is-link" : ""}">
      ${link}
      <div class="card-top">
        <div>
          <p class="kickoff">${formatKickoff(item.kickoff_date, item.kickoff_time)}</p>
          <h3 class="fixture-title">
            ${crestHtml(item.home_team, item.home_crest, "crest-sm")}
            <span>${escapeHtml(item.home_team)}</span>
            <span class="vs-inline">vs</span>
            ${crestHtml(item.away_team, item.away_crest, "crest-sm")}
            <span>${escapeHtml(item.away_team)}</span>
          </h3>
        </div>
        ${note}
      </div>
      <div class="probs">
        ${probabilityRow(`${item.home_team} win`, item.p_home_pct, "home", predicted === item.home_team)}
        ${probabilityRow("Draw", item.p_draw_pct, "draw", predicted === "Draw")}
        ${probabilityRow(`${item.away_team} win`, item.p_away_pct, "away", predicted === item.away_team)}
      </div>
      <div class="outcome">
        <p class="predicted">Predicted outcome: ${escapeHtml(predicted)}</p>
        <div class="outcome">${result}</div>
      </div>
    </article>
  `;
}

function renderGroups(groups, { settled = false } = {}) {
  if (!groups.length) {
    return `<p class="empty">No matches for these filters.</p>`;
  }
  return groups
    .map(
      (group) => html`
        <section class="group">
          <h2 class="group-title">${formatDay(group.date)}</h2>
          <div class="fixture-list">
            ${group.fixtures.map((item) => fixtureCard(item, { settled })).join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

function pager(data, buildHref) {
  if (data.pages <= 1) {
    return `<p class="pager">${data.total} matches</p>`;
  }
  const prev = data.page > 1 ? `<a class="pill" data-link href="${buildHref(data.page - 1)}">Previous</a>` : "";
  const next =
    data.page < data.pages ? `<a class="pill" data-link href="${buildHref(data.page + 1)}">Next</a>` : "";
  return html`
    <div class="pager">
      <span>Page ${data.page} of ${data.pages} · ${data.total} matches</span>
      <div class="pager-actions">${prev}${next}</div>
    </div>
  `;
}

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
        <input name="date_from" id="date-from" type="date" value="${values.date_from || ""}" />
      </label>
      <label>
        To
        <input name="date_to" id="date-to" type="date" value="${values.date_to || ""}" />
      </label>
      <button type="submit">Apply</button>
      <button type="button" class="ghost" id="clear-filters">Clear</button>
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

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = "Could not load this page from the database.";
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (error) {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}

async function renderOverview() {
  const data = await fetchJson("/api/overview");
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  const live = data.live_scorecard;
  const latest = data.latest_completed_match;
  view.innerHTML = html`
    <div class="page-head">
      <div>
        <p class="eyebrow">Live season ${data.live_season}</p>
        <h1>Overview</h1>
        <p class="lede">Next fixtures, latest result, and live scorecard from frozen pre-match probabilities. The model is not retrained here.</p>
      </div>
      <p class="help">${data.model.model_name} · ${data.model.feature_version}</p>
    </div>
    <section class="stat-grid">
      <article class="stat-card">
        <h2>Predictions updated</h2>
        <p class="stat-value">${formatStamp(data.predictions_updated_at)}</p>
      </article>
      <article class="stat-card">
        <h2>Live accuracy</h2>
        <p class="stat-value">${live.n ? pct(live.accuracy) : "—"}</p>
        <p class="help">${live.n ? `log loss ${live.log_loss.toFixed(3)} · ${live.correct}/${live.n}` : "No settled live predictions yet"}</p>
      </article>
      <article class="stat-card">
        <h2>Upcoming</h2>
        <p class="stat-value">${data.n_upcoming}</p>
        <p class="help">Unplayed fixtures with a stored prediction</p>
      </article>
      <article class="stat-card">
        <h2>Settled</h2>
        <p class="stat-value">${data.n_settled}</p>
        <p class="help">Finished 2026/27 matches with frozen probabilities</p>
      </article>
    </section>
    <div class="split">
      <article class="panel">
        <h2>Latest completed match</h2>
        ${
          latest
            ? `<p class="stat-value">${escapeHtml(latest.home_team)} ${latest.home_goals}–${latest.away_goals} ${escapeHtml(latest.away_team)}</p>
               <p class="help">${formatKickoff(latest.kickoff_date, latest.kickoff_time)}</p>`
            : `<p class="help">None yet.</p>`
        }
      </article>
      <div>
        <div class="page-head">
          <h2 class="group-title" style="margin:0">Next up</h2>
          <a class="pill" data-link href="/upcoming">All upcoming</a>
        </div>
        <div class="fixture-list">
          ${
            data.next_upcoming.length
              ? data.next_upcoming.map((item) => fixtureCard(item, { compact: true })).join("")
              : `<p class="empty">No upcoming fixtures.</p>`
          }
        </div>
      </div>
    </div>
  `;
  bindCrestFallbacks(view);
}

async function renderUpcoming() {
  const params = new URLSearchParams(window.location.search);
  const query = {
    team: params.get("team") || "",
    date_from: params.get("date_from") || "",
    date_to: params.get("date_to") || "",
    page: params.get("page") || "1",
    page_size: String(PAGE_SIZE),
  };
  const data = await fetchJson(`/api/upcoming${qs(query)}`);
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  const hrefFor = (page) =>
    `/upcoming${qs({ team: query.team, date_from: query.date_from, date_to: query.date_to, page })}`;
  view.innerHTML = html`
    <div class="page-head">
      <div>
        <p class="eyebrow">${data.model.model_name} · ${data.model.feature_version}</p>
        <h1>Upcoming predictions</h1>
        <p class="lede">Unplayed fixtures only, grouped by date. ${data.total} remaining. Showing ${data.page_size} at a time.</p>
      </div>
    </div>
    ${filters(query)}
    ${renderGroups(data.groups)}
    ${pager(data, hrefFor)}
  `;
  fillTeams(data.teams, query.team);
  bindFilters("/upcoming");
  bindCrestFallbacks(view);
}

async function renderResults() {
  const params = new URLSearchParams(window.location.search);
  const query = {
    team: params.get("team") || "",
    date_from: params.get("date_from") || "",
    date_to: params.get("date_to") || "",
    page: params.get("page") || "1",
    page_size: String(PAGE_SIZE),
  };
  const data = await fetchJson(`/api/results${qs(query)}`);
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  const hrefFor = (page) =>
    `/results${qs({ team: query.team, date_from: query.date_from, date_to: query.date_to, page })}`;
  view.innerHTML = html`
    <div class="page-head">
      <div>
        <p class="eyebrow">Frozen pre-match probabilities</p>
        <h1>Results</h1>
        <p class="lede">Completed 2026/27 matches compared with the probabilities stored before kickoff. Those numbers are not rewritten after the score is known.</p>
      </div>
    </div>
    ${filters(query)}
    ${renderGroups(data.groups, { settled: true })}
    ${pager(data, hrefFor)}
  `;
  fillTeams(data.teams, query.team);
  bindFilters("/results");
  bindCrestFallbacks(view);
}

function chartRows(slice, emptyLabel) {
  const names = [
    ["home", "Home"],
    ["draw", "Draw"],
    ["away", "Away"],
  ];
  const maxN = Math.max(1, ...names.map(([key]) => slice[key]?.n || 0));
  return names
    .map(([key, label]) => {
      const row = slice[key] || { n: 0, correct: 0, rate: null };
      const width = Math.round(((row.n || 0) / maxN) * 100);
      const rate = row.n ? `${row.correct}/${row.n} · ${pct(row.rate)}` : emptyLabel;
      return html`
        <div class="chart-row">
          <span>${label}</span>
          <div class="chart-track"><span style="width:${row.n ? width : 0}%"></span></div>
          <span class="chart-label">${rate}</span>
        </div>
      `;
    })
    .join("");
}

async function renderPerformance() {
  const data = await fetchJson("/api/performance");
  const live = data.live_scorecard;
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  view.innerHTML = html`
    <div class="page-head">
      <div>
        <p class="eyebrow">Live 2026/27 only</p>
        <h1>Performance</h1>
        <p class="lede">Cumulative accuracy and log loss on settled live predictions. Walk-forward selection and the untouched 2025/26 test live on the About page.</p>
      </div>
    </div>
    <section class="stat-grid">
      <article class="stat-card">
        <h2>Accuracy</h2>
        <p class="stat-value">${live.n ? pct(live.accuracy) : "—"}</p>
      </article>
      <article class="stat-card">
        <h2>Log loss</h2>
        <p class="stat-value">${live.n ? live.log_loss.toFixed(3) : "—"}</p>
      </article>
      <article class="stat-card">
        <h2>Evaluated</h2>
        <p class="stat-value">${live.n}</p>
      </article>
      <article class="stat-card">
        <h2>Correct</h2>
        <p class="stat-value">${live.correct || 0}</p>
      </article>
    </section>
    <div class="split">
      <article class="panel">
        <h2>By actual result</h2>
        <p class="help">How often the stored argmax matched the true Home / Draw / Away class.</p>
        ${live.n ? chartRows(live.by_actual, "0") : `<p class="empty">No settled live predictions yet.</p>`}
      </article>
      <article class="panel">
        <h2>By predicted class</h2>
        <p class="help">When the model’s most likely outcome was this class, how often it was right.</p>
        ${live.n ? chartRows(live.by_predicted, "Never selected") : `<p class="empty">No settled live predictions yet.</p>`}
      </article>
    </div>
  `;
}

function metric(label, value) {
  return html`<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

async function renderAbout() {
  const data = await fetchJson("/api/about");
  const test = data.test || {};
  document.getElementById("season-label").textContent = "Premier League";
  view.innerHTML = html`
    <div class="page-head">
      <div>
        <p class="eyebrow">${data.model.model_name} · ${data.feature_version}</p>
        <h1>About the model</h1>
        <p class="lede">Production predictions come from an unweighted logistic regression chosen by walk-forward log loss. This page describes that model. It does not retrain it.</p>
      </div>
    </div>
    <div class="about-grid">
      <article class="panel">
        <h2>Selection</h2>
        <p>${data.selection_reason || "Unweighted logistic regression had the lowest mean walk-forward log loss."}</p>
        <ul>
          ${(data.walkforward_folds || [])
            .map((fold) => `<li>Train through ${fold.train_through}, validate ${fold.valid_season}</li>`)
            .join("")}
        </ul>
      </article>
      <article class="panel">
        <h2>Historical seasons</h2>
        <ul>
          <li>${data.history.ingest}</li>
          <li>${data.history.walkforward}</li>
          <li>${data.history.production_train}</li>
          <li>${data.history.test}</li>
          <li>${data.history.live}</li>
        </ul>
      </article>
      <article class="panel">
        <h2>Untouched ${data.test_season} test</h2>
        <div class="metrics">
          ${metric("Accuracy", test.n ? pct(test.accuracy) : "—")}
          ${metric("Log loss", test.log_loss != null ? Number(test.log_loss).toFixed(3) : "—")}
          ${metric("Matches", test.n ?? "—")}
          ${metric("Home F1", test.f1_home != null ? Number(test.f1_home).toFixed(3) : "—")}
          ${metric("Draw F1", test.f1_draw != null ? Number(test.f1_draw).toFixed(3) : "—")}
          ${metric("Away F1", test.f1_away != null ? Number(test.f1_away).toFixed(3) : "—")}
        </div>
      </article>
      <article class="panel">
        <h2>Major features</h2>
        <ul>
          ${data.features.map((item) => `<li><strong>${item.title}.</strong> ${item.detail}</li>`).join("")}
        </ul>
        <p>${data.leakage_note}</p>
      </article>
      <article class="panel" style="grid-column: 1 / -1">
        <h2>Draw limitation</h2>
        <p>${data.draw_limitation}</p>
      </article>
    </div>
  `;
}

async function renderForecast() {
  const data = await fetchJson("/api/forecast");
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  const rows = (data.teams || [])
    .map((row, index) => {
      const cls = row.contender ? "contender" : "";
      return html`
        <tr class="${cls}">
          <td class="rank">${index + 1}</td>
          <td class="team-cell">${escapeHtml(row.team)}${row.contender ? ` <span class="note watch">Title race</span>` : ""}</td>
          <td class="title-cell">${fmtPct(row.title_prob)}</td>
          <td>${fmtPct(row.top4_prob)}</td>
          <td>${fmtPct(row.relegation_prob)}</td>
          <td>${fmtNum(row.expected_position)}</td>
          <td>${fmtNum(row.expected_points)}</td>
          <td class="muted">${row.current_points} pts · ${row.current_played} played</td>
        </tr>
      `;
    })
    .join("");
  view.innerHTML = html`
    <div class="page-head">
      <div>
        <p class="eyebrow">Frozen remaining-fixture probabilities · ${data.n_sims.toLocaleString("en-GB")} simulations · seed ${data.seed}</p>
        <h1>Season Forecast</h1>
        <p class="lede">Each remaining 2026/27 match is sampled from the stored Home/Draw/Away probabilities after the Airflow pipeline writes this forecast. The production model is not retrained and those probabilities are not rewritten.</p>
      </div>
    </div>
    <section class="stat-grid">
      <article class="stat-card">
        <h2>Predictions updated</h2>
        <p class="stat-value">${formatStamp(data.predictions_updated_at)}</p>
        <p class="help">${data.model.model_name} · ${data.feature_version}</p>
      </article>
      <article class="stat-card">
        <h2>Forecast generated</h2>
        <p class="stat-value">${formatStamp(data.generated_at)}</p>
        <p class="help">${data.n_sims.toLocaleString("en-GB")} simulations</p>
      </article>
      <article class="stat-card">
        <h2>Completed matches</h2>
        <p class="stat-value">${data.n_completed}</p>
        <p class="help">Current 2026/27 table</p>
      </article>
      <article class="stat-card">
        <h2>Remaining fixtures</h2>
        <p class="stat-value">${data.n_remaining}</p>
        <p class="help">Unplayed matches with stored probabilities</p>
      </article>
    </section>
    <article class="panel forecast-panel">
      <h2>Club outlook</h2>
      <p class="help" style="margin-bottom:12px">Title favourite: <strong class="predicted">${data.teams.length ? escapeHtml(data.teams[0].team) : "—"}</strong> ${data.teams.length ? fmtPct(data.teams[0].title_prob) : ""}</p>
      <div class="table-wrap">
        <table class="forecast-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Club</th>
              <th>Title %</th>
              <th>Top 4 %</th>
              <th>Relegation %</th>
              <th>Expected pos.</th>
              <th>Expected pts</th>
              <th>Now</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </article>
    <article class="panel">
      <h2>Tie-break limitation</h2>
      <p class="help">${escapeHtml(data.tiebreak)}</p>
    </article>
    <article class="panel">
      <h2>Early-season limitation</h2>
      <p class="help">${escapeHtml(data.early_season_note || "")}</p>
    </article>
  `;
}

function teamSide(name, crest, positionLabel, align) {
  const rank = positionLabel ? `<span class="rank-pill">${escapeHtml(positionLabel)}</span>` : "";
  return html`
    <div class="match-side ${align}">
      ${crestHtml(name, crest, "crest-lg")}
      <p class="match-team">${escapeHtml(name)}</p>
      ${rank}
    </div>
  `;
}

function winProbability(match, { secondary = false } = {}) {
  if (!match.has_prediction) {
    if (secondary) return "";
    return `<section class="win-prob"><p class="help">No stored pre-match prediction for this fixture.</p></section>`;
  }
  const predicted = match.predicted_outcome;
  const homeOn = predicted === match.home_team ? "is-predicted" : "";
  const drawOn = predicted === "Draw" ? "is-predicted" : "";
  const awayOn = predicted === match.away_team ? "is-predicted" : "";
  const verdict = match.correct == null
    ? ""
    : `<span class="pill ${match.correct ? "ok" : "wrong"}">${match.correct ? "Correct" : "Missed"}</span>`;
  const actual = match.actual_outcome
    ? `<p class="help">Actual outcome: ${escapeHtml(match.actual_outcome)}</p>`
    : "";
  return html`
    <section class="win-prob ${secondary ? "is-secondary" : ""}">
      <h2>${secondary ? "Pre-match prediction" : "Win probability"}</h2>
      <div class="win-labels">
        <div class="${homeOn}">
          <span>${escapeHtml(match.home_team)}</span>
          <strong>${match.p_home_pct}%</strong>
        </div>
        <div class="${drawOn}">
          <span>Draw</span>
          <strong>${match.p_draw_pct}%</strong>
        </div>
        <div class="${awayOn}">
          <span>${escapeHtml(match.away_team)}</span>
          <strong>${match.p_away_pct}%</strong>
        </div>
      </div>
      <div class="win-bar" role="img" aria-label="Home ${match.p_home_pct} percent, draw ${match.p_draw_pct} percent, away ${match.p_away_pct} percent">
        <span class="seg home" style="width:${match.p_home_pct}%"></span>
        <span class="seg draw" style="width:${match.p_draw_pct}%"></span>
        <span class="seg away" style="width:${match.p_away_pct}%"></span>
      </div>
      <div class="win-outcome">
        <div>
          <p class="predicted">Predicted outcome: ${escapeHtml(predicted)}</p>
          ${actual}
        </div>
        <div class="outcome">
          ${verdict}
          <span class="pill">${escapeHtml(match.model_name)}${match.feature_version ? ` · ${escapeHtml(match.feature_version)}` : ""}</span>
        </div>
      </div>
    </section>
  `;
}

function teamStatsBlock(match, stats, available) {
  if (!available || !stats.length) {
    return `<p class="stats-fallback">Detailed match stats not available</p>`;
  }
  const rows = stats
    .map((stat) => {
      const homeCls = stat.leader === "home" ? "is-leader" : "";
      const awayCls = stat.leader === "away" ? "is-leader" : "";
      return html`
        <div class="stat-line">
          <span class="stat-val ${homeCls}">${stat.home}</span>
          <span class="stat-name">${escapeHtml(stat.label)}</span>
          <span class="stat-val ${awayCls}">${stat.away}</span>
        </div>
      `;
    })
    .join("");
  return html`
    <div class="team-stats">
      <div class="stat-line stat-head">
        ${crestHtml(match.home_team, match.home_crest, "crest-xs")}
        <span class="stat-name">Team stats</span>
        ${crestHtml(match.away_team, match.away_crest, "crest-xs")}
      </div>
      ${rows}
    </div>
  `;
}

function h2hCard(meeting) {
  const href = meeting.detail_path || `/results/${meeting.match_id}`;
  return html`
    <a class="h2h-card is-link" data-link href="${href}" aria-label="${escapeHtml(meeting.home_team)} ${meeting.home_goals}–${meeting.away_goals} ${escapeHtml(meeting.away_team)}">
      <p class="h2h-date">${formatKickoff(meeting.kickoff_date, meeting.kickoff_time)}</p>
      <div class="h2h-board">
        <div class="h2h-club">
          ${crestHtml(meeting.home_team, meeting.home_crest, "crest-md")}
          <span>${escapeHtml(meeting.home_team)}</span>
        </div>
        <p class="h2h-score">${meeting.home_goals}–${meeting.away_goals}</p>
        <div class="h2h-club away">
          ${crestHtml(meeting.away_team, meeting.away_crest, "crest-md")}
          <span>${escapeHtml(meeting.away_team)}</span>
        </div>
      </div>
      <p class="h2h-result">${escapeHtml(meeting.result_label || "")}</p>
    </a>
  `;
}

function bindMatchTabs(root) {
  const tabs = [...root.querySelectorAll("[data-match-tab]")];
  const panels = [...root.querySelectorAll("[data-match-panel]")];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.matchTab;
      tabs.forEach((item) => {
        item.classList.toggle("active", item === tab);
        item.setAttribute("aria-selected", item === tab ? "true" : "false");
      });
      panels.forEach((panel) => {
        panel.hidden = panel.dataset.matchPanel !== id;
      });
    });
  });
}

async function renderMatchDetail(matchId, requestedKind) {
  const data = await fetchJson(`/api/matches/${matchId}`);
  const canonical = data.kind === "result" ? `/results/${matchId}` : `/upcoming/${matchId}`;
  if (requestedKind && requestedKind !== data.kind) {
    window.history.replaceState({}, "", canonical);
  }
  if (data.kind === "result") {
    renderResultDetail(data);
  } else {
    renderUpcomingDetail(data);
  }
  bindCrestFallbacks(view);
}

function renderUpcomingDetail(data) {
  const match = data.match;
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  const note = match.note
    ? `<span class="note ${match.note === "Draw watch" ? "watch" : ""}">${escapeHtml(match.note)}</span>`
    : "";
  const meetings = data.head_to_head || [];
  const meetingCards = meetings.length
    ? meetings.map(h2hCard).join("")
    : `<p class="empty">No previous meetings in the ingested history.</p>`;
  view.innerHTML = html`
    <p class="back-row">
      <a class="pill" data-link href="/upcoming">Back to upcoming</a>
    </p>
    <article class="match-hero">
      <div class="match-meta">
        <span>${escapeHtml(match.competition || "Premier League")} · ${formatKickoff(match.kickoff_date, match.kickoff_time)}</span>
        <span class="match-status">${escapeHtml(match.status || "Upcoming")}</span>
      </div>
      <div class="match-board">
        ${teamSide(match.home_team, match.home_crest, match.home_position_label, "home")}
        <div class="match-center">
          <p class="match-vs">vs</p>
          <p class="help">Home vs Away</p>
        </div>
        ${teamSide(match.away_team, match.away_crest, match.away_position_label, "away")}
      </div>
      ${note}
    </article>
    <div class="match-tabs" role="tablist">
      <button type="button" class="active" data-match-tab="prediction" role="tab" aria-selected="true">Prediction</button>
      <button type="button" data-match-tab="meetings" role="tab" aria-selected="false">Previous meetings</button>
    </div>
    <div data-match-panel="prediction">
      ${winProbability(match)}
    </div>
    <div data-match-panel="meetings" hidden>
      <section class="h2h-section">
        <h2>Last ${data.h2h_limit} head-to-head</h2>
        <p class="help">Played meetings before ${formatKickoff(match.kickoff_date, match.kickoff_time)}. ${data.h2h_count} shown. Open a result for that match’s stats.</p>
        <div class="h2h-list">${meetingCards}</div>
      </section>
    </div>
  `;
  bindMatchTabs(view);
}

function renderResultDetail(data) {
  const match = data.match;
  document.getElementById("season-label").textContent = `Premier League · ${data.live_season}`;
  view.innerHTML = html`
    <p class="back-row">
      <a class="pill" data-link href="/results">Back to results</a>
    </p>
    <article class="match-hero">
      <div class="match-meta">
        <span>${escapeHtml(match.competition || "Premier League")} · ${formatKickoff(match.kickoff_date, match.kickoff_time)}</span>
        <span class="match-status">${escapeHtml(match.status || "Full-time")}</span>
      </div>
      <div class="match-board">
        ${teamSide(match.home_team, match.home_crest, match.home_position_label, "home")}
        <div class="match-center">
          <p class="match-score">${match.home_goals}–${match.away_goals}</p>
          <p class="help">${escapeHtml(match.actual_outcome || "Full-time")}</p>
        </div>
        ${teamSide(match.away_team, match.away_crest, match.away_position_label, "away")}
      </div>
    </article>
    ${match.has_prediction ? winProbability(match, { secondary: true }) : ""}
    <section class="h2h-section">
      <h2>Match stats</h2>
      ${teamStatsBlock(match, data.stats || [], data.stats_available)}
    </section>
    ${
      data.timeline_available
        ? html`<section class="h2h-section timeline-note">
            <h2>Goal timeline</h2>
            <p class="help">Goal scorers from the ingested source.</p>
          </section>`
        : ""
    }
  `;
}

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
    view.innerHTML = `<p class="empty">${escapeHtml(error.message || "Could not load this page from the database.")}</p>`;
  }
}

function go(href) {
  const url = new URL(href, window.location.origin);
  window.history.pushState({}, "", `${url.pathname}${url.search}`);
  render();
}

document.addEventListener("click", (event) => {
  const link = event.target.closest("[data-link]");
  if (!link) return;
  const url = new URL(link.href, window.location.origin);
  if (url.origin !== window.location.origin) return;
  event.preventDefault();
  go(`${url.pathname}${url.search}`);
});

window.addEventListener("popstate", render);
render();
