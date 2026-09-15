/* MatchLab dashboard.
   Read-only client for the FastAPI endpoints. Renders the stored probabilities
   exactly as the API returns them and never recomputes a number. */

const view = document.getElementById("view");
const footMeta = document.getElementById("foot-meta");
const PAGE_SIZE = 16;
const DEFAULT_LEAGUE = "E0";
const LEAGUE_KEY = "matchlab.league";

/* ==========================================================================
   League state

   The URL is the source of truth so every page stays linkable to one league.
   The last choice is also mirrored into localStorage, so opening a bare
   /upcoming restores the league you were reading instead of silently
   dropping back to the Premier League.
   ========================================================================== */

let catalog = { competitions: [], default: DEFAULT_LEAGUE, byCode: {} };

function storedLeague() {
  try {
    return window.localStorage.getItem(LEAGUE_KEY);
  } catch (error) {
    return null; /* private mode, or storage blocked */
  }
}

function rememberLeague(code) {
  try {
    window.localStorage.setItem(LEAGUE_KEY, code);
  } catch (error) {
    /* nothing to do: the URL still carries the league */
  }
}

function knownLeague(code) {
  if (!code) return null;
  if (!catalog.competitions.length) return code;
  return catalog.byCode[code] ? code : null;
}

function currentLeague() {
  const fromUrl = knownLeague(new URLSearchParams(window.location.search).get("league"));
  if (fromUrl) return fromUrl;
  return knownLeague(storedLeague()) || catalog.default || DEFAULT_LEAGUE;
}

function currentSpec() {
  return catalog.byCode[currentLeague()] || null;
}

/* Put the resolved league into the address bar without adding a history
   entry, so a shared link always names its competition. */
function pinLeagueToUrl() {
  const league = currentLeague();
  const url = new URL(window.location.href);
  if (url.searchParams.get("league") === league) return;
  url.searchParams.set("league", league);
  window.history.replaceState({}, "", `${url.pathname}${url.search}`);
}

function withLeague(href, extra = {}) {
  const url = new URL(href, window.location.origin);
  url.searchParams.set("league", currentLeague());
  Object.entries(extra).forEach(([key, value]) => {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  });
  return `${url.pathname}${url.search}`;
}

function leagueName(data) {
  if (data && data.competition && data.competition.name) return data.competition.name;
  const spec = currentSpec();
  return spec ? spec.name : "Premier League";
}

/* ------------------------------------------------------------------ utils */

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
  const day = parseDate(dateStr, timeStr).toLocaleDateString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  return timeStr ? `${day}, ${timeStr}` : day;
}

function formatDay(dateStr) {
  return parseDate(dateStr, "12:00").toLocaleDateString(undefined, {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

function formatLongDay(dateStr) {
  return parseDate(dateStr, "12:00").toLocaleDateString(undefined, {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
}

function formatShortDate(dateStr) {
  return parseDate(dateStr, "12:00").toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "2-digit",
  });
}

function formatStamp(iso) {
  if (!iso) return "unknown";
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatCount(value) {
  return Number(value ?? 0).toLocaleString();
}

const NONE = "-";

function pct(value) {
  if (value == null) return NONE;
  return `${Math.round(value * 100)}%`;
}

function fmtPct(value) {
  if (value == null) return NONE;
  const p = value * 100;
  if (p >= 1) return `${p.toFixed(1)}%`;
  if (p > 0) return `${p.toFixed(2)}%`;
  return "0%";
}

function fmtNum(value, digits = 1) {
  if (value == null) return NONE;
  return Number(value).toFixed(digits);
}

function setFootMeta(model, extra) {
  if (!model) {
    footMeta.textContent = "";
    return;
  }
  const bits = [`${model.model_name} model, feature set ${model.feature_version}`];
  if (extra) bits.push(extra);
  footMeta.textContent = bits.join(". ");
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

/* ==========================================================================
   Header: navigation and the league menu
   ========================================================================== */

const leagueButton = document.getElementById("league-button");
const leagueMenu = document.getElementById("league-menu");
const leaguePicker = document.getElementById("league-picker");

function closeLeagueMenu() {
  if (!leagueMenu || leagueMenu.hidden) return;
  leagueMenu.hidden = true;
  leagueButton.setAttribute("aria-expanded", "false");
}

function openLeagueMenu() {
  if (!leagueMenu) return;
  leagueMenu.hidden = false;
  leagueButton.setAttribute("aria-expanded", "true");
  const active = leagueMenu.querySelector(".league-option.is-active") || leagueMenu.querySelector(".league-option");
  active?.focus();
}

leagueButton?.addEventListener("click", () => {
  if (leagueMenu.hidden) openLeagueMenu();
  else closeLeagueMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || leagueMenu?.hidden) return;
  closeLeagueMenu();
  leagueButton?.focus();
});

document.addEventListener("pointerdown", (event) => {
  if (leagueMenu?.hidden) return;
  if (leaguePicker && !leaguePicker.contains(event.target)) closeLeagueMenu();
});

function paintLeaguePicker() {
  const league = currentLeague();
  const spec = currentSpec();
  const markEl = document.getElementById("league-button-mark");
  const nameEl = document.getElementById("league-button-name");
  if (spec) {
    if (markEl) markEl.textContent = spec.mark;
    if (nameEl) nameEl.textContent = spec.name;
    leagueButton?.setAttribute("aria-label", `League: ${spec.name}. Choose a different league`);
  }

  const options = document.getElementById("league-options");
  if (!options || !catalog.competitions.length) return;

  /* The menu keeps the current path, so switching league stays on the page
     you were reading rather than bouncing you back to the overview. */
  const here = pathOf();
  const target = /^\/(upcoming|results)\/\d+$/.test(here) ? "/" : here;

  options.innerHTML = catalog.competitions
    .map((spec_) => {
      const active = spec_.code === league;
      return html`
        <a
          class="league-option ${active ? "is-active" : ""}"
          role="menuitem"
          data-link
          data-league="${spec_.code}"
          href="${target}?league=${spec_.code}"
          ${active ? 'aria-current="true"' : ""}
        >
          <span class="mark" aria-hidden="true">${escapeHtml(spec_.mark)}</span>
          <span>
            ${escapeHtml(spec_.name)}
            <span class="country">${escapeHtml(spec_.country)}</span>
          </span>
          ${active ? '<span class="tick" aria-hidden="true">✓</span>' : ""}
        </a>
      `;
    })
    .join("");
}

function setNav() {
  const current = pathOf();
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const target = link.dataset.nav;
    link.setAttribute("href", withLeague(target));
    const active =
      current === target ||
      (target === "/upcoming" && current.startsWith("/upcoming/")) ||
      (target === "/results" && current.startsWith("/results/"));
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  const brand = document.querySelector(".brand");
  if (brand) brand.setAttribute("href", withLeague("/"));
  paintLeaguePicker();
  document.querySelector(".nav a.active")?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function matchDetailRoute() {
  const upcoming = pathOf().match(/^\/upcoming\/(\d+)$/);
  if (upcoming) return { kind: "upcoming", id: Number(upcoming[1]) };
  const result = pathOf().match(/^\/results\/(\d+)$/);
  if (result) return { kind: "result", id: Number(result[1]) };
  return null;
}

/* ==========================================================================
   Crests
   ========================================================================== */

const CREST_PX = { "crest-xs": 20, "crest-sm": 26, "crest-md": 40, "crest-lg": 64 };

function crestHtml(team, crest, sizeClass = "crest-sm") {
  const badge = crest || {};
  const initials = escapeHtml(badge.initials || String(team || "?").slice(0, 3).toUpperCase());
  const color = escapeHtml(badge.color || "#3d4338");
  const ink = escapeHtml(badge.ink || "#f4f1e8");
  const size = CREST_PX[sizeClass] || 26;
  /* The club name always sits next to the crest, so the initials badge is
     decorative and must not be announced a second time. */
  const fallback = `<span class="crest-fallback" aria-hidden="true" style="background:${color};color:${ink}">${initials}</span>`;
  if (!badge.crest_url) {
    return `<span class="crest-wrap ${sizeClass}">${fallback}</span>`;
  }
  /* width/height are set so the crest reserves its box before it loads. */
  return html`<span class="crest-wrap ${sizeClass}"
    >${fallback}<img
      class="crest"
      src="${escapeHtml(badge.crest_url)}"
      width="${size}"
      height="${size}"
      alt=""
      loading="lazy"
      decoding="async"
    /></span
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

/* ==========================================================================
   Probability bar: the signature component

   One geometry everywhere. Segment width is the probability, the filled
   segment is the model's most likely outcome, and the colour of each third
   means the same thing on every page in the product.
   ========================================================================== */

function probColumns(match) {
  return [
    { key: "home", label: "Home", value: match.p_home_pct, name: match.home_team },
    { key: "draw", label: "Draw", value: match.p_draw_pct, name: "Draw" },
    { key: "away", label: "Away", value: match.p_away_pct, name: match.away_team },
  ];
}

function probBar(match, { large = false } = {}) {
  const pick = match.predicted_outcome;
  const cols = probColumns(match);
  /* Below this share the number cannot be read inside its own segment, so it
     moves down to the label row instead of being clipped. */
  const tightAt = large ? 5 : 9;

  const segs = cols
    .map((col) => {
      const tight = col.value < tightAt;
      const classes = ["seg", col.key, pick === col.name ? "is-pick" : "", tight ? "is-tight" : ""]
        .filter(Boolean)
        .join(" ");
      return `<span class="${classes}" style="flex:${Math.max(col.value, 1)} 1 0">${tight ? "" : `${col.value}%`}</span>`;
    })
    .join("");

  const legend = cols
    .map((col) => `<span>${col.label}${col.value < tightAt ? ` ${col.value}%` : ""}</span>`)
    .join("");

  const aria = `Home win ${match.p_home_pct} percent, draw ${match.p_draw_pct} percent, away win ${match.p_away_pct} percent`;

  return html`
    <div class="probs ${large ? "probs-lg" : ""}">
      <div class="probbar" role="img" aria-label="${escapeHtml(aria)}">${segs}</div>
      <div class="problegend" aria-hidden="true">${legend}</div>
    </div>
  `;
}

/* ==========================================================================
   Match cards

   Teams are stacked vertically, so a full club name never has to be
   truncated and each name appears exactly once on the card.
   ========================================================================== */

function noteTag(note) {
  if (!note) return "";
  return `<span class="tag tag-watch">${escapeHtml(note)}</span>`;
}

function verdictBadge(correct) {
  if (correct == null) return "";
  return correct
    ? `<span class="verdict verdict-ok"><span aria-hidden="true">✓</span>Correct</span>`
    : `<span class="verdict verdict-miss"><span aria-hidden="true">✕</span>Missed</span>`;
}

function teamRow(name, crest, { goals = null, down = false, pick = false } = {}) {
  const classes = ["team-row", down ? "is-down" : "", pick ? "is-pick" : ""].filter(Boolean).join(" ");
  const right = goals == null ? "" : `<span class="goals">${goals}</span>`;
  return html`
    <div class="${classes}">
      ${crestHtml(name, crest, "crest-sm")}
      <span class="team-name">${escapeHtml(name)}</span>
      ${right}
    </div>
  `;
}

function matchCard(item, { settled = false, showDate = true } = {}) {
  const href = item.match_id
    ? withLeague(`${settled ? "/results" : "/upcoming"}/${item.match_id}`)
    : "";
  const label = settled
    ? `${item.home_team} ${item.home_goals}-${item.away_goals} ${item.away_team}, match detail`
    : `${item.home_team} versus ${item.away_team}, match detail`;
  const link = href
    ? `<a class="stretch-link" data-link href="${href}" aria-label="${escapeHtml(label)}"></a>`
    : "";

  const timeText = settled
    ? showDate
      ? formatKickoff(item.kickoff_date, item.kickoff_time)
      : item.kickoff_time || formatShortDate(item.kickoff_date)
    : showDate
      ? formatKickoff(item.kickoff_date, item.kickoff_time)
      : item.kickoff_time || formatShortDate(item.kickoff_date);

  const topRight = settled ? verdictBadge(item.correct) : noteTag(item.note);

  const homeWon = settled && item.home_goals > item.away_goals;
  const awayWon = settled && item.away_goals > item.home_goals;
  const pickHome = !settled && item.has_prediction && item.predicted_outcome === item.home_team;
  const pickAway = !settled && item.has_prediction && item.predicted_outcome === item.away_team;

  const teams = settled
    ? teamRow(item.home_team, item.home_crest, { goals: item.home_goals, down: awayWon }) +
      teamRow(item.away_team, item.away_crest, { goals: item.away_goals, down: homeWon })
    : teamRow(item.home_team, item.home_crest, { pick: pickHome }) +
      teamRow(item.away_team, item.away_crest, { pick: pickAway });

  const body = item.has_prediction
    ? probBar(item)
    : `<p class="no-pred">No prediction was stored before this match.</p>`;

  return html`
    <article class="match-card ${href ? "is-link" : ""}">
      ${link}
      <div class="match-card-top">
        <time class="match-time" datetime="${escapeHtml(item.kickoff_date)}">${escapeHtml(timeText)}</time>
        ${topRight}
      </div>
      <div class="teams">${teams}</div>
      ${body}
    </article>
  `;
}

function renderGroups(groups, { settled = false, emptyMessage = "" } = {}) {
  if (!groups.length) {
    return `<p class="empty">${escapeHtml(emptyMessage)}</p>`;
  }
  return groups
    .map(
      (group) => html`
        <section class="daygroup">
          <div class="day-head"><h2>${formatDay(group.date)}</h2></div>
          <div class="match-grid">
            ${group.fixtures.map((item) => matchCard(item, { settled, showDate: false })).join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

/* ------------------------------------------------------- filters and pager */

function filters(values) {
  return html`
    <form class="filters" id="filter-form">
      <label>
        Team
        <select name="team" id="team-filter" autocomplete="off">
          <option value="">All teams</option>
        </select>
      </label>
      <label>
        From
        <input name="date_from" id="date-from" type="date" autocomplete="off" value="${escapeHtml(values.date_from || "")}" />
      </label>
      <label>
        To
        <input name="date_to" id="date-to" type="date" autocomplete="off" value="${escapeHtml(values.date_to || "")}" />
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
        league: currentLeague(),
        team: data.get("team"),
        date_from: data.get("date_from"),
        date_to: data.get("date_to"),
        page: "1",
      })}`,
    );
  });
  document.getElementById("clear-filters")?.addEventListener("click", () => go(withLeague(basePath)));
}

function pager(data, buildHref) {
  const total = `${formatCount(data.total)} ${data.total === 1 ? "match" : "matches"}`;
  if (data.pages <= 1) {
    return `<div class="pager"><span>${total}</span></div>`;
  }
  const prev =
    data.page > 1 ? `<a class="btn pill-sm" data-link href="${buildHref(data.page - 1)}">Previous</a>` : "";
  const next =
    data.page < data.pages ? `<a class="btn pill-sm" data-link href="${buildHref(data.page + 1)}">Next</a>` : "";
  return html`
    <div class="pager">
      <span>Page ${data.page} of ${data.pages}, ${total}</span>
      <div class="pager-actions">${prev}${next}</div>
    </div>
  `;
}

/* ------------------------------------------------------------ page heading */

function pageHead(season, title, lede, aside = "") {
  return html`
    <div class="page-head">
      <div class="page-head-main">
        ${season ? `<p class="season-tag"><span class="dot" aria-hidden="true"></span>${escapeHtml(season)}</p>` : ""}
        <h1>${escapeHtml(title)}</h1>
        ${lede ? `<p class="lede">${lede}</p>` : ""}
      </div>
      ${aside}
    </div>
  `;
}

function statCell(label, value, foot, { accent = false, small = false } = {}) {
  return html`
    <div class="stat ${accent ? "stat-accent" : ""}">
      <span class="stat-label">${escapeHtml(label)}</span>
      <span class="stat-value ${small ? "sm" : ""}">${value}</span>
      <span class="stat-foot">${escapeHtml(foot)}</span>
    </div>
  `;
}

/* ==========================================================================
   Overview
   ========================================================================== */

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
        <a class="pill pill-sm" data-link href="${withLeague("/forecast")}">Full table</a>
      </div>
      <p class="help">Chance of finishing first across ${formatCount(forecast.n_sims)} simulated seasons.</p>
      <div class="race-list" style="margin-top:var(--sp-3)">${rows}</div>
    </article>
  `;
}

async function renderOverview() {
  const league = currentLeague();
  const data = await fetchJson(`/api/overview${qs({ league })}`);
  const forecast = await fetchJson(`/api/forecast${qs({ league })}`).catch(() => null);

  setFootMeta(data.model, `Predictions updated ${formatStamp(data.predictions_updated_at)}`);

  const live = data.live_scorecard;
  const latest = data.latest_completed_match;
  const name = leagueName(data);

  const latestPanel = latest
    ? html`
        <div class="latest">
          <div class="latest-row ${latest.away_goals > latest.home_goals ? "is-down" : ""}">
            ${crestHtml(latest.home_team, latest.home_crest, "crest-sm")}
            <span>${escapeHtml(latest.home_team)}</span>
            <span class="goals">${latest.home_goals}</span>
          </div>
          <div class="latest-row ${latest.home_goals > latest.away_goals ? "is-down" : ""}">
            ${crestHtml(latest.away_team, latest.away_crest, "crest-sm")}
            <span>${escapeHtml(latest.away_team)}</span>
            <span class="goals">${latest.away_goals}</span>
          </div>
          <p class="help">${formatKickoff(latest.kickoff_date, latest.kickoff_time)}</p>
        </div>
      `
    : `<p class="help">No completed matches yet this season.</p>`;

  view.innerHTML = html`
    ${pageHead(
      `${name}, ${data.live_season}`,
      `${name} predictions`,
      "Home, Draw and Away probabilities for every remaining fixture. Each one is frozen before kickoff, so results are always judged against what was predicted beforehand.",
    )}

    <section class="statbar" aria-label="Season at a glance">
      ${statCell("Live accuracy", live.n ? pct(live.accuracy) : NONE, live.n ? `${live.correct} of ${live.n} correct` : "Awaiting results", { accent: true })}
      ${statCell("Matches played", formatCount(data.n_settled), `So far in ${data.live_season}`)}
      ${statCell("Fixtures ahead", formatCount(data.n_upcoming), "Every one has a prediction")}
      ${statCell("Log loss", live.n ? live.log_loss.toFixed(3) : NONE, "Lower is better")}
    </section>

    <div class="overview-cols">
      <div>
        <div class="section-head">
          <h2>Next fixtures</h2>
          <a class="pill pill-sm" data-link href="${withLeague("/upcoming")}">All fixtures</a>
        </div>
        <div class="match-grid">
          ${data.next_upcoming.length
            ? data.next_upcoming.slice(0, 6).map((item) => matchCard(item)).join("")
            : `<p class="empty">No upcoming fixtures with a stored prediction.</p>`}
        </div>
      </div>

      <div class="side-stack">
        <article class="panel">
          <div class="section-head" style="margin-bottom:var(--sp-3)">
            <h2>Latest result</h2>
            <a class="pill pill-sm" data-link href="${withLeague("/results")}">All results</a>
          </div>
          ${latestPanel}
        </article>
        ${titleRacePanel(forecast)}
      </div>
    </div>
  `;
  bindCrestFallbacks(view);
}

/* ==========================================================================
   Fixtures and results
   ========================================================================== */

function listQuery() {
  const params = new URLSearchParams(window.location.search);
  return {
    league: currentLeague(),
    team: params.get("team") || "",
    date_from: params.get("date_from") || "",
    date_to: params.get("date_to") || "",
    page: params.get("page") || "1",
    page_size: String(PAGE_SIZE),
  };
}

/* A filtered list that finds nothing and a season that has not started are
   different situations, so they do not share a message. */
function emptyListMessage(query, nothingYet) {
  const filtered = Boolean(query.team || query.date_from || query.date_to);
  return filtered ? "No matches match these filters. Clear them to see everything again." : nothingYet;
}

async function renderUpcoming() {
  const query = listQuery();
  const data = await fetchJson(`/api/upcoming${qs(query)}`);
  setFootMeta(data.model);

  const hrefFor = (page) =>
    `/upcoming${qs({ league: query.league, team: query.team, date_from: query.date_from, date_to: query.date_to, page })}`;

  view.innerHTML = html`
    ${pageHead(
      `${leagueName(data)}, ${data.live_season}`,
      "Fixtures",
      `Every unplayed match with the probabilities recorded for it. ${formatCount(data.total)} to come.`,
    )}
    ${filters(query)}
    ${renderGroups(data.groups, { emptyMessage: emptyListMessage(query, "No fixtures are scheduled with a stored prediction yet.") })}
    ${pager(data, hrefFor)}
  `;
  fillTeams(data.teams, query.team);
  bindFilters("/upcoming");
  bindCrestFallbacks(view);
}

async function renderResults() {
  const query = listQuery();
  const data = await fetchJson(`/api/results${qs(query)}`);
  setFootMeta(data.model);

  const hrefFor = (page) =>
    `/results${qs({ league: query.league, team: query.team, date_from: query.date_from, date_to: query.date_to, page })}`;

  view.innerHTML = html`
    ${pageHead(
      `${leagueName(data)}, ${data.live_season}`,
      "Results",
      "Completed matches shown against the probabilities that were stored before kickoff. Those numbers are never rewritten once the result is known.",
    )}
    ${filters(query)}
    ${renderGroups(data.groups, {
      settled: true,
      emptyMessage: emptyListMessage(query, "No matches have been played yet this season."),
    })}
    ${pager(data, hrefFor)}
  `;
  fillTeams(data.teams, query.team);
  bindFilters("/results");
  bindCrestFallbacks(view);
}

/* ==========================================================================
   Accuracy

   Written for someone who follows football, not for someone reading a model
   report. The bar length is the success rate, and the sample size sits next
   to it as plain text.
   ========================================================================== */

function rateRows(slice, emptyLabel) {
  const names = [
    ["home", "Home win"],
    ["draw", "Draw"],
    ["away", "Away win"],
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
            ${row.n ? pct(row.rate) : NONE}<small>${row.n ? `${row.correct} of ${row.n}` : emptyLabel}</small>
          </span>
        </div>
      `;
    })
    .join("");
}

async function renderPerformance() {
  const league = currentLeague();
  const data = await fetchJson(`/api/performance${qs({ league })}`);
  const recent = await fetchJson(`/api/results${qs({ league, page: "1", page_size: "20" })}`).catch(() => null);
  const live = data.live_scorecard;
  setFootMeta(data.model);

  const settled = recent
    ? recent.groups.flatMap((group) => group.fixtures).filter((item) => item.has_prediction)
    : [];

  const formRows = settled
    .slice(0, 10)
    .map(
      (item) => html`
        <a class="form-row" data-link href="${withLeague(`/results/${item.match_id}`)}">
          <span class="form-mark ${item.correct ? "ok" : "miss"}" aria-hidden="true">${item.correct ? "✓" : "✕"}</span>
          <span class="form-teams">
            ${escapeHtml(item.home_team)} v ${escapeHtml(item.away_team)}
            <span class="vh">${item.correct ? "predicted correctly" : "prediction missed"}</span>
          </span>
          <span class="form-score">${item.home_goals}-${item.away_goals}</span>
          <span class="form-pick">picked ${escapeHtml(item.predicted_outcome)}</span>
        </a>
      `,
    )
    .join("");

  view.innerHTML = html`
    ${pageHead(
      `${leagueName(data)}, ${data.live_season}`,
      "Accuracy",
      "How the predictions have actually held up this season, judged only against probabilities that were stored before kickoff.",
      `<a class="pill" data-link href="${withLeague("/about")}">How the model is built</a>`,
    )}

    <section class="statbar" aria-label="Accuracy at a glance">
      ${statCell("Correct calls", live.n ? pct(live.accuracy) : NONE, live.n ? `${live.correct} of ${live.n} matches` : "Awaiting results", { accent: true })}
      ${statCell("Matches judged", formatCount(live.n), "Played, with a stored prediction")}
      ${statCell("Best on", bestSlice(live), "Result type called most often")}
      ${statCell("Log loss", live.n ? live.log_loss.toFixed(3) : NONE, "Scores the odds, not just the pick")}
    </section>

    <div class="overview-cols">
      <div>
        ${settled.length
          ? html`
              <div class="section-head">
                <h2>Recent matches</h2>
                <a class="pill pill-sm" data-link href="${withLeague("/results")}">All results</a>
              </div>
              <div class="panel" style="padding:var(--sp-3)">
                <div class="formlist">${formRows}</div>
              </div>
            `
          : `<p class="empty">No results to judge yet this season.</p>`}
      </div>

      <div class="side-stack">
        <article class="panel">
          <h2>When a match ended this way</h2>
          <p class="help">Of the matches that really finished as a home win, a draw or an away win, how many the model had called.</p>
          ${live.n
            ? `<div class="rate-list">${rateRows(live.by_actual, "none yet")}</div>`
            : `<p class="empty">Nothing settled yet.</p>`}
        </article>
        <article class="panel">
          <h2>When the model called it</h2>
          <p class="help">Of the matches where this was the most likely outcome, how many turned out that way.</p>
          ${live.n
            ? `<div class="rate-list">${rateRows(live.by_predicted, "never called")}</div>`
            : `<p class="empty">Nothing settled yet.</p>`}
        </article>
      </div>
    </div>
  `;
  bindCrestFallbacks(view);
}

/* The strongest of the three outcome classes, by hit rate on a non-empty
   sample. Used for a plain-language headline stat. */
function bestSlice(live) {
  if (!live.n) return NONE;
  const labels = { home: "Home wins", draw: "Draws", away: "Away wins" };
  let best = null;
  for (const key of ["home", "draw", "away"]) {
    const row = live.by_actual[key];
    if (!row || !row.n || row.rate == null) continue;
    if (!best || row.rate > best.rate) best = { key, rate: row.rate };
  }
  return best ? labels[best.key] : NONE;
}

/* ==========================================================================
   Season forecast
   ========================================================================== */

let forecastState = { rows: [], sort: null, dir: "desc", meta: null };

const FORECAST_COLUMNS = [
  { key: "projected_rank", label: "#", sortable: false },
  { key: "team", label: "Club", sortable: true, type: "text" },
  { key: "current_played", label: "Pld", sortable: true },
  { key: "current_points", label: "Pts", sortable: true },
  { key: "expected_points", label: "Proj. pts", sortable: true },
  { key: "title_prob", label: "Title", sortable: true },
  { key: "ucl_prob", label: "UCL", sortable: true },
  { key: "europe_prob", label: "Europe", sortable: true },
  { key: "relegation_prob", label: "Relegation", sortable: true },
];

/* Zones come from the competition payload, so an 18-club league with three
   Champions League places marks exactly those rows and no others. */
function zoneFor(rank, meta) {
  if (!meta) return "";
  if (rank <= meta.ucl_places) return "ucl";
  if (rank < meta.europe_places) return "europa";
  if (rank <= meta.europe_places) return "conference";
  if (rank > meta.n_teams - meta.relegation_places) return "releg";
  return "";
}

function databar(kind, value) {
  const zero = !value || value <= 0;
  return html`
    <span class="databar ${kind} ${zero ? "is-zero" : ""}">
      ${zero ? "" : `<span class="fill" style="--w:${Math.min(100, Math.round(value * 100))}%"></span>`}
      <span class="v">${fmtPct(value)}</span>
    </span>
  `;
}

function forecastRow(row) {
  const zone = zoneFor(row.projected_rank, forecastState.meta);
  return html`
    <tr ${zone ? `data-zone="${zone}"` : ""}>
      <td class="col-pos">${row.projected_rank}</td>
      <td class="col-club">
        <span class="club-cell">
          ${crestHtml(row.team, row.crest, "crest-xs")}
          <span class="name">${escapeHtml(row.team)}</span>
        </span>
      </td>
      <td class="num-soft">${row.current_played}</td>
      <td class="num">${row.current_points}</td>
      <td class="num">${fmtNum(row.expected_points)}</td>
      <td>${databar("title", row.title_prob)}</td>
      <td>${databar("ucl", row.ucl_prob ?? row.top4_prob)}</td>
      <td>${databar("europe", row.europe_prob || 0)}</td>
      <td>${databar("rel", row.relegation_prob)}</td>
    </tr>
  `;
}

function forecastCard(row) {
  const zone = zoneFor(row.projected_rank, forecastState.meta);
  const meta = forecastState.meta || {};
  const odd = (label, kind, value) => html`
    <div class="odd">
      <span class="odd-label">${escapeHtml(label)}</span>
      <span class="odd-track"><i class="${kind}" style="width:${Math.min(100, Math.round((value || 0) * 100))}%"></i></span>
      <span class="odd-value">${fmtPct(value)}</span>
    </div>
  `;
  return html`
    <article class="club-card" ${zone ? `data-zone="${zone}"` : ""}>
      <div class="club-card-top">
        <span class="pos">${row.projected_rank}</span>
        ${crestHtml(row.team, row.crest, "crest-sm")}
        <span class="name">${escapeHtml(row.team)}</span>
        <span class="pts">${fmtNum(row.expected_points)} <small>PROJ PTS</small></span>
      </div>
      <div class="club-card-odds">
        ${odd("Title", "title", row.title_prob)}
        ${odd(meta.ucl_label || "UCL", "ucl", row.ucl_prob ?? row.top4_prob)}
        ${odd("Relegation", "rel", row.relegation_prob)}
      </div>
    </article>
  `;
}

function sortedForecastRows() {
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
  return rows;
}

function paintForecast() {
  const rows = sortedForecastRows();
  const body = document.getElementById("forecast-body");
  if (body) {
    body.innerHTML = rows.map(forecastRow).join("");
    bindCrestFallbacks(body);
  }
  const cards = document.getElementById("forecast-cards");
  if (cards) {
    cards.innerHTML = rows.map(forecastCard).join("");
    bindCrestFallbacks(cards);
  }
  document.querySelectorAll("#forecast-table th[data-sort]").forEach((th) => {
    const active = th.dataset.sort === forecastState.sort;
    th.setAttribute("aria-sort", active ? (forecastState.dir === "asc" ? "ascending" : "descending") : "none");
    const arrow = th.querySelector(".arrow");
    if (arrow) arrow.textContent = active ? (forecastState.dir === "asc" ? "▲" : "▼") : "";
  });
}

function bindForecastSort() {
  document.querySelectorAll("#forecast-table th[data-sort] .th-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.parentElement.dataset.sort;
      if (forecastState.sort === key) {
        forecastState.dir = forecastState.dir === "desc" ? "asc" : "desc";
      } else {
        forecastState.sort = key;
        forecastState.dir = key === "team" ? "asc" : "desc";
      }
      paintForecast();
    });
  });
}

async function renderForecast() {
  const data = await fetchJson(`/api/forecast${qs({ league: currentLeague() })}`);
  setFootMeta(data.model, `Forecast generated ${formatStamp(data.generated_at)}`);

  /* The projected finishing order is the table's own ordering: rank by mean
     finishing position. The probabilities themselves are untouched. */
  const rows = [...(data.teams || [])].sort((a, b) => a.expected_position - b.expected_position);
  rows.forEach((row, index) => {
    row.projected_rank = index + 1;
  });

  forecastState = {
    rows,
    sort: null,
    dir: "desc",
    meta: {
      n_teams: data.n_teams,
      ucl_places: data.ucl_places,
      europe_places: data.europe_places,
      relegation_places: data.relegation_places,
      ucl_label: data.ucl_label,
      europa_label: data.europa_label,
      conference_label: data.conference_label,
    },
  };

  const leader = [...(data.teams || [])].sort((a, b) => b.title_prob - a.title_prob)[0];
  const chasers = [...(data.teams || [])]
    .sort((a, b) => b.title_prob - a.title_prob)
    .slice(1, 5)
    .filter((row) => row.title_prob > 0);

  const max = Math.max(leader ? leader.title_prob : 0.0001, 0.0001);
  const chaseRows = chasers
    .map(
      (row, index) => html`
        <div class="race-row">
          <span class="race-fill" style="--w:${Math.round((row.title_prob / max) * 100)}%"></span>
          <span class="race-pos">${index + 2}</span>
          ${crestHtml(row.team, row.crest, "crest-xs")}
          <span class="race-name">${escapeHtml(row.team)}</span>
          <span class="race-pct">${fmtPct(row.title_prob)}</span>
        </div>
      `,
    )
    .join("");

  const headCells = FORECAST_COLUMNS.map((col) => {
    if (!col.sortable) return `<th scope="col"><span class="th-flat">${col.label}</span></th>`;
    return html`<th scope="col" data-sort="${col.key}" aria-sort="none">
      <button type="button" class="th-btn">${col.label} <span class="arrow" aria-hidden="true"></span></button>
    </th>`;
  }).join("");

  const key = [
    ["ucl", data.ucl_label, `top ${data.ucl_places}`],
    ["europa", data.europa_label, ""],
    ["conference", data.conference_label, ""],
    ["releg", "Relegation", `bottom ${data.relegation_places}`],
  ]
    .filter(([, label]) => label)
    .map(
      ([zone, label, hint]) =>
        `<span><i style="background:var(--${zone})"></i>${escapeHtml(label)}${hint ? ` (${hint})` : ""}</span>`,
    )
    .join("");

  view.innerHTML = html`
    ${pageHead(
      `${leagueName(data)}, ${data.live_season}`,
      "Season forecast",
      `Every remaining fixture is replayed ${formatCount(data.n_sims)} times using the stored Home, Draw and Away probabilities, then the final table is counted up.`,
    )}

    <section class="race-lead">
      ${leader
        ? html`
            <article class="champ">
              ${crestHtml(leader.team, leader.crest, "crest-lg")}
              <div class="champ-body">
                <span class="champ-label">Most likely champion</span>
                <span class="champ-name">${escapeHtml(leader.team)}</span>
              </div>
              <span class="champ-pct">${fmtPct(leader.title_prob)}</span>
            </article>
          `
        : ""}
      ${chaseRows
        ? html`<article class="chase">
            <span class="champ-label">Chasing</span>
            <div class="race-list">${chaseRows}</div>
          </article>`
        : ""}
    </section>

    <section class="statbar section" aria-label="Simulation detail">
      ${statCell("Simulations", formatCount(data.n_sims), `Seed ${data.seed}, reproducible`)}
      ${statCell("Played", formatCount(data.n_completed), "Points already on the board")}
      ${statCell("Still to play", formatCount(data.n_remaining), "Sampled in every simulation")}
      ${statCell("Generated", formatStamp(data.generated_at), "Rebuilt by the pipeline", { small: true })}
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Projected table</h2>
        <span class="help">Ordered by projected finish</span>
      </div>
      <div class="table-scroll">
        <table class="standings" id="forecast-table">
          <caption class="vh">Projected final table with qualification and relegation probabilities</caption>
          <thead>
            <tr>${headCells}</tr>
          </thead>
          <tbody id="forecast-body"></tbody>
        </table>
      </div>
      <div class="standings-cards" id="forecast-cards"></div>
      <div class="zonekey">${key}</div>
    </section>

    <div class="detail-cols">
      <article class="panel">
        <h2>How the places are decided</h2>
        <p class="help">${escapeHtml(data.qualification_note || "")}</p>
      </article>
      <article class="panel">
        <h2>How ties are broken</h2>
        <p class="help">${escapeHtml(data.tiebreak)}</p>
      </article>
      <article class="panel">
        <h2>Early in the season</h2>
        <p class="help">${escapeHtml(data.early_season_note || "")}</p>
      </article>
    </div>
  `;

  paintForecast();
  bindForecastSort();
  bindCrestFallbacks(view);
}

/* ==========================================================================
   Match detail
   ========================================================================== */

function heroSide(name, crest, positionLabel) {
  return html`
    <div class="hero-side">
      ${crestHtml(name, crest, "crest-lg")}
      <p class="hero-team">${escapeHtml(name)}</p>
      ${positionLabel
        ? `<span class="rank-pill">${escapeHtml(positionLabel)}<span class="rank-suffix"> in the table</span></span>`
        : ""}
    </div>
  `;
}

function matchHero(match, { settled }) {
  const center = settled
    ? html`
        <span class="hero-score">${match.home_goals}-${match.away_goals}</span>
        <span class="hero-sub">${escapeHtml(match.actual_outcome ? `${match.actual_outcome} won` : "Full time")}</span>
      `
    : html`
        <span class="hero-kick">${escapeHtml(match.kickoff_time || "TBC")}</span>
        <span class="hero-sub">${escapeHtml(formatLongDay(match.kickoff_date))}</span>
      `;

  const statusTag = settled
    ? `<span class="tag">${escapeHtml(match.status || "Full time")}</span>`
    : `<span class="tag tag-live">${escapeHtml(match.status || "Upcoming")}</span>`;

  const prediction = match.has_prediction
    ? html`
        <div class="hero-prediction">
          <div class="hero-pred-head">
            <h2>${settled ? "What was predicted before kickoff" : "Win probability"}</h2>
            <div class="hero-tags">
              ${settled ? verdictBadge(match.correct) : noteTag(match.note)}
              <span class="tag">${escapeHtml(match.model_name || "Model")}</span>
            </div>
          </div>
          ${probBar(match, { large: true })}
        </div>
      `
    : html`
        <div class="hero-prediction">
          <p class="no-pred">
            No prediction was stored before this match. MatchLab only shows probabilities that were
            recorded ahead of kickoff, so nothing is filled in after the fact.
          </p>
        </div>
      `;

  return html`
    <article class="match-hero">
      <div class="hero-meta">
        <span>${escapeHtml(match.competition || "")}, ${formatKickoff(match.kickoff_date, match.kickoff_time)}</span>
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
  const href = meeting.detail_path
    ? withLeague(meeting.detail_path)
    : withLeague(`/results/${meeting.match_id}`);
  /* result_code is the stored integer class: 0 away, 1 draw, 2 home. */
  const side = { 0: "away", 1: "draw", 2: "home" }[meeting.result_code] || "";
  return html`
    <a class="h2h-row" data-link href="${href}">
      <span class="h2h-date">${formatShortDate(meeting.kickoff_date)}</span>
      <span class="h2h-teams">
        ${crestHtml(meeting.home_team, meeting.home_crest, "crest-xs")}
        <span class="nm">${escapeHtml(meeting.home_team)}</span>
        <span class="sc">${meeting.home_goals}-${meeting.away_goals}</span>
        <span class="nm">${escapeHtml(meeting.away_team)}</span>
        ${crestHtml(meeting.away_team, meeting.away_crest, "crest-xs")}
      </span>
      <span class="h2h-out ${side}">${escapeHtml(meeting.result_label || "")}</span>
    </a>
  `;
}

function h2hPanel(data) {
  const meetings = data.head_to_head || [];
  return html`
    <article class="panel">
      <h2>Previous meetings</h2>
      <p class="help">
        ${meetings.length
          ? `The last ${meetings.length} time${meetings.length === 1 ? "" : "s"} these clubs met in this competition before this match.`
          : "These clubs have not met before in the history MatchLab has ingested for this competition."}
      </p>
      ${meetings.length ? `<div class="h2h-list">${meetings.map(h2hRow).join("")}</div>` : ""}
    </article>
  `;
}

function statsPanel(match, stats, available) {
  if (!available || !stats.length) {
    return html`
      <article class="panel">
        <h2>Match stats</h2>
        <p class="help">Detailed stats are not available for this match in the source data.</p>
      </article>
    `;
  }
  const rows = stats
    .map((stat) => {
      const total = (Number(stat.home) || 0) + (Number(stat.away) || 0);
      const homeShare = total > 0 ? (Number(stat.home) / total) * 100 : 50;
      const flat = total === 0;
      return html`
        <div class="statrow">
          <div class="statrow-top">
            <span class="n ${stat.leader === "home" ? "lead" : ""}">${stat.home}</span>
            <span class="label">${escapeHtml(stat.label)}</span>
            <span class="n ${stat.leader === "away" ? "lead" : ""}">${stat.away}</span>
          </div>
          <div class="stat-track">
            <i class="${flat ? "flat" : "home"}" style="width:${flat ? 50 : homeShare}%"></i>
            <i class="${flat ? "flat" : "away"}" style="width:${flat ? 50 : 100 - homeShare}%"></i>
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

function chanceGiven(match) {
  if (match.actual_outcome === "Draw") return match.p_draw_pct;
  if (match.actual_outcome === match.home_team) return match.p_home_pct;
  return match.p_away_pct;
}

async function renderMatchDetail(matchId) {
  const data = await fetchJson(`/api/matches/${matchId}`);
  const league = (data.competition && data.competition.code) || currentLeague();
  rememberLeague(league);
  const canonical = `${data.kind === "result" ? "/results" : "/upcoming"}/${matchId}?league=${league}`;
  if (`${pathOf()}${window.location.search}` !== canonical) {
    window.history.replaceState({}, "", canonical);
  }
  setNav();

  const match = data.match;
  const settled = data.kind === "result";
  setFootMeta(data.model);

  const verdictPanel =
    settled && match.has_prediction
      ? html`
        <article class="panel">
          <h2>Prediction against the result</h2>
          ${html`
                <p class="help">What the model called before kickoff, and what actually happened.</p>
                <div class="metric-grid">
                  <div class="metric"><span>Called</span><strong class="name">${escapeHtml(match.predicted_outcome)}</strong></div>
                  <div class="metric"><span>Result</span><strong class="name">${escapeHtml(match.actual_outcome || NONE)}</strong></div>
                  <div class="metric"><span>Odds it gave that</span><strong>${chanceGiven(match)}%</strong></div>
                </div>
                <p class="help" style="margin-top:var(--sp-4)">
                  These probabilities were frozen at ${formatStamp(match.predicted_at)} and have not been
                  touched since the result came in.
                </p>
              `}
        </article>
      `
      : "";

  const secondary = settled
    ? `<div class="detail-cols">${statsPanel(match, data.stats || [], data.stats_available)}${verdictPanel}</div>
       <div class="detail-cols">${h2hPanel(data)}</div>`
    : `<div class="detail-cols">${h2hPanel(data)}</div>`;

  view.innerHTML = html`
    <p class="back-row">
      <a class="pill" data-link href="${withLeague(settled ? "/results" : "/upcoming")}">
        Back to ${settled ? "results" : "fixtures"}
      </a>
    </p>
    ${matchHero(match, { settled })} ${secondary}
  `;
  bindCrestFallbacks(view);
}

/* ==========================================================================
   Method
   ========================================================================== */

async function renderAbout() {
  const data = await fetchJson(`/api/about${qs({ league: currentLeague() })}`);
  const test = data.test || {};
  setFootMeta(data.model);

  const folds = (data.walkforward_folds || [])
    .map(
      (fold) => html`
        <div class="fold-row">
          <span class="fold-name">${escapeHtml(fold.valid_season)}</span>
          <span class="help">trained on everything up to ${escapeHtml(fold.train_through)}</span>
        </div>
      `,
    )
    .join("");

  view.innerHTML = html`
    ${pageHead(
      leagueName(data),
      "How MatchLab predicts",
      "Every probability on this site comes from one model per league, chosen on seasons it had never seen, and never trained on a match it is asked to predict.",
    )}

    <section class="statbar" aria-label="Model at a glance">
      ${statCell("Holdout accuracy", test.accuracy != null ? pct(test.accuracy) : NONE, `${data.test_season}, scored once`, { accent: true })}
      ${statCell("Holdout matches", formatCount(test.n ?? 0), "Never used to pick the model")}
      ${statCell("Selection score", data.walkforward_mean_log_loss != null ? Number(data.walkforward_mean_log_loss).toFixed(3) : NONE, `Mean log loss over ${(data.walkforward_folds || []).length} unseen seasons`)}
      ${statCell("Model", data.model.model_name, `Feature set ${data.feature_version}`, { small: true })}
    </section>

    <div class="about-grid section">
      <article class="panel">
        <h2>What it predicts</h2>
        <p class="help">
          One probability each for a home win, a draw and an away win. The three always add up to
          100%, and the highest of them is what MatchLab calls as the likely result. It does not
          predict scorelines.
        </p>
        <ul class="plain-list">
          <li>Probabilities are written down before kickoff and never changed afterwards.</li>
          <li>Matches played before MatchLab covered a league simply show no prediction.</li>
          <li>Every league has its own model, trained only on that competition.</li>
        </ul>
      </article>

      <article class="panel">
        <h2>What it looks at</h2>
        <ul class="feature-list">
          ${data.features
            .map((item) => `<li><b>${escapeHtml(item.title)}.</b> ${escapeHtml(item.detail)}</li>`)
            .join("")}
        </ul>
      </article>

      <article class="panel">
        <h2>Why it can be trusted</h2>
        <p class="help">${escapeHtml(data.leakage_note)}</p>
        <ul class="plain-list">
          ${(data.history ? [data.history.ingest, data.history.walkforward, data.history.production_train, data.history.test, data.history.live] : [])
            .filter(Boolean)
            .map((line) => `<li>${escapeHtml(line)}</li>`)
            .join("")}
        </ul>
      </article>

      <article class="panel">
        <h2>Where it falls down</h2>
        <p class="help">${escapeHtml(data.draw_limitation)}</p>
      </article>

      <article class="panel">
        <h2>How the model was chosen</h2>
        <p class="help">
          Each candidate was retrained repeatedly, always on older seasons only, then scored on the
          following season it had never seen. The one with the best average score won.
        </p>
        <div class="fold-list">${folds}</div>
      </article>

      <article class="panel">
        <h2>The numbers behind it</h2>
        <p class="help">For anyone who wants the detail. These are diagnostics, not the headline.</p>
        <dl class="spec">
          <dt>Selection</dt>
          <dd>${escapeHtml(data.selection_reason || "")}</dd>
          <dt>Holdout season</dt>
          <dd>${escapeHtml(data.test_season)}</dd>
          <dt>Holdout log loss</dt>
          <dd>${test.log_loss != null ? Number(test.log_loss).toFixed(3) : NONE}</dd>
          <dt>Holdout F1</dt>
          <dd>
            home ${test.f1_home != null ? Number(test.f1_home).toFixed(3) : NONE}, draw
            ${test.f1_draw != null ? Number(test.f1_draw).toFixed(3) : NONE}, away
            ${test.f1_away != null ? Number(test.f1_away).toFixed(3) : NONE}
          </dd>
          <dt>Walk-forward accuracy</dt>
          <dd>${data.walkforward_mean_accuracy != null ? pct(data.walkforward_mean_accuracy) : NONE}</dd>
        </dl>
      </article>
    </div>
  `;
}

/* ==========================================================================
   Router
   ========================================================================== */

const routes = {
  "/": renderOverview,
  "/upcoming": renderUpcoming,
  "/results": renderResults,
  "/forecast": renderForecast,
  "/performance": renderPerformance,
  "/about": renderAbout,
};

async function render() {
  closeLeagueMenu();
  rememberLeague(currentLeague());
  pinLeagueToUrl();
  setNav();
  try {
    const detail = matchDetailRoute();
    if (detail) {
      await renderMatchDetail(detail.id);
      return;
    }
    const route = routes[pathOf()] || renderOverview;
    await route();
  } catch (error) {
    view.innerHTML = html`
      ${pageHead("", "This page could not load", escapeHtml(error.message || "Could not load this page from the database."))}
      <p class="back-row"><a class="btn btn-primary" data-link href="${withLeague("/")}">Back to the overview</a></p>
    `;
  }
}

function go(href) {
  const url = new URL(href, window.location.origin);
  window.history.pushState({}, "", `${url.pathname}${url.search}`);
  window.scrollTo({ top: 0, behavior: "instant" });
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

/* Load the competition catalog once, so the league menu and the forecast
   zones both come from the backend rather than a second copy in the client. */
async function boot() {
  try {
    const payload = await fetchJson("/api/competitions");
    catalog = {
      competitions: payload.competitions || [],
      default: payload.default || DEFAULT_LEAGUE,
      byCode: Object.fromEntries((payload.competitions || []).map((spec) => [spec.code, spec])),
    };
  } catch (error) {
    /* The league menu stays empty, but every page still renders. */
  }
  await render();
}

boot();
