/*
 * viewer.js — charts and log viewer for the run dashboard (OBS-11/OBS-12).
 *
 * Single entry point: window.obsRender(root, data) with
 * data = {header, metrics, events, live}. Snapshot pages inline the data
 * arrays and leave live undefined; the live page (OBS-12) passes
 * live = {metricsUrl, eventsUrl, pollMs} and the viewer polls for appended
 * JSONL lines, extending the charts and tailing the log.
 */
(function () {
  "use strict";

  var CHART_GROUPS = [
    { title: "Loss", keys: ["loss", "train/loss", "eval/holdout_loss"] },
    { title: "Learning rate", keys: ["lr", "train/lr"] },
    {
      title: "Throughput (tokens/s)",
      keys: ["tokens_per_s", "train/tokens_per_s"],
    },
    {
      title: "Device memory (GB)",
      keys: ["gpu_mem_allocated_gb", "gpu_vram_used_gb"],
    },
    // Host resident set size, written by the rust harness. Additive: a
    // training run carries no rss_gb and renders this group empty, the same
    // way a CPU-only harness run renders the device-memory group above.
    { title: "Process memory (GB)", keys: ["rss_gb"] },
    // Per-rung wall clock, written by the ladder. Its step axis is a rung
    // index rather than a time step -- five ordered points, not a series --
    // which is why only the duration is charted: the rungs' own measures are
    // different quantities in different units, and a line through them would
    // mean nothing. They stay in metrics.jsonl for jq.
    { title: "Rung duration (s)", keys: ["rung_duration_s"] },
  ];
  var TRACKED = {};
  CHART_GROUPS.forEach(function (group) {
    group.keys.forEach(function (key) {
      TRACKED[key] = true;
    });
  });
  var SERIES_COLORS = ["#2c7fb8", "#d95f0e", "#31a354"];
  var MARKER_COLORS = { checkpoint: "#31a354", watchdog: "#d7301f" };
  var LEVEL_RANK = { debug: 0, info: 1, warning: 2, error: 3 };
  var EVENT_BASE_FIELDS = [
    "ts",
    "level",
    "run_id",
    "cli",
    "command",
    "pid",
    "event",
    "msg",
  ];

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function newState() {
    var state = {
      header: null,
      steps: [],
      series: {},
      markers: [],
      events: [],
      charts: {},
      pinned: true,
    };
    Object.keys(TRACKED).forEach(function (key) {
      state.series[key] = [];
    });
    return state;
  }

  function forEachSeries(state, fn) {
    Object.keys(state.series).forEach(function (key) {
      fn(state.series[key]);
    });
  }

  /* Index of step in the sorted axis, inserting (null-padded across every
   * series) when unseen. Live appends hit the fast push path. */
  function indexForStep(state, step) {
    var steps = state.steps;
    var n = steps.length;
    if (n === 0 || step > steps[n - 1]) {
      steps.push(step);
      forEachSeries(state, function (arr) {
        arr.push(null);
      });
      return n;
    }
    var lo = 0;
    var hi = n - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (steps[mid] === step) return mid;
      if (steps[mid] < step) lo = mid + 1;
      else hi = mid - 1;
    }
    steps.splice(lo, 0, step);
    forEachSeries(state, function (arr) {
      arr.splice(lo, 0, null);
    });
    return lo;
  }

  function ingestMetric(state, rec) {
    if (!rec || typeof rec !== "object") return;
    if (rec.report === "run_metrics") {
      /* first header wins, matching the Python reader: a duplicate
       * header mid-file must not flip the strip to a different run */
      if (!state.header) state.header = rec;
      return;
    }
    if (typeof rec.step !== "number") return;
    var idx = indexForStep(state, rec.step);
    Object.keys(rec).forEach(function (key) {
      if (key === "step" || key === "ts") return;
      if (key === "checkpoint_saved") {
        state.markers.push({ step: rec.step, kind: "checkpoint" });
        return;
      }
      if (key.indexOf("watchdog/") === 0) {
        state.markers.push({ step: rec.step, kind: "watchdog", label: key });
        return;
      }
      if (TRACKED[key]) state.series[key][idx] = rec[key];
    });
  }

  function appendRecords(state, metricRecords, eventRecords) {
    (metricRecords || []).forEach(function (rec) {
      ingestMetric(state, rec);
    });
    (eventRecords || []).forEach(function (rec) {
      if (rec && typeof rec === "object") state.events.push(rec);
    });
  }

  function drawMarkers(state) {
    return function (u) {
      var ctx = u.ctx;
      var px = uPlot.pxRatio || 1;
      ctx.save();
      /* explicit width: the context otherwise inherits whatever uPlot
       * last stroked, varying by draw order and device pixel ratio */
      ctx.lineWidth = px;
      state.markers.forEach(function (m) {
        var x = u.valToPos(m.step, "x", true);
        if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) return;
        ctx.strokeStyle = MARKER_COLORS[m.kind] || "#999";
        ctx.setLineDash(m.kind === "checkpoint" ? [4 * px, 4 * px] : []);
        ctx.beginPath();
        ctx.moveTo(x, u.bbox.top);
        ctx.lineTo(x, u.bbox.top + u.bbox.height);
        ctx.stroke();
      });
      ctx.restore();
    };
  }

  function activeKeys(state, group) {
    return group.keys.filter(function (key) {
      return state.series[key].some(function (v) {
        return v !== null && v !== undefined;
      });
    });
  }

  function chartData(state, keys) {
    return [state.steps].concat(
      keys.map(function (key) {
        return state.series[key];
      })
    );
  }

  /* Y-axis labels with as many decimals as the tick spacing needs, so a
   * narrow range gets distinguishable labels. uPlot's default rounds to a
   * fixed precision; RHX.4's process-memory chart, flat within a few MB of
   * 14.24 GB, showed six ticks all reading 14.242. Capped at six decimals so a
   * degenerate spacing cannot produce an unreadable axis. */
  function axisLabels(u, splits) {
    var gap = Infinity;
    for (var i = 1; i < splits.length; i += 1) {
      var d = Math.abs(splits[i] - splits[i - 1]);
      if (d > 0 && d < gap) gap = d;
    }
    var decimals = 0;
    if (isFinite(gap) && gap < 1) {
      decimals = Math.min(6, Math.ceil(-Math.log10(gap)));
    }
    return splits.map(function (v) {
      return v.toFixed(decimals);
    });
  }

  function chartWidth(chartsRoot) {
    return Math.min((chartsRoot.clientWidth || 960) - 18, 1100);
  }

  function refreshCharts(state, chartsRoot) {
    CHART_GROUPS.forEach(function (group) {
      var entry = state.charts[group.title];
      var keys = activeKeys(state, group);
      /* A group with no series is a group this run does not have: a
       * training run has no rss_gb, a harness run no loss. Its holder is
       * created up front so groups keep their order when a series arrives
       * live, and hidden until then; RHX.4 found the four empty frames a
       * harness page otherwise shows. */
      entry.holder.style.display = keys.length ? "" : "none";
      if (!keys.length) return;
      if (entry.plot && entry.keys.join(" ") === keys.join(" ")) {
        /* a user zoom survives new data: resetScales only when unzoomed,
         * so the live page keeps extending for a reader at full range */
        entry.plot.setData(chartData(state, keys), !entry.zoomed);
        return;
      }
      if (entry.plot) entry.plot.destroy();
      entry.keys = keys;
      entry.zoomed = false;
      var hooks = {
        setScale: [
          function (u, scaleKey) {
            if (scaleKey !== "x") return;
            var xs = u.data[0];
            if (!xs.length) return;
            var scale = u.scales.x;
            entry.zoomed = scale.min > xs[0] || scale.max < xs[xs.length - 1];
          },
        ],
      };
      if (group.title === "Loss") hooks.draw = [drawMarkers(state)];
      var opts = {
        title: group.title,
        width: chartWidth(chartsRoot),
        height: 240,
        scales: { x: { time: false } },
        axes: [{}, { values: axisLabels }],
        series: [{ label: "step" }].concat(
          keys.map(function (key, i) {
            return {
              label: key,
              stroke: SERIES_COLORS[i % SERIES_COLORS.length],
              width: 1,
              /* sparse series (eval points, watchdog samples) sit on the
               * union step axis with nulls between their points; without
               * spanGaps uPlot breaks the line at every null and a series
               * sparser than the axis never draws a line at all */
              spanGaps: true,
            };
          })
        ),
        hooks: hooks,
      };
      entry.plot = new uPlot(opts, chartData(state, keys), entry.holder);
    });
  }

  /* The strip states the facts the emitter's own header states, keyed by
   * the envelope's `cli`. The harness merges RunHeader::fields() into its
   * envelope (schemas/run_metrics.json, header.extra_fields), so its strip is
   * the terminal's header: model, hash, fold and build. Until RHX.4 (the
   * 2026-09-12 run) the strip read only the training keys and showed a dash
   * for everything but the run id on every harness page. */
  function headerFields(h) {
    if (h.cli === "harness") {
      var fold =
        h.dot || h.batch || h.threads
          ? "dot=" + h.dot + " batch=" + h.batch + " threads=" + h.threads
          : "-";
      var build = h.binary_version
        ? h.binary_version + " git " + (h.git_sha || "unknown")
        : "-";
      return [
        ["run", h.run_id || "-"],
        ["command", h.command || "-"],
        ["model", h.model || "-"],
        ["hash", h.model_hash || "-"],
        ["fold", fold],
        ["build", build],
      ];
    }
    return [
      ["run", h.run_id || "-"],
      ["tier", h.tier || "-"],
      ["model", h.model_id || "-"],
      ["recipe", (h.recipe_hash || "").slice(0, 12) || "-"],
      ["schedule", h.schedule ? JSON.stringify(h.schedule) : "-"],
    ];
  }

  function refreshHeader(state, node) {
    node.textContent = "";
    var h = state.header;
    if (!h) {
      node.appendChild(el("span", "muted", "waiting for run_metrics header"));
      return;
    }
    headerFields(h).forEach(function (field) {
      var item = el("span", "hdr-item");
      item.appendChild(el("span", "hdr-key", field[0]));
      item.appendChild(el("span", "hdr-val", field[1]));
      node.appendChild(item);
    });
  }

  function logRow(rec, minRank, needle) {
    var rank = LEVEL_RANK[rec.level];
    /* unknown or missing levels always show: hiding a future "critical"
     * behind a severity filter would invert the failure mode */
    if (rank === undefined) rank = Infinity;
    if (rank < minRank) return null;
    var extra = {};
    Object.keys(rec).forEach(function (key) {
      if (EVENT_BASE_FIELDS.indexOf(key) < 0) extra[key] = rec[key];
    });
    var line =
      (rec.ts || "") +
      " " +
      (rec.level || "") +
      " " +
      (rec.event || "") +
      " " +
      (rec.msg || "") +
      (Object.keys(extra).length ? " " + JSON.stringify(extra) : "");
    if (needle && line.toLowerCase().indexOf(needle) < 0) return null;
    return el("div", "log-row log-" + (rec.level || "info"), line);
  }

  function refreshLog(state, ui) {
    var minRank = LEVEL_RANK[ui.levelSelect.value] || 0;
    var needle = ui.searchInput.value.toLowerCase();
    var sig = minRank + "|" + needle;
    var start = ui.renderedCount || 0;
    if (ui.filterSig !== sig) {
      /* filter changed: full rebuild, preserving the reading position of
       * an unpinned viewer as closely as the new content allows */
      var prevTop = ui.logBody.scrollTop;
      ui.filterSig = sig;
      ui.logBody.textContent = "";
      start = 0;
      if (!state.pinned) ui.logBody.scrollTop = prevTop;
    }
    /* steady state appends only the new records: a full per-poll rebuild
     * scaled with total events and eventually locked the page, and
     * clearing the DOM yanked an unpinned reader back to the top */
    for (var i = start; i < state.events.length; i++) {
      var row = logRow(state.events[i], minRank, needle);
      if (row) ui.logBody.appendChild(row);
    }
    ui.renderedCount = state.events.length;
    if (state.pinned) ui.logBody.scrollTop = ui.logBody.scrollHeight;
  }

  /* Poll the serve endpoints (OBS-12) with client-owned byte offsets.
   * Polling continues after cmd.end — a finished run simply stops
   * producing lines — and a fetch failure is retried on the next tick. */
  function startPolling(state, live, refreshAll) {
    var offsets = { metrics: 0, events: 0 };

    function pull(url, offsetKey, ingest) {
      return fetch(url + "?offset=" + offsets[offsetKey])
        .then(function (resp) {
          return resp.json();
        })
        .then(function (payload) {
          offsets[offsetKey] = payload.offset;
          var parsed = [];
          (payload.lines || []).forEach(function (line) {
            try {
              parsed.push(JSON.parse(line));
            } catch (err) {
              /* skip unparseable lines */
            }
          });
          ingest(parsed);
          return parsed.length;
        });
    }

    function poll() {
      /* each endpoint fails independently: one bad fetch must not leave
       * the other's already-ingested records unrendered */
      Promise.all([
        pull(live.metricsUrl, "metrics", function (records) {
          appendRecords(state, records, []);
        }).catch(function () {
          return 0; /* retried at the same offset next tick */
        }),
        pull(live.eventsUrl, "events", function (records) {
          appendRecords(state, [], records);
        }).catch(function () {
          return 0;
        }),
      ])
        .then(function (counts) {
          if (counts[0] || counts[1]) refreshAll();
        })
        .finally(function () {
          /* chained timeout, not setInterval: a poll slower than the
           * interval must finish before the next fires, or two in-flight
           * polls read the same offset and ingest everything twice */
          setTimeout(poll, live.pollMs || 2000);
        });
    }

    poll();
  }

  function render(root, data) {
    var state = newState();

    root.textContent = "";
    var headerNode = el("div", "header");
    var chartsRoot = el("div", "charts");
    var logSection = el("div", "log");
    var controls = el("div", "log-controls");
    var levelSelect = el("select");
    ["debug", "info", "warning", "error"].forEach(function (level) {
      var opt = el("option", null, level);
      opt.value = level;
      levelSelect.appendChild(opt);
    });
    levelSelect.value = "debug";
    var searchInput = el("input");
    searchInput.type = "search";
    searchInput.placeholder = "filter log records";
    var logBody = el("div", "log-body");
    controls.appendChild(levelSelect);
    controls.appendChild(searchInput);
    logSection.appendChild(el("div", "log-title", "Run log"));
    logSection.appendChild(controls);
    logSection.appendChild(logBody);
    root.appendChild(headerNode);
    root.appendChild(chartsRoot);
    root.appendChild(logSection);

    CHART_GROUPS.forEach(function (group) {
      var holder = el("div", "chart");
      chartsRoot.appendChild(holder);
      state.charts[group.title] = { holder: holder, plot: null, keys: [] };
    });

    var ui = {
      levelSelect: levelSelect,
      searchInput: searchInput,
      logBody: logBody,
    };

    function refreshAll() {
      refreshHeader(state, headerNode);
      refreshCharts(state, chartsRoot);
      refreshLog(state, ui);
    }

    if (data.header) state.header = data.header;
    appendRecords(state, data.metrics, data.events);
    refreshAll();

    logBody.addEventListener("scroll", function () {
      state.pinned =
        logBody.scrollTop + logBody.clientHeight >= logBody.scrollHeight - 4;
    });
    levelSelect.addEventListener("change", function () {
      refreshLog(state, ui);
    });
    searchInput.addEventListener("input", function () {
      refreshLog(state, ui);
    });
    window.addEventListener("resize", function () {
      Object.keys(state.charts).forEach(function (title) {
        var entry = state.charts[title];
        if (entry.plot) {
          entry.plot.setSize({ width: chartWidth(chartsRoot), height: 240 });
        }
      });
    });

    if (data.live) startPolling(state, data.live, refreshAll);

    /* introspection handle for the browser verification harness */
    root.__obsState = state;

    return { state: state, refreshAll: refreshAll };
  }

  window.obsRender = render;
})();
