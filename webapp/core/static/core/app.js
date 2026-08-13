// Small shared behaviors used across pages. No framework, no build step —
// kept consistent with the inline polling scripts already used on the
// extraction/training detail pages.

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("nav-toggle");
  const nav = document.getElementById("site-nav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  document.querySelectorAll(".messages .msg-close").forEach((btn) => {
    btn.addEventListener("click", () => {
      const li = btn.closest("li");
      if (li) li.remove();
    });
  });

  document.querySelectorAll("[data-copy-target]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = document.getElementById(btn.getAttribute("data-copy-target"));
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent);
        const original = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = original; }, 1500);
      } catch (err) {
        // clipboard API unavailable (e.g. insecure context) — fail quietly,
        // the log is still selectable/copyable by hand
      }
    });
  });

  const filterInput = document.querySelector("[data-table-filter]");
  if (filterInput) {
    const table = document.querySelector(filterInput.getAttribute("data-table-filter"));
    if (table) {
      const rows = Array.from(table.querySelectorAll("tbody tr"));
      filterInput.addEventListener("input", () => {
        const q = filterInput.value.trim().toLowerCase();
        rows.forEach((row) => {
          row.style.display = !q || row.textContent.toLowerCase().includes(q) ? "" : "none";
        });
      });
    }
  }

  // Video/folder/manifest picker: turns a plain text input into a
  // searchable, paginated browser over values already seen on disk (video
  // paths, video-holding folders, manifest files), supplied as a
  // json_script sibling named by the input's data-picker attribute. Typing
  // filters by label (substring, case-insensitive); free text that matches
  // nothing still submits as-is, so a brand-new path always works.
  const PICKER_PAGE_SIZE = 8;
  document.querySelectorAll("[data-picker]").forEach((input) => {
    const dataEl = document.getElementById(input.getAttribute("data-picker"));
    if (!dataEl) return;
    let raw;
    try {
      raw = JSON.parse(dataEl.textContent);
    } catch (err) {
      return;
    }
    if (!Array.isArray(raw) || !raw.length) return;

    const valueKey = input.getAttribute("data-picker-value-key") || "value";
    const labelKey = input.getAttribute("data-picker-label-key") || "label";
    const items = raw.map((entry) =>
      typeof entry === "string" ? { value: entry, label: entry } : { value: entry[valueKey], label: entry[labelKey] }
    );

    const wrap = document.createElement("div");
    wrap.className = "picker-wrap";
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);

    const panel = document.createElement("div");
    panel.className = "picker-panel";
    panel.hidden = true;
    wrap.appendChild(panel);

    let page = 0;
    let rowEls = [];
    let activeIndex = -1;

    const matching = () => {
      const q = input.value.trim().toLowerCase();
      return q ? items.filter((it) => it.label.toLowerCase().includes(q)) : items;
    };

    function setActive(index) {
      if (rowEls[activeIndex]) {
        rowEls[activeIndex].classList.remove("picker-item-active");
        rowEls[activeIndex].removeAttribute("aria-selected");
      }
      activeIndex = index;
      const row = rowEls[activeIndex];
      if (row) {
        row.classList.add("picker-item-active");
        row.setAttribute("aria-selected", "true");
        row.scrollIntoView({ block: "nearest" });
      }
    }

    function selectItem(it) {
      input.value = it.value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      close();
      input.focus();
    }

    function render() {
      const results = matching();
      const pageCount = Math.max(1, Math.ceil(results.length / PICKER_PAGE_SIZE));
      page = Math.min(page, pageCount - 1);
      const start = page * PICKER_PAGE_SIZE;
      const pageItems = results.slice(start, start + PICKER_PAGE_SIZE);

      panel.innerHTML = "";
      rowEls = [];
      activeIndex = -1;
      if (!pageItems.length) {
        const empty = document.createElement("div");
        empty.className = "picker-empty";
        empty.textContent = "No matches — free text still works.";
        panel.appendChild(empty);
      }
      pageItems.forEach((it) => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "picker-item";
        row.setAttribute("role", "option");
        const main = document.createElement("span");
        main.className = "picker-item-label";
        main.textContent = it.label;
        row.appendChild(main);
        if (it.value !== it.label) {
          const sub = document.createElement("span");
          sub.className = "picker-item-value";
          sub.textContent = it.value;
          row.appendChild(sub);
        }
        row.addEventListener("mouseenter", () => setActive(rowEls.indexOf(row)));
        row.addEventListener("click", () => selectItem(it));
        panel.appendChild(row);
        rowEls.push(row);
      });

      if (results.length > PICKER_PAGE_SIZE) {
        const footer = document.createElement("div");
        footer.className = "picker-footer";

        const prev = document.createElement("button");
        prev.type = "button";
        prev.className = "picker-page-btn";
        prev.textContent = "‹ Prev";
        prev.disabled = page === 0;
        prev.addEventListener("click", (e) => {
          e.stopPropagation();
          page -= 1;
          render();
        });

        const status = document.createElement("span");
        status.className = "picker-page-status";
        status.textContent = `${start + 1}–${Math.min(start + PICKER_PAGE_SIZE, results.length)} of ${results.length}`;

        const next = document.createElement("button");
        next.type = "button";
        next.className = "picker-page-btn";
        next.textContent = "Next ›";
        next.disabled = page >= pageCount - 1;
        next.addEventListener("click", (e) => {
          e.stopPropagation();
          page += 1;
          render();
        });

        footer.appendChild(prev);
        footer.appendChild(status);
        footer.appendChild(next);
        panel.appendChild(footer);
      }
    }

    function open() {
      page = 0;
      render();
      panel.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }
    function close() {
      panel.hidden = true;
      activeIndex = -1;
      input.setAttribute("aria-expanded", "false");
    }

    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    panel.setAttribute("role", "listbox");

    input.addEventListener("focus", open);
    input.addEventListener("click", open);
    input.addEventListener("input", () => {
      page = 0;
      render();
      panel.hidden = false;
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        close();
        return;
      }
      if (panel.hidden && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
        open();
        return;
      }
      if (panel.hidden || !rowEls.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((activeIndex + 1) % rowEls.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((activeIndex - 1 + rowEls.length) % rowEls.length);
      } else if (e.key === "Enter" && activeIndex >= 0) {
        e.preventDefault();
        rowEls[activeIndex].click();
      }
    });
    document.addEventListener("click", (e) => {
      if (!wrap.contains(e.target)) close();
    });
  });

  // Live log polling for a running training/extraction run. The log <pre>
  // carries data-log-poll-url only while status == "running" (set by the
  // template); polling keeps going through transient fetch/JSON failures
  // (dev-server autoreload restart, a dropped connection) instead of dying
  // silently on the first error and leaving the log frozen with no
  // indication anything is wrong -- that used to happen and looked like
  // "the log just stops updating". A failed poll is shown via a "stale"
  // class on the status pill and retried on the same interval.
  const logEl = document.getElementById("log");
  const pollUrl = logEl && logEl.getAttribute("data-log-poll-url");
  if (logEl && pollUrl) {
    const pillEl = document.getElementById("status-pill");
    const countEl = document.getElementById("clip-count"); // extraction only, absent elsewhere
    // extraction only, absent elsewhere -- per-video progress bar/label
    const progressWrapEl = document.getElementById("progress-wrap");
    const progressTextEl = document.getElementById("progress-text");
    const progressFillEl = document.getElementById("progress-fill");
    const POLL_INTERVAL_MS = 3000;

    async function poll() {
      try {
        const res = await fetch(pollUrl);
        if (!res.ok) throw new Error(`log poll failed: HTTP ${res.status}`);
        const data = await res.json();
        if (pillEl) pillEl.classList.remove("status-pill-stale");
        logEl.textContent = data.log_tail;
        logEl.scrollTop = logEl.scrollHeight;
        if (countEl && "clip_count" in data) countEl.textContent = data.clip_count ?? "—";
        if (progressWrapEl && data.progress) {
          progressWrapEl.hidden = false;
          const { video, current, total } = data.progress;
          if (progressTextEl) {
            progressTextEl.textContent = "";
            progressTextEl.append("Processing ");
            const strong = document.createElement("strong");
            strong.textContent = video;
            progressTextEl.append(strong, ` — video ${current} of ${total}`);
          }
          if (progressFillEl) progressFillEl.style.width = `${Math.round((current / total) * 100)}%`;
        }
        if (data.status !== "running") {
          if (pillEl) {
            pillEl.textContent = data.status;
            pillEl.className = "status-pill status-" + data.status;
          }
          return;
        }
      } catch (err) {
        // Network blip or a mid-restart 500 -- keep polling rather than
        // leaving the page stuck showing a stale log with no explanation.
        if (pillEl) pillEl.classList.add("status-pill-stale");
      }
      setTimeout(poll, POLL_INTERVAL_MS);
    }
    setTimeout(poll, POLL_INTERVAL_MS);
  }

  // Collapsible "how it works" cards remember their open/closed state per
  // page across visits -- a returning user who's already read the onboarding
  // explainer can collapse it once and not see it re-expanded every reload.
  // localStorage can throw (private-mode Safari, disabled storage); on any
  // failure the card just stays at its default (open) state.
  document.querySelectorAll("details[data-persist-key]").forEach((details) => {
    const key = "ar-howto:" + details.getAttribute("data-persist-key");
    try {
      const saved = window.localStorage.getItem(key);
      if (saved === "closed") details.open = false;
      else if (saved === "open") details.open = true;
    } catch (err) {
      // ignore -- keep the template's default `open` state
    }
    details.addEventListener("toggle", () => {
      try {
        window.localStorage.setItem(key, details.open ? "open" : "closed");
      } catch (err) {
        // ignore -- state just won't persist this session
      }
    });
  });

  // Upload progress bar: progressive enhancement over a plain multipart POST
  // (large video files can take a while with zero feedback otherwise). Falls
  // back to the ordinary form submit if XHR/FormData aren't available; on
  // any XHR error it also falls back by submitting the form normally rather
  // than showing a stuck progress bar with no explanation.
  document.querySelectorAll("[data-upload-progress]").forEach((form) => {
    if (typeof XMLHttpRequest === "undefined" || typeof FormData === "undefined") return;
    const bar = form.querySelector("[data-upload-progress-bar]");
    const barFill = bar && bar.querySelector("span");
    const text = form.querySelector("[data-upload-progress-text]");
    const btn = form.querySelector("button[type=submit]");

    form.addEventListener("submit", (e) => {
      const fileInput = form.querySelector("input[type=file]");
      if (!fileInput || !fileInput.files.length) return; // let normal validation handle it
      e.preventDefault();

      const xhr = new XMLHttpRequest();
      xhr.open(form.method || "POST", form.action, true);
      if (btn) btn.disabled = true;
      if (bar) bar.hidden = false;
      if (text) {
        text.hidden = false;
        text.textContent = "Uploading…";
      }

      xhr.upload.addEventListener("progress", (ev) => {
        if (!ev.lengthComputable) return;
        const pct = Math.round((ev.loaded / ev.total) * 100);
        if (barFill) barFill.style.width = pct + "%";
        if (text) text.textContent = `Uploading… ${pct}%`;
      });
      xhr.addEventListener("load", () => {
        // Follow the server's redirect the same way a normal form submit
        // would, so Django messages on the destination page still show.
        window.location.href = xhr.responseURL || window.location.href;
      });
      xhr.addEventListener("error", () => {
        // Network failure -- fall back to a plain submit rather than leaving
        // the user stuck on a frozen progress bar.
        if (btn) btn.disabled = false;
        if (bar) bar.hidden = true;
        if (text) text.hidden = true;
        form.removeAttribute("data-upload-progress");
        form.submit();
      });

      xhr.send(new FormData(form));
    });
  });

  // Submit-once guard: disables the submit button and swaps its label the
  // moment a form is submitted, so a slow background-process launch (or a
  // double click) can't fire the same "start extraction"/"start training"
  // request twice. The form still submits normally either way.
  document.querySelectorAll("[data-submit-once]").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type=submit]");
      if (!btn) return;
      btn.disabled = true;
      btn.dataset.originalText = btn.textContent;
      btn.textContent = btn.getAttribute("data-submit-once") || "Starting…";
    });
  });

  // "Set = current time" buttons on the Label page's span editor — reads
  // currentTime off the video named by data-span-video and drops it into
  // the matching start/end number input, rounded to one decimal.
  const spanPicker = document.querySelector("[data-span-picker]");
  if (spanPicker) {
    const video = document.getElementById(spanPicker.getAttribute("data-span-video"));
    const startInput = spanPicker.querySelector("[data-span-start]");
    const endInput = spanPicker.querySelector("[data-span-end]");
    const setStartBtn = spanPicker.querySelector("[data-span-set-start]");
    const setEndBtn = spanPicker.querySelector("[data-span-set-end]");
    if (video && startInput && setStartBtn) {
      setStartBtn.addEventListener("click", () => {
        startInput.value = video.currentTime.toFixed(1);
      });
    }
    if (video && endInput && setEndBtn) {
      setEndBtn.addEventListener("click", () => {
        endInput.value = video.currentTime.toFixed(1);
      });
    }
  }
});
