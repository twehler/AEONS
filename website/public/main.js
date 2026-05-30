/* ===========================================================
   AEONS — front-end interactivity
   - drifting "cell" background canvas
   - scroll reveals + progress bar + nav state
   - count-up statistics
   - Darwin / Lamarck mode toggle (recolours the whole page)
   =========================================================== */

(() => {
  "use strict";

  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  /* ---------- Nav state + reading progress ---------- */
  const nav = document.getElementById("nav");
  const progress = document.getElementById("progress");

  function onScroll() {
    const y = window.scrollY;
    nav.classList.toggle("scrolled", y > 40);
    const h = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (h > 0 ? (y / h) * 100 : 0) + "%";
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- Scroll reveals ---------- */
  const revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reduceMotion) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("in");
            if (e.target.classList.contains("stat")) countUp(e.target);
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.16 }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("in"));
    document.querySelectorAll(".stat").forEach(countUp);
  }

  /* ---------- Count-up stats ---------- */
  function countUp(el) {
    const target = parseFloat(el.dataset.count || "0");
    const suffix = el.dataset.suffix || "";
    const numEl = el.querySelector(".stat-num");
    if (!numEl) return;
    if (reduceMotion) {
      numEl.textContent = target + suffix;
      return;
    }
    const dur = 1200;
    const start = performance.now();
    function step(now) {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      numEl.textContent = Math.round(target * eased) + suffix;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ---------- Darwin / Lamarck mode toggle ---------- */
  const toggle = document.querySelector(".mode-toggle");
  const modeBtns = document.querySelectorAll(".mode-btn");
  const panels = document.querySelectorAll("[data-mode-panel]");
  const root = document.documentElement;

  const MODE_THEME = {
    darwin: { accent: "#4be38f", accent2: "#9d7bff" }, // green-led
    lamarck: { accent: "#9d7bff", accent2: "#ff8fae" }, // violet-led
  };

  function setMode(mode) {
    modeBtns.forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-selected", String(on));
    });
    panels.forEach((p) => {
      p.hidden = p.dataset.modePanel !== mode;
    });
    if (toggle) toggle.dataset.active = mode;
    const theme = MODE_THEME[mode];
    root.style.setProperty("--accent", theme.accent);
    root.style.setProperty("--accent-2", theme.accent2);
    bg.setAccent(theme.accent, theme.accent2);
  }
  modeBtns.forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode))
  );

  /* ===========================================================
     Background canvas — drifting bioluminescent cells
     =========================================================== */
  const bg = (() => {
    const canvas = document.getElementById("bg-canvas");
    const ctx = canvas.getContext("2d");
    let w, h, dpr, cells, accent = "#4be38f", accent2 = "#9d7bff";
    let raf = null;

    function hexToRgb(hex) {
      const n = parseInt(hex.slice(1), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.width = innerWidth * dpr;
      h = canvas.height = innerHeight * dpr;
      canvas.style.width = innerWidth + "px";
      canvas.style.height = innerHeight + "px";
    }

    function seed() {
      const count = Math.round(
        Math.min(70, (innerWidth * innerHeight) / 26000)
      );
      cells = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: (6 + Math.random() * 16) * dpr,
        vx: (Math.random() - 0.5) * 0.22 * dpr,
        vy: (Math.random() - 0.5) * 0.22 * dpr,
        green: Math.random() > 0.42,
        ph: Math.random() * Math.PI * 2,
      }));
    }

    function frame(t) {
      ctx.clearRect(0, 0, w, h);
      const linkDist = 130 * dpr;

      // connective filaments
      for (let i = 0; i < cells.length; i++) {
        for (let j = i + 1; j < cells.length; j++) {
          const a = cells[i], b = cells[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < linkDist * linkDist) {
            const alpha = (1 - Math.sqrt(d2) / linkDist) * 0.12;
            ctx.strokeStyle = `rgba(120,200,170,${alpha})`;
            ctx.lineWidth = dpr;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      const [gr, gg, gb] = hexToRgb(accent);
      const [rr, rg, rb] = hexToRgb("#ff6b6b");

      cells.forEach((c) => {
        c.x += c.vx;
        c.y += c.vy;
        if (c.x < -40) c.x = w + 40;
        if (c.x > w + 40) c.x = -40;
        if (c.y < -40) c.y = h + 40;
        if (c.y > h + 40) c.y = -40;

        const pulse = 0.5 + 0.5 * Math.sin(t * 0.001 + c.ph);
        const R = c.green ? gr : rr;
        const G = c.green ? gg : rg;
        const B = c.green ? gb : rb;

        const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, c.r);
        grad.addColorStop(0, `rgba(${R},${G},${B},${0.5 + pulse * 0.3})`);
        grad.addColorStop(1, `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
        ctx.fill();

        // nucleus
        ctx.fillStyle = `rgba(${R},${G},${B},${0.7 + pulse * 0.3})`;
        ctx.beginPath();
        ctx.arc(c.x, c.y, c.r * 0.22, 0, Math.PI * 2);
        ctx.fill();
      });

      raf = requestAnimationFrame(frame);
    }

    function start() {
      resize();
      seed();
      if (reduceMotion) {
        frame(0);
        cancelAnimationFrame(raf);
        return;
      }
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    }

    let rt;
    window.addEventListener("resize", () => {
      clearTimeout(rt);
      rt = setTimeout(() => {
        resize();
        seed();
      }, 200);
    });

    return {
      start,
      setAccent(a, a2) {
        accent = a;
        accent2 = a2;
      },
    };
  })();

  bg.start();
  setMode("darwin");
})();
