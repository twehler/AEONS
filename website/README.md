# AEONS — Website

A small, public-facing presentation site for the **AEONS** evolution simulator
(*Accelerated Evolution, Open-Naturalistic Simulator*). It documents the project
for a general audience — not at code level — and is built with plain HTML, CSS
and JavaScript served by a tiny zero-dependency Node.js server.

## Run it

```bash
cd website
node server.js              # → http://localhost:4173
# or pick a port:
PORT=8080 node server.js
```

No `npm install` is needed — the server uses only Node's built-in modules.

## What's inside

```
website/
├── server.js          # zero-dependency static file server
├── package.json       # npm start / npm run dev
├── summary.txt        # source text the content is based on
└── public/
    ├── index.html     # the single-page site
    ├── styles.css     # all styling + animations
    ├── main.js        # background canvas, scroll reveals, mode toggle, Cell Lab
    └── assets/
        └── dna.png    # AEONS logo
```

## Interactive bits

- **Generative background** — drifting bioluminescent "cells" linked by filaments.
- **Darwin / Lamarck toggle** — flips the page's accent palette and swaps the
  explanatory panels.
- **Scroll reveals & reading-progress bar.**

Screenshots and the heightmap/material-map visuals are placeholders for now.
