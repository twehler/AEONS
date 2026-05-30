#!/usr/bin/env node
/*
 * AEONS — minimal zero-dependency static web server.
 *
 * Serves the contents of ./public on http://localhost:PORT.
 * No npm install required; uses only Node's built-in modules.
 *
 *   node server.js            # serves on port 4173
 *   PORT=8080 node server.js  # custom port
 */

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 4173;
const ROOT = path.join(__dirname, "public");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

function safeResolve(urlPath) {
  // Strip query string and decode, then prevent directory traversal.
  const clean = decodeURIComponent(urlPath.split("?")[0]);
  const resolved = path.normalize(path.join(ROOT, clean));
  if (!resolved.startsWith(ROOT)) return null;
  return resolved;
}

const server = http.createServer((req, res) => {
  let target = safeResolve(req.url === "/" ? "/index.html" : req.url);

  if (!target) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.stat(target, (err, stat) => {
    if (!err && stat.isDirectory()) {
      target = path.join(target, "index.html");
    }

    fs.readFile(target, (readErr, data) => {
      if (readErr) {
        res.writeHead(404, { "Content-Type": "text/html; charset=utf-8" });
        res.end(
          "<h1 style='font-family:sans-serif'>404 &mdash; lost in the primordial soup</h1>"
        );
        return;
      }
      const ext = path.extname(target).toLowerCase();
      res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
      res.end(data);
    });
  });
});

server.listen(PORT, () => {
  console.log("\n  AEONS website running");
  console.log(`  →  http://localhost:${PORT}\n`);
});
