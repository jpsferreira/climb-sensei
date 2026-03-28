// ClimbSensei Service Worker — offline caching
const CACHE_NAME = "climbsensei-v1";

// Shell assets to pre-cache on install
const SHELL_ASSETS = [
  "/",
  "/login",
  "/static/manifest.json",
  "/static/icon-192.svg",
  "/static/icon-512.svg",
];

// Install: pre-cache shell, then activate immediately
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting()),
  );
});

// Activate: clean old caches, then claim clients
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

// Fetch: network-only for API/auth, cache strategies for everything else
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== "GET") return;

  // Skip API, auth-protected, and upload routes — always network-only
  if (
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/upload") ||
    url.pathname.startsWith("/outputs/") ||
    url.pathname.startsWith("/download")
  ) {
    return;
  }

  // For CDN assets (fonts, chart.js) — cache-first
  // Cache opaque responses too since cross-origin resources may not have CORS
  if (url.origin !== self.location.origin) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) =>
          cached ||
          fetch(event.request).then((response) => {
            if (response.ok || response.type === "opaque") {
              const clone = response.clone();
              caches.open(CACHE_NAME).then((cache) => {
                cache.put(event.request, clone);
              });
            }
            return response;
          }),
      ),
    );
    return;
  }

  // Pages and static assets — stale-while-revalidate
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const fetchPromise = fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, clone);
            });
          }
          return response;
        })
        .catch(() => {
          // Network failed — return cached or offline fallback
          if (cached) return cached;
          // For HTML pages, return cached home page as fallback
          if (
            event.request.headers.get("accept") &&
            event.request.headers.get("accept").includes("text/html")
          ) {
            return caches
              .match("/")
              .then(
                (homeResponse) =>
                  homeResponse || new Response("Offline", { status: 503 }),
              );
          }
          return new Response("Offline", { status: 503 });
        });

      return cached || fetchPromise;
    }),
  );
});
