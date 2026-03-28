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

// Install: pre-cache shell
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)),
  );
  self.skipWaiting();
});

// Activate: clean old caches
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
      ),
  );
  self.clients.claim();
});

// Fetch: network-first for API/auth, stale-while-revalidate for pages/assets
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== "GET") return;

  // Skip API requests and auth — always go to network
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/upload")) {
    return;
  }

  // For CDN assets (fonts, chart.js) — cache-first
  if (url.origin !== self.location.origin) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) =>
          cached ||
          fetch(event.request).then((response) => {
            if (response.ok) {
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
            return caches.match("/");
          }
          return new Response("Offline", { status: 503 });
        });

      return cached || fetchPromise;
    }),
  );
});
