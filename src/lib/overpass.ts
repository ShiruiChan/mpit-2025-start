// Лёгкий клиент Overpass: зеркала + бэкофф + кэш
type Mirror = { name: string; url: string };

const MIRRORS: Mirror[] = [
  { name: "kumi",   url: "https://overpass.kumi.systems/api/interpreter" },
  { name: "de",     url: "https://overpass-api.de/api/interpreter" },
  { name: "zde",    url: "https://z.overpass-api.de/api/interpreter" },
  { name: "ru",     url: "https://overpass.openstreetmap.ru/api/interpreter" },
];

// in-memory кэш на сессию
const memCache = new Map<string, string | null>();

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

function keyFor(lat: number, lon: number, radius = 80) {
  const k = `${lat.toFixed(5)}:${lon.toFixed(5)}:${radius}`;
  return `overpass:nearestRoad:${k}`;
}

export async function fetchNearestRoadReliable(
  lat: number,
  lon: number,
  radius = 80,
  maxRetries = 3
): Promise<string | null> {
  const cacheKey = keyFor(lat, lon, radius);


  if (memCache.has(cacheKey)) return memCache.get(cacheKey) ?? null;

  try {
    const ls = localStorage.getItem(cacheKey);
    if (ls !== null) {
      memCache.set(cacheKey, ls === "null" ? null : ls);
      return ls === "null" ? null : ls;
    }
  } catch {}

  const body = new URLSearchParams({
    data: `
      [out:json][timeout:15];
      way(around:${radius},${lat},${lon})["highway"]["name"];
      out tags 1;
    `.trim(),
  });

  let lastError: unknown = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    for (const mirror of MIRRORS) {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 15000);

        const resp = await fetch(mirror.url, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
          body,
          signal: ctrl.signal,
        });

        clearTimeout(t);

        if (resp.status === 429 || resp.status === 503) {
          // перегруз — пробуем следующее зеркало
          lastError = new Error(`${mirror.name} ${resp.status}`);
          continue;
        }
        if (!resp.ok) {
          lastError = new Error(`${mirror.name} ${resp.status}`);
          continue;
        }

        const json = await resp.json();
        const name: string | undefined = json?.elements?.[0]?.tags?.name ?? json?.elements?.[0]?.tags?.["name:ru"];
        const result = name ?? null;

        memCache.set(cacheKey, result);
        try { localStorage.setItem(cacheKey, result === null ? "null" : result); } catch {}
        return result;
      } catch (e) {
        lastError = e;
        continue;
      }
    }

    const backoff = 400 * Math.pow(2, attempt);
    await sleep(backoff + Math.random() * 200);
  }

  console.warn("Overpass failed; last error:", lastError);
  memCache.set(cacheKey, null);
  try { localStorage.setItem(cacheKey, "null"); } catch {}
  return null;
}
const inflight = new Map<string, Promise<string | null>>();
export function singleFlightNearestRoad(lat: number, lon: number) {
  const k = keyFor(lat, lon);
  if (!inflight.has(k)) {
    inflight.set(k, fetchNearestRoadReliable(lat, lon).finally(() => inflight.delete(k)));
  }
  return inflight.get(k)!;
}
