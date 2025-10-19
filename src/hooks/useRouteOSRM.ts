import { useEffect, useState } from "react";
import type { LineString } from "geojson";

export function useRouteOSRM(from: [number, number], to: [number, number]) {
  const [route, setRoute] = useState<LineString | null>(null);
  const [distance, setDistance] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancel = false;
    const fetchRoute = async () => {
      setLoading(true);
      try {
        const url = `https://router.project-osrm.org/route/v1/driving/${from[0]},${from[1]};${to[0]},${to[1]}?overview=full&geometries=geojson`;
        const res = await fetch(url);
        const json = await res.json();

        const geometry: LineString | undefined = json.routes?.[0]?.geometry;
        const distanceMeters: number | undefined = json.routes?.[0]?.distance;
        if (!cancel) {
          setRoute(geometry ?? null);
          setDistance(distanceMeters ? distanceMeters / 1000 : null);
        }
      } catch {
        if (!cancel) {
          setRoute(null);
          setDistance(null);
        }
      } finally {
        if (!cancel) setLoading(false);
      }
    };
    fetchRoute();
    return () => {
      cancel = true;
    };
  }, [from[0], from[1], to[0], to[1]]);

  return { route, distance, loading };
}
