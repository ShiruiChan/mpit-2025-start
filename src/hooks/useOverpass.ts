import { useEffect, useState } from "react";
import { singleFlightNearestRoad } from "../lib/overpass";

export function useNearestRoad(lat: number, lon: number) {
  const [road, setRoad] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    singleFlightNearestRoad(lat, lon)
      .then(name => { if (!cancelled) setRoad(name); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [lat, lon]);

  return { road, loading };
}
