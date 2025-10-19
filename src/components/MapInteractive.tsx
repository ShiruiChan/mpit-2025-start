import React from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Tooltip,
  useMap,
  ZoomControl
} from "react-leaflet";
import { useRouteOSRM } from "../hooks/useRouteOSRM";
import type { LineString, Position } from "geojson";

type Pt = { lat: number; lon: number };

function FitBounds({ from, to }: { from: Pt; to: Pt }) {
  const map = useMap();
  React.useEffect(() => {
    map.fitBounds(
      [
        [from.lat, from.lon],
        [to.lat, to.lon],
      ],
      { padding: [24, 24] }
    );
  }, [from, to, map]);
  return null;
}

// безопасный конвертер GeoJSON LineString -> [lat, lon][]
function toLatLngPath(geometry: LineString | null | undefined): [number, number][] {
  if (!geometry?.coordinates?.length) return [];
  return (geometry.coordinates as Position[]).map((pos) => {
    const lon = Number(pos[0] ?? 0);
    const lat = Number(pos[1] ?? 0);
    return [lat, lon] as [number, number];
  });
}

export default function MapInteractive({
  from,
  to,
  height = 300,
}: {
  from: Pt;
  to: Pt;
  height?: number;
}) {
  const { route } = useRouteOSRM([from.lon, from.lat], [to.lon, to.lat]);

  const path: [number, number][] = React.useMemo(() => {
    const routed = toLatLngPath(route);
    return routed.length ? routed : [[from.lat, from.lon], [to.lat, to.lon]];
  }, [route, from.lat, from.lon, to.lat, to.lon]);

  return (
    <div className="rounded-[16px] overflow-hidden border border-neutral-200 dark:border-neutral-800 relative">
      <MapContainer
        style={{ height }}
        center={[from.lat, from.lon]}
        zoom={13}
        scrollWheelZoom
        doubleClickZoom
        className="w-full relative"
      zoomControl={false}         // отключаем дефолт (слева-сверху)
    >
      <ZoomControl position="bottomright" />  {/* <— вправо-вниз */}
        <div className="absolute w-19 h-4 bg-white z-[9999] bottom-0 right-23 text-center">
            <a className="text-slate-700">Melunai</a>
        </div>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        <FitBounds from={from} to={to} />

        <Marker position={[from.lat, from.lon]}>
          <Tooltip direction="top" offset={[0, -8]} opacity={1} permanent>
            A
          </Tooltip>
        </Marker>
        <Marker position={[to.lat, to.lon]}>
          <Tooltip direction="top" offset={[0, -8]} opacity={1} permanent>
            B
          </Tooltip>
        </Marker>

        <Polyline positions={path} color="#32d74b" weight={5} opacity={0.8} />
      </MapContainer>
    </div>
  );
}
