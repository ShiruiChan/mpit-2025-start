import React, { useMemo, useState, useCallback } from "react";

type Pt = { lat: number; lon: number };
type ProviderKey = "yandex" | "osm_de" | "osm_zde" | "osm_ru";

function buildUrl(p: ProviderKey, from: Pt, to: Pt, center: Pt, zoom: number, w=450, h=260) {
  switch (p) {
    case "yandex":
      return `https://static-maps.yandex.ru/1.x/?lang=ru_RU&ll=${center.lon},${center.lat}&z=${zoom}&size=${w},${h}&pt=${from.lon},${from.lat},pm2blm~${to.lon},${to.lat},pm2grm&l=map`;
    case "osm_de":
      return `https://staticmap.openstreetmap.de/staticmap.php?center=${center.lat},${center.lon}&zoom=${zoom}&size=${w}x${h}&markers=${from.lat},${from.lon},lightblue1|${to.lat},${to.lon},green&maptype=mapnik`;
    case "osm_zde":
      return `https://z.overpass-api.de/tiles/staticmap.php?center=${center.lat},${center.lon}&zoom=${zoom}&size=${w}x${h}&markers=${from.lat},${from.lon},lightblue1|${to.lat},${to.lon},green`;
    case "osm_ru":
      return `https://tile.openstreetmap.org/cgi-bin/export?bbox=${center.lon-0.02},${center.lat-0.02},${center.lon+0.02},${center.lat+0.02}&scale=${zoom}&format=png`;
  }
}

export default function MapCard({
  from, to,
  zoom, onZoomChange,
  providersOrder = ["yandex","osm_de","osm_zde","osm_ru"]
}: {
  from: Pt;
  to: Pt;
  zoom: number;                             // ← управляется родителем
  onZoomChange: (z: number) => void;        // ← изменение зума наружу
  providersOrder?: ProviderKey[];
}) {
  const center = useMemo(
    () => ({ lat: (from.lat + to.lat) / 2, lon: (from.lon + to.lon) / 2 }),
    [from, to]
  );

  const [pi, setPi] = useState(0); // индекс текущего провайдера
  const url = buildUrl(providersOrder[pi], from, to, center, zoom);

  const handleError = useCallback(() => {
    setPi((i) => Math.min(i + 1, providersOrder.length - 1));
  }, [providersOrder.length]);

  const clamp = (z: number) => Math.max(3, Math.min(18, z));

  const onPlus  = () => onZoomChange(clamp(zoom + 1));
  const onMinus = () => onZoomChange(clamp(zoom - 1));

  const onWheel: React.WheelEventHandler<HTMLDivElement> = (e) => {
    // тачпад/колесо — меняем зум плавно
    if (e.deltaY < 0) onZoomChange(clamp(zoom + 1));
    else if (e.deltaY > 0) onZoomChange(clamp(zoom - 1));
  };

  const onDblClick: React.MouseEventHandler<HTMLDivElement> = () => {
    onZoomChange(clamp(zoom + 1));
  };

  return (
    <div
      className="rounded-[16px] overflow-hidden border border-neutral-200 dark:border-neutral-800 bg-neutral-200"
    >
      <div
        className="relative h-[220px] w-full select-none"
        onWheel={onWheel}
        onDoubleClick={onDblClick}
      >
        <img
          src={url}
          alt="Маршрут"
          className="w-full h-full object-cover"
          onError={handleError}
          referrerPolicy="no-referrer"
          draggable={false}
        />

        {/* зум-кнопки */}
        <div className="absolute right-2 top-2 flex flex-col gap-2">
          <button onClick={onPlus}  className="h-8 w-8 rounded-full bg-white/90 shadow text-xl leading-8">+</button>
          <button onClick={onMinus} className="h-8 w-8 rounded-full bg-white/90 shadow text-xl leading-8">−</button>
        </div>

        {/* демо-плашки */}
        <div className="absolute left-2 top-2 px-2 py-1 rounded-md text-xs font-medium text-white" style={{background:"#2aa84a"}}>
          20 мин. • 4,2 км
        </div>
        <div className="absolute right-10 top-3 px-2 py-1 rounded-md text-xs font-medium text-white" style={{background:"#2a75e6"}}>
          2 мин. • 331 м
        </div>

        {/* бейдж провайдера и текущего масштаба */}
        <div className="absolute bottom-2 right-2 text-[10px] px-2 py-1 rounded bg-black/50 text-white">
          {providersOrder[pi]} • z{zoom}
        </div>
      </div>
    </div>
  );
}
