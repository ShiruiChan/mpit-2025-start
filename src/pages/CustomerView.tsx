import React, { useMemo } from "react";

type Props = {
  from: { lat: number; lon: number };
  to:   { lat: number; lon: number };
  proposedPrice: number | null;
  status: "new" | "proposed" | "accepted" | "rejected";
  onAccept: () => void;
  onReject: () => void;
};

export default function CustomerView({ from, to, proposedPrice, status, onAccept, onReject }: Props) {
  const eta = useMemo(() => "6–8 мин", []);
  const fromTxt = `${from.lat.toFixed(3)}, ${from.lon.toFixed(3)}`;
  const toTxt   = `${to.lat.toFixed(3)}, ${to.lon.toFixed(3)}`;

  const canDecide = status === "proposed" && proposedPrice !== null;

  return (
    <div className="rounded-[14px] bg-[color:var(--surface)] border border-transparent p-2.5">
      <p className="text-[13px] leading-[1.35]">
        <span className="font-semibold" style={{ color: "var(--blue)" }}>A </span>
        Откуда: {fromTxt}
      </p>
      <p className="text-[13px] leading-[1.35] mt-0.5">
        <span className="font-semibold" style={{ color: "var(--green-strong)" }}>B </span>
        Куда: {toTxt}
      </p>

      <p className="text-[12px] leading-[1.35] mt-0.5 text-[color:var(--text-secondary)]">
        Подача через {eta}
      </p>

      <div className="mt-1 text-[16px] font-semibold leading-[1.3]" style={{ color: "var(--green-strong)" }}>
        {proposedPrice !== null ? `${proposedPrice} ₽` : "Ожидаем предложение цены…"}
      </div>

      {canDecide ? (
        <div className="mt-3 flex items-center justify-center gap-2">
          <button className="btn-drivee" onClick={onAccept}>Принять</button>
          <button className="btn-decline" onClick={onReject}>Отказаться</button>
        </div>
      ) : (
        <div className="mt-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          {status === "accepted" ? "Вы приняли цену. Водитель едет к вам." :
           status === "rejected" ? "Вы отказались от цены. Ожидайте новое предложение." :
           "Ожидаем предложение цены от водителя…"}
        </div>
      )}
    </div>
  );
}
