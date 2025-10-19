import { useMemo, useState, useEffect } from "react";
import OrderInfo from "../components/OrderInfo";
import CustomPriceModal from "../components/CustomPriceModal";
import Spinner from "../components/Spinner";
import mock from "../data/mockData";
import Modal from "../components/Modal";

type Props = { onDecline?: () => void };
function decide(prob: number) {
  return Math.random() * 100 < prob;
}

export default function Orders({ onDecline }: Props) {
  const { fromText, toText, basePrice } = mock;

  /* настройки вероятностей */
  const [uplift0, setUplift0] = useState(0); // было 0%
  const [uplift1, setUplift1] = useState(5); // было +5%
  const [uplift2, setUplift2] = useState(10); // было +10%
  const [prob0, setProb0] = useState(60);
  const [prob5, setProb5] = useState(50);
  const [prob10, setProb10] = useState(40);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    try {
      const s = localStorage.getItem("drivee.uplifts");
      if (!s) return;
      const { u0, u1, u2 } = JSON.parse(s);
      if (typeof u0 === "number") setUplift0(u0);
      if (typeof u1 === "number") setUplift1(u1);
      if (typeof u2 === "number") setUplift2(u2);
    } catch {}
  }, []);
  useEffect(() => {
    localStorage.setItem(
      "drivee.uplifts",
      JSON.stringify({ u0: uplift0, u1: uplift1, u2: uplift2 })
    );
  }, [uplift0, uplift1, uplift2]);

  useEffect(() => {
    try {
      const s = localStorage.getItem("drivee.probs");
      if (!s) return;
      const { p0, p5, p10 } = JSON.parse(s);
      if (typeof p0 === "number") setProb0(p0);
      if (typeof p5 === "number") setProb5(p5);
      if (typeof p10 === "number") setProb10(p10);
    } catch {}
  }, []);
  useEffect(() => {
    localStorage.setItem(
      "drivee.probs",
      JSON.stringify({ p0: prob0, p5: prob5, p10: prob10 })
    );
  }, [prob0, prob5, prob10]);

  /* сценарии */
  const options = useMemo(() => {
  const p0 = Math.round(basePrice * (1 + uplift0 / 100));
  const p1 = Math.round(basePrice * (1 + uplift1 / 100));
  const p2 = Math.round(basePrice * (1 + uplift2 / 100));

  return [
    { id: "base",   name: uplift0 === 0 ? "Начальная" : `На ${uplift0}% выше`, value: p0, label: `${p0} ₽`,  prob: prob0,  style: "green" },
    { id: "p5",     name: `На ${uplift1}% выше`,                               value: p1, label: `${p1} ₽`,  prob: prob5,  style: "blue" },
    { id: "p10",    name: `На ${uplift2}% выше`,                               value: p2, label: `${p2} ₽`,  prob: prob10, style: "red"  },
    { id: "custom", name: "Своя цена",                                         value: NaN, label: "+",       prob: NaN,    style: ""     },
  ];
}, [basePrice, uplift0, uplift1, uplift2, prob0, prob5, prob10]);

  /* выбор/статусы */
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedPrice, setSelectedPrice] = useState<number | null>(null);
  const [selectedProb, setSelectedProb] = useState<number | null>(null);
  const [status, setStatus] = useState<
    "idle" | "sending" | "accepted" | "declined" | "error"
  >("idle");

  const selected = useMemo(
    () => options.find((o) => o.id === selectedId) ?? null,
    [options, selectedId]
  );

  const selectionText = useMemo(() => {
    if (!selected && selectedPrice == null) return null;
    const name  = selected?.name ?? "Своя цена";
    const price = selected?.value ?? selectedPrice!;
    return `Выбрано: “${name}” — ${price} ₽`;
  }, [selected, selectedPrice]);

  // const pForBar = useMemo(() => {
  //   const prob = selected?.prob ?? selectedProb ?? prob0;
  //   return Math.max(0, Math.min(1, prob / 100));
  // }, [selected, selectedProb, prob0]);

  // const primaryLabel = useMemo(() => {
  //   if (status === "sending") return "Отправка…";
  //   if (selectedPrice != null) return `Предложить ${selectedPrice} ₽`;
  //   return `Принять за ${basePrice} ₽`;
  // }, [status, selectedPrice, basePrice]);

  function toggleScenario(id: string) {
    if (id === "custom") {
      setSelectedId("custom");
      setStatus("idle");
      return;
    }
    if (selectedId === id) {
      setSelectedId(null);
      setSelectedPrice(null);
      setSelectedProb(null);
      setStatus("idle");
    } else {
      const opt = options.find((o) => o.id === id)!;
      setSelectedId(id);
      setSelectedPrice(opt.value);
      setSelectedProb(opt.prob);
      setStatus("idle");
    }
  }
  function onCustom(price: number) {
    setSelectedId("custom");
    setSelectedPrice(price);
    setSelectedProb(null);
    setStatus("idle");
  }
  async function onPrimary() {
    // const offer = selectedPrice ?? basePrice;
    const prob = selectedProb ?? prob0;
    setStatus("sending");
    try {
      await new Promise((res) => setTimeout(res, 900));
      const ok = decide(prob);
      setStatus(ok ? "accepted" : "declined");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="space-y-4 relative">
      {/* шестерёнка — в правом верхнем углу блока */}
      <button
        className="absolute right-0 -top-1 rounded-lg p-2 border border-neutral-200 bg-white shadow"
        onClick={() => setSettingsOpen(true)}
        title="Настройки сценариев"
        aria-label="Настройки сценариев"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="w-5 h-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.89 3.31.877 2.42 2.42a1.724 1.724 0 001.065 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.89 1.543-.877 3.31-2.42 2.42a1.724 1.724 0 00-2.573 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.89-3.31-.877-2.42-2.42A1.724 1.724 0 004.317 14.35c-1.756-.426-1.756-2.924 0-3.35.61-.148 1.1-.638 1.248-1.248.89-1.543-.877-3.31 2.42-2.42A1.724 1.724 0 0010.325 4.317z"
          />
        </svg>
      </button>

      {/* инфо: цена сверху не меняется */}
      <OrderInfo fromText={fromText} toText={toText} price={basePrice} />

      {/* подпись над полосой */}
      {selectionText && (
        <div className="text-sm text-neutral-600 -mt-2">{selectionText}</div>
      )}

      {/* полоса вероятности — БЕЗ текста снизу */}

      {/* чипы */}
      <div className="flex flex-wrap gap-2 mt-3">
        {options.map((opt) => (
          <button
            key={opt.id}
            onClick={() => toggleScenario(opt.id)}
            disabled={status === "sending"}
            className={`price-chip ${opt.style} ${
              selectedId === opt.id ? "active" : ""
            }`}
            aria-pressed={selectedId === opt.id}
            title={opt.name}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* статусы */}
      {status === "sending" && (
        <div className="flex items-center gap-2 text-sm text-neutral-500">
          <Spinner /> Отправка...
        </div>
      )}
      {status === "accepted" && (
        <div className="text-green-600 text-sm">Подтверждено заказчиком.</div>
      )}
      {status === "declined" && (
        <div className="text-orange-600 text-sm">Не подтверждено.</div>
      )}
      {status === "error" && (
        <div className="text-red-500 text-sm">Ошибка при отправке</div>
      )}

      {/* ЛИПКИЙ футер с кнопками — ВСЕГДА ВЛЕЗАЕТ */}
      <div className="sheet-actions">
        <div className="grid grid-cols-1 gap-3">
          <button
            className="btn-drivee w-full disabled:opacity-60"
            onClick={onPrimary}
            disabled={status === "sending"}
          >
            {status === "sending"
              ? "Отправка…"
              : selectedPrice != null
              ? `Предложить ${selectedPrice} ₽`
              : `Принять за ${basePrice} ₽`}
          </button>
          <button
            className="btn-ghost w-full text-red-600 border border-red-200 bg-white disabled:opacity-60"
            onClick={() => onDecline?.()}
            disabled={status === "sending"}
          >
            Отказаться
          </button>
        </div>
      </div>

      {/* Кастомная цена (модалка — через портал в #modal-root) */}
      <CustomPriceModal
        open={selectedId === "custom"}
        basePrice={basePrice}
        onClose={() => setSelectedId(null)}
        onConfirm={(price) => onCustom(price)}
      />

      {/* Настройки надбавок (модалка) */}
      <Modal open={settingsOpen} onClose={() => setSettingsOpen(false)} ariaLabel="Настройки сценариев">
        <div className="p-4">
          <div className="text-lg font-semibold mb-3">Настройки сценариев</div>
          <div className="space-y-3">
            <label className="flex items-center justify-between gap-3">
              <span className="text-sm text-neutral-700">Надбавка #1</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={-100} max={200} step={1}
                  value={uplift0}
                  onChange={e => setUplift0(Number(e.target.value))}
                  className="w-24 border rounded px-2 py-1 text-right"
                  aria-label="Надбавка 1 в процентах"
                />
                <span>%</span>
              </div>
            </label>

            <label className="flex items-center justify-between gap-3">
              <span className="text-sm text-neutral-700">Надбавка #2</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={-100} max={200} step={1}
                  value={uplift1}
                  onChange={e => setUplift1(Number(e.target.value))}
                  className="w-24 border rounded px-2 py-1 text-right"
                  aria-label="Надбавка 2 в процентах"
                />
                <span>%</span>
              </div>
            </label>

            <label className="flex items-center justify-between gap-3">
              <span className="text-sm text-neutral-700">Надбавка #3</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={-100} max={200} step={1}
                  value={uplift2}
                  onChange={e => setUplift2(Number(e.target.value))}
                  className="w-24 border rounded px-2 py-1 text-right"
                  aria-label="Надбавка 3 в процентах"
                />
                <span>%</span>
              </div>
            </label>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2">
            <button className="btn-ghost" onClick={() => setSettingsOpen(false)}>Отмена</button>
            <button className="btn-drivee" onClick={() => setSettingsOpen(false)}>Сохранить</button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
