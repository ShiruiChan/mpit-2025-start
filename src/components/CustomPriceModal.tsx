import { useEffect, useMemo, useRef, useState } from "react";
import Modal from "./Modal";

type Props = {
  open: boolean;
  basePrice: number;
  onClose: () => void;
  onConfirm: (price: number) => void;
};

type Preset = { name: string; value: number };

export default function CustomPriceModal({ open, basePrice, onClose, onConfirm }: Props) {
  const [value, setValue] = useState<number>(basePrice);
  const inputRef = useRef<HTMLInputElement>(null);

  // Предустановленные сценарии
  const presets: Preset[] = useMemo(() => [
    { name: "Аккуратный",  value: Math.round((basePrice * 1.05) / 5) * 5 },   // +5%
    { name: "Оптимальный", value: Math.round((basePrice * 1.15) / 5) * 5 },   // +15%
    { name: "Смелый",      value: Math.round((basePrice * 1.30) / 5) * 5 },   // +30%
  ], [basePrice]);

  useEffect(() => {
    if (!open) return;
    setValue(basePrice);
    const timer = setTimeout(() => inputRef.current?.focus(), 50);
    return () => clearTimeout(timer);
  }, [open, basePrice]);

  const percentMarkup = useMemo(() => {
    if (!basePrice || !value) return 0;
    return Math.round(((value - basePrice) / basePrice) * 100);
  }, [value, basePrice]);

  const error = useMemo(() => {
    if (Number.isNaN(value)) return "Введите сумму";
    if (value < 5) return "Минимум 5 ₽";
    if (value % 5 !== 0) return "Сумма должна быть кратна 5 ₽";
    return "";
  }, [value]);

  const submit = () => {
    if (!error) onConfirm(value);
  };

  return (
    <Modal open={open} onClose={onClose} ariaLabel="Своя цена">
      <div className="p-4 modal-body" style={{ color: "var(--text-primary)" }}>
        {/* Заголовок */}
        <div className="text-lg font-semibold mb-1">Своя цена</div>
        <div className="caption mb-4">
          Базовая цена: <span className="font-semibold">{basePrice} ₽</span>
        </div>

        {/* Предустановленные варианты */}
        <div className="mb-3 grid grid-cols-3 gap-2">
          {presets.map((p, i) => {
            const up = Math.round(((p.value - basePrice) / basePrice) * 100);
            const tone = i === 0 ? "green" : i === 1 ? "blue" : "red"; // теперь красный, не orange
            const active = value === p.value ? "active" : "";
            return (
              <button
                key={p.name}
                className={`price-chip ${tone} ${active}`}
                onClick={() => setValue(p.value)}
                type="button"
              >
                <div className="leading-tight text-center">
                  <div>{p.value} ₽</div>
                  <div
                    className="text-[11px] opacity-90"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    +{up}%
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Поле ввода */}
        <label className="caption block mb-1" style={{ color: "var(--text-secondary)" }}>
          Другая сумма
        </label>
        <input
          ref={inputRef}
          type="number"
          inputMode="numeric"
          step={5}
          min={5}
          className="w-full rounded-xl px-3 py-2 mb-2 border"
          value={Number.isNaN(value) ? "" : value}
          onChange={(e) => setValue(e.target.value === "" ? NaN : Number(e.target.value))}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          style={{
            backgroundColor: "color-mix(in srgb, var(--surface-contrast) 10%, transparent)",
            color: "var(--text-primary)",
            borderColor: "color-mix(in srgb, var(--text-primary) 18%, transparent)",
          }}
        />

        {/* Пояснение или ошибка */}
        {error ? (
          <div className="text-xs mb-3" style={{ color: "var(--danger)" }}>
            {error}
          </div>
        ) : (
          <div className="text-xs caption mb-3" style={{ color: "var(--text-secondary)" }}>
            Надбавка:{" "}
            <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
              +{percentMarkup}%
            </span>{" "}
            — будет предложено{" "}
            <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
              {value} ₽
            </span>
          </div>
        )}

        {/* Кнопки */}
        <div className="grid grid-cols-2 gap-2">
          <button className="btn-ghost" onClick={onClose} type="button">
            Отмена
          </button>
          <button
            className="btn-drivee disabled:opacity-60"
            disabled={!!error}
            onClick={submit}
            type="button"
          >
            Предложить
          </button>
        </div>
      </div>
    </Modal>
  );
}
