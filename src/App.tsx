import { useState, useCallback, useEffect } from "react";
import "./index.css";
import Orders from "./pages/Orders";
import MapInteractive from "./components/MapInteractive";
import CustomerView from "./pages/CustomerView";

type Role = "driver" | "customer";
type OrderStatus = "new" | "proposed" | "accepted" | "rejected";

type Order = {
  id: string;
  basePrice: number;
  proposedPrice: number | null;
  status: OrderStatus;
};

export default function App() {
  // 🌓 ТЕМА
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("drivee.theme") as "light" | "dark" | null;
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("drivee.theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(t => (t === "dark" ? "light" : "dark"));
  }, []);

  // 📍 Пример координат (карта остаётся как у тебя)
  const from = { lat: 62.028, lon: 129.734 };
  const to   = { lat: 62.042, lon: 129.720 };

  // 🔀 Роль
  const [role, setRole] = useState<Role>("driver");

  // 📦 Заказ (центральное состояние)
  const [order, setOrder] = useState<Order>({
    id: "order-1",
    basePrice: 300,
    proposedPrice: null,
    status: "new",
  });

  // Шторка (водитель)
  const [open, setOpen] = useState(true);
  const [hasOrder, setHasOrder] = useState(true);

  const openSheet  = useCallback(() => hasOrder && setOpen(true), [hasOrder]);
  const toggle     = useCallback(() => setOpen(v => !v), []);
  useEffect(() => {
    document.body.classList.toggle("scroll-lock", role === "driver" ? open : true);
    return () => document.body.classList.remove("scroll-lock");
  }, [open, role]);

  const handleDeclineOrder = useCallback(() => {
    setOpen(false);
    setHasOrder(false);
  }, []);

  // === СЦЕНАРИЙ ===
  // 1) Водитель предложил цену
  const handlePropose = useCallback((price: number) => {
    setOrder(o => ({ ...o, proposedPrice: price, status: "proposed" }));
  }, []);

  // 2) Заказчик принял/отклонил
  const handleDecision = useCallback((decision: "accept" | "reject") => {
    setOrder(o => ({ ...o, status: decision === "accept" ? "accepted" : "rejected" }));
    setRole("driver");
    setOpen(true);
  }, []);

  // Адаптивная высота шторки (оставил твою механику)
  useEffect(() => {
    const apply = () => {
      const h = window.innerHeight;
      let factor = 0.48;
      const isLandscape = window.matchMedia("(orientation: landscape)").matches;
      const isTablet    = window.matchMedia("(min-width: 768px)").matches;
      if (h <= 640) factor = 0.44;
      if (isLandscape && window.innerWidth <= 900) factor = 0.38;
      if (isTablet) factor = 0.40;
      document.documentElement.style.setProperty("--sheet-open-height", Math.round(h * factor) + "px");
    };
    apply();
    window.addEventListener("resize", apply);
    window.addEventListener("orientationchange", apply);
    return () => {
      window.removeEventListener("resize", apply);
      window.removeEventListener("orientationchange", apply);
    };
  }, []);

  return (
    <div className="h-full w-full relative isolate" style={{ backgroundColor: "var(--drivee-bg)", color: "var(--text-primary)" }}>
      {/* КАРТА */}
      <div className="absolute inset-0 z-0" id="map-layer">
        <MapInteractive from={from} to={to} height={window.innerHeight} />
      </div>

      {/* Верхняя пилюля */}
      {role === "driver" && hasOrder && (
        <div className="pointer-events-none absolute left-1/2 top-5 -translate-x-1/2 z-40">
          <div className="top-pill">
            {order.status === "proposed" ? "Ожидаем решение заказчика…" :
             order.status === "accepted" ? "Цена принята ✔" :
             order.status === "rejected" ? "Цена отклонена ✖" : "Новый заказ"}
          </div>
        </div>
      )}

      {/* Переключатель роли — ниже «Новый заказ», слева */}
      <div className="absolute z-40 left-4" style={{ top: "72px" }}>
        <div className="role-switch">
          <button
            className={`role-btn ${role === "driver" ? "active" : ""}`}
            onClick={() => setRole("driver")}
            title="Режим водителя"
          >
            🚗 Водитель
          </button>
          <button
            className={`role-btn ${role === "customer" ? "active" : ""}`}
            onClick={() => setRole("customer")}
            title="Режим заказчика"
          >
            🧑‍💼 Заказчик
          </button>
        </div>
      </div>

      {/* 🌓 Переключатель темы */}
      <div className="absolute top-4 right-4 z-40">
        <button
          onClick={toggleTheme}
          aria-label="Toggle theme"
          className="h-10 w-10 rounded-full border border-white/20 bg-[var(--surface-contrast)] text-white shadow flex items-center justify-center hover:opacity-90 active:translate-y-px"
          title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
        >
          {theme === "dark" ? "🌞" : "🌙"}
        </button>
      </div>

      {/* Вид водителя */}
      {role === "driver" && hasOrder && (
        <div className={`sheet ${open ? "sheet--open" : "sheet--hidden"}`} aria-hidden={!open}>
          <div className="sheet-card relative">
            <div className="sheet-handle" onClick={toggle}>
              <div className="sheet-handle-dot" />
            </div>

            <div className="px-4 mx-auto">
              <div
                className="mx-auto mb-2 inline-block px-4 py-2 text-center font-semibold rounded-2xl tracking-[.10em]"
                style={{ color: "var(--text-primary)" }}
              >
                Заказ
              </div>
            </div>

            <div className="sheet-scroll">
              <Orders
                onDecline={handleDeclineOrder}
                // пробрасываем колбэк предложения цены
                onPropose={handlePropose}
                // пробрасываем статус/последнюю цену, чтобы подсказки были умнее (необязательно)
                lastStatus={order.status}
              />
            </div>
          </div>
        </div>
      )}

      {/* Вид заказчика */}
      {role === "customer" && (
        <div className="sheet sheet--open" aria-hidden={false}>
          <div className="sheet-card relative">
            <div className="sheet-handle">
              <div className="sheet-handle-dot" />
            </div>

            <div className="px-3 mx-auto">
              <div
                className="mx-auto mb-1 inline-block px-3 py-1.5 text-center text-sm font-semibold rounded-xl tracking-wide"
                style={{ color: "var(--text-primary)" }}
              >
                Ваш заказ
              </div>
            </div>

            <div className="sheet-scroll">
              <CustomerView
                from={from}
                to={to}
                proposedPrice={order.proposedPrice}
                status={order.status}
                onAccept={() => handleDecision("accept")}
                onReject={() => handleDecision("reject")}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
