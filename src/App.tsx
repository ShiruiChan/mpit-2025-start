import { useState, useCallback, useEffect } from "react";
import "./index.css";
import Orders from "./pages/Orders";
import MapInteractive from "./components/MapInteractive";

export default function App() {
  // 🌓 ТЕМА
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("drivee.theme") as "light" | "dark" | null;
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme; // <html data-theme="dark|light">
    localStorage.setItem("drivee.theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(t => (t === "dark" ? "light" : "dark"));
  }, []);

  // Пример координат
  const from = { lat: 62.028, lon: 129.734 };
  const to   = { lat: 62.042, lon: 129.720 };

  // Состояния шторки и наличия заказа
  const [open, setOpen] = useState(true);
  const [hasOrder, setHasOrder] = useState(true);

  const openSheet  = useCallback(() => hasOrder && setOpen(true), [hasOrder]);
  const toggle     = useCallback(() => setOpen(v => !v), []);
  useEffect(() => {
    document.body.classList.toggle("scroll-lock", open);
    return () => document.body.classList.remove("scroll-lock");
  }, [open]);

  const handleDeclineOrder = useCallback(() => {
    setOpen(false);
    setHasOrder(false);
  }, []);

  return (
    <div className="h-full w-full relative isolate" style={{ backgroundColor: "var(--drivee-bg)", color: "var(--text-primary)" }}>
      {/* Портал для модалок */}
      {/* <div id="modal-root" className="absolute inset-0 z-[80]" /> */}

      {/* КАРТА */}
      <div className="absolute inset-0 z-0" id="map-layer">
        <MapInteractive from={from} to={to} height={window.innerHeight} />
      </div>

      {/* Верхняя пилюля */}
      {hasOrder && (
        <div className="pointer-events-none absolute left-1/2 top-5 -translate-x-1/2 z-40">
          <div className="top-pill">Новый заказ</div>
        </div>
      )}

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

      {/* FAB для открытия шторки */}
      {!open && hasOrder && (
        <div className="fab-open px-4 w-full max-w-md">
          <button className="btn-drivee w-full" onClick={openSheet}>
            Показать заказ
          </button>
        </div>
      )}

      {/* Шторка */}
      {hasOrder && (
        <div className={`sheet ${open ? "sheet--open" : "sheet--hidden"}`} aria-hidden={!open}>
          <div className="sheet-card relative">
            <div className="sheet-handle" onClick={toggle}>
              <div className="sheet-handle-dot" />
            </div>

            <div className="px-4 mx-auto">
              <div
                className="mx-auto mb-2 inline-block px-4 py-2 text-center font-semibold rounded-2xl tracking-[.10em]"
                style={{
                  color: "var(--text-primary)",
                }}
              >
                Заказ
              </div>
            </div>

            <div className="sheet-scroll">
              <Orders onDecline={handleDeclineOrder} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}