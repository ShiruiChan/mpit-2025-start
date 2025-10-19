import type{ BidOption } from "../domain/types";

export default function BidCard({
  title,
  subtitle,
  options,
  onPick
}: {
  title: string;
  subtitle: string;
  options: BidOption[];
  onPick: (opt: BidOption) => void;
}) {
  const best = options.slice().sort((a,b)=>b.expectedIncome-a.expectedIncome)[0];

  return (
    <div className="card p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <div className="text-xs text-neutral-500">{subtitle}</div>
        </div>
        <span className="badge">Рекомендовано</span>
      </div>

      <div className="mt-4 grid gap-2">
        {options.map(opt => {
          const isBest = opt === best;
          return (
            <button
              key={opt.label}
              onClick={()=>onPick(opt)}
              className={`group flex items-center justify-between w-full rounded-xl px-3 py-3 border text-left transition
                ${isBest
                  ? "border-[var(--accent)] ring-2 ring-[var(--accent)]/30 bg-[linear-gradient(90deg,rgba(50,215,75,0.08),transparent)]"
                  : "border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50/80 dark:hover:bg-white/5"
                }`}
            >
              <div className="flex flex-col">
                <div className="text-sm font-medium flex items-center gap-2">
                  {opt.label}
                  {isBest && <span className="badge">Оптимум</span>}
                </div>
                <div className="text-xs text-neutral-500">
                  Ожид. доход ≈ {Math.round(opt.expectedIncome)} ₽
                </div>
              </div>

              <div className="w-48">
                <div className="text-right text-sm font-semibold">{opt.price} ₽</div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="mt-4">
        <button className="btn-accent">
          Отправить бид по «{best.label}»
        </button>
      </div>
    </div>
  );
}
