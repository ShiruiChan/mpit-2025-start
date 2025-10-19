export default function OrderInfo({
  fromText,
  toText,
  price
}: { fromText: string; toText: string; price: number }) {
  return (
    <div className="rounded-[16px] border border-transparent bg-[color:var(--surface)] p-3 text-[color:var(--text-primary)]">

      <div className="flex items-start gap-3">
        <div className="h-10 w-10 rounded-full bg-neutral-200 dark:bg-neutral-800 flex items-center justify-center text-neutral-500">👤</div>

        <div className="flex-1 min-w-0">
          <div className="text-sm">
            <span className="font-semibold" style={{color:"var(--blue)"}}>A </span>{fromText}
          </div>
          <div className="text-sm mt-1">
            <span className="font-semibold" style={{color:"var(--green-strong)"}}>B </span>{toText}
          </div>

          <div className="mt-2 text-[18px] font-semibold" style={{color:"var(--green-strong)"}}>
            {price} ₽
          </div>

          <div>
            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-neutral-200/70 dark:bg-neutral-800/70 text-xs text-neutral-700 dark:text-neutral-300">
              Перевод
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
