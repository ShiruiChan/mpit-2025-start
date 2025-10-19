export default function Spinner({ size=18 }: { size?: number }) {
  const s = `${size}px`;
  return (
    <svg viewBox="0 0 50 50" width={s} height={s} className="animate-spin">
      <circle cx="25" cy="25" r="20" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="6"/>
      <path d="M45 25a20 20 0 0 1-20 20" fill="none" stroke="currentColor" strokeWidth="6" strokeLinecap="round"/>
    </svg>
  );
}
