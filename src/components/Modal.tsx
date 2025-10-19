
import { useEffect } from "react";
import { createPortal } from "react-dom";

type Props = {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
  ariaLabel?: string;
};

export default function Modal({ open, onClose, children, ariaLabel = "Диалог" }: Props) {
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  if (!open) return null;

  const node = (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      className="fixed inset-0 z-[80] flex items-center justify-center"
      onClick={onClose}
    >
      <div className="absolute inset-0 modal-backdrop" />
      <div
        className="relative z-[81] w-[min(92vw,360px)] max-h-[82vh] overflow-auto modal-card rounded-2xl shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
