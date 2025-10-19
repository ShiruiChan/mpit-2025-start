import type{ Order, BidOption, BidLabel } from "./types";
import { BID_LABELS } from "./types";

export function predictPAccept(bid: number, order: Order): number {
  const base = order.priceStart;
  const rel = (bid - base) / Math.max(1, base);
  let p = 0.9 * Math.exp(-2.2 * Math.max(0, rel));
  if (order.userRating && order.userRating < 4) p *= 0.9;
  if (order.pickupM > 3000) p *= 0.85;
  return Math.max(0.02, Math.min(0.98, p));
}

export function proposeOptions(order: Order): BidOption[] {
  const base = order.priceStart;

  const candidates: ReadonlyArray<{ label: BidLabel; price: number }> = [
    { label: BID_LABELS[0], price: Math.round(base * 1.05) },
    { label: BID_LABELS[1], price: Math.round(base * 1.15) },
    { label: BID_LABELS[2], price: Math.round(base * 1.30) },
  ] as const;

  return candidates.map(({ label, price }) => {
    const p = predictPAccept(price, order);
    const expectedIncome = price * p;
    return { label, price, pAccept: p, expectedIncome };
  });
}
