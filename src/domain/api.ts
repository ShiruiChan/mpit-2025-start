export type BidPayload = { orderId: string; price: number };

export async function sendBid(): Promise<{ ok: boolean; bidId?: string }> {
  const wait = 1200 + Math.random() * 1300;
  await new Promise(r => setTimeout(r, wait));
  const ok = Math.random() < 0.8;
  return ok ? { ok: true, bidId: `BID-${Math.random().toString(36).slice(2, 8)}` } : { ok: false };
}
