export const BID_LABELS = ["Надёжный", "Оптимальный", "Смелый"] as const;
export type BidLabel = typeof BID_LABELS[number];

export type Order = {
  id: string;
  priceStart: number;
  distanceM: number;
  pickupM: number;
  userRating?: number;
};

export type BidOption = {
  label: BidLabel;
  price: number;
  pAccept: number;
  expectedIncome: number;
};
