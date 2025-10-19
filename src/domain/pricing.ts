import type{ BidOption } from "./types";

export const expectedIncome = (price: number, p: number) => price * p;

export function bestByExpected(options: BidOption[]) {
  return options.slice().sort((a,b)=>b.expectedIncome - a.expectedIncome)[0];
}
