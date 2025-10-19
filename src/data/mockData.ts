// mockData.ts

export type BidStyle = "green" | "blue" | "red" | "gray";

export type BidOption = {
  id: string;
  name: string;        // короткое название стратегии (Осторожный / Баланс / Смелый / Экстремальный)
  label: string;       // подпись на чипе (например, "315 ₽")
  value: number;       // цена в ₽
  style: BidStyle;     // стилизация чипа
  probability: number; // вероятность принятия (0..100)
};

export type OrderMock = {
  fromText: string;
  toText: string;
  basePrice: number;
  options: BidOption[];
};

const mock: OrderMock = {
  fromText: "A какое-то место\nАОРДЛОАРОДЛОДОАДОАД",
  toText:   "B какое-то место\nАРОДЛАОАДЛОАДЛОАДЛОАД",
  basePrice: 300,
  options: [
    {
      id: "safe",
      name: "Осторожный",
      label: "295 ₽",
      value: 295,
      style: "gray",
      probability: 72,
    },
    {
      id: "balanced",
      name: "Баланс",
      label: "315 ₽",
      value: 315,
      style: "green",
      probability: 61,
    },
    {
      id: "bold",
      name: "Смелый",
      label: "345 ₽",
      value: 345,
      style: "blue",
      probability: 47,
    },
    {
      id: "extreme",
      name: "Экстремальный",
      label: "395 ₽",
      value: 395,
      style: "red",
      probability: 31,
    },
  ],
};

export default mock;
