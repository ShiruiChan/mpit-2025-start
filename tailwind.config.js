export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui"] },
      colors: {
        drivee: {
          accent: "var(--accent)",
          accent600: "var(--accent-600)",
          bg: {
            light: "#f8fafb",
            dark:  "#0b0d0f",
          }
        }
      },
      boxShadow: {
        card: "0 6px 24px rgba(0,0,0,0.08)",
        cardDark: "0 6px 24px rgba(0,0,0,0.45)"
      },
      borderRadius: {
        xl2: "1.25rem"
      },
      keyframes: {
        shine: { "0%": { transform:"translateX(-100%)" }, "100%": { transform:"translateX(100%)" } },
        breathe: { "0%,100%": { opacity:.9 }, "50%": { opacity:1 } }
      },
      animation: {
        shine: "shine 1.8s linear infinite",
        breathe: "breathe 4s ease-in-out infinite"
      }
    },
  },
  plugins: [],
};
