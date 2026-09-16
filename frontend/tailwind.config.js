/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B0D10",
        paper: "#E9E5DA",
        line: "#30363D",
        slate: "#A6AFB8",
        signal: "#B7F36B",
        warning: "#F6B269",
        danger: "#FF7070",
      },
      letterSpacing: { terminal: ".16em" },
      boxShadow: { signal: "0 0 0 1px rgba(183,243,107,.16), 0 18px 60px rgba(0,0,0,.35)" },
    },
  },
  plugins: [],
};
