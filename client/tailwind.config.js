/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        pitch: {
          900: "#0a1f14",
          800: "#0f2e1c",
          700: "#1a4d32",
          500: "#2d8f5c",
          300: "#6fcf97",
        },
      },
    },
  },
  plugins: [],
};
