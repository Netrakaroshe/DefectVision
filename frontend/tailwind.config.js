/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        status: {
          normal: '#10b981',       // Emerald 500
          'normal-dark': '#059669',  // Emerald 600
          'normal-bg': '#064e3b',    // Emerald 900
          defective: '#ef4444',      // Red 500
          'defective-dark': '#dc2626', // Red 600
          'defective-bg': '#7f1d1d',   // Red 900
          ready: '#0ea5e9',          // Sky 500
          unconfigured: '#f59e0b',   // Amber 500
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
