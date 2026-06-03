import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development, the dashboard calls /api/... and Vite proxies those calls to the
// FastAPI backend on :8000 (so we avoid CORS during dev). In production the frontend
// uses VITE_API_BASE (the deployed backend URL) — see src/api.js.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
