import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/dashboard/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api/dashboard': 'http://127.0.0.1:18790',
    },
  },
  build: {
    outDir: 'dist',
  },
})
